import tequila as tq
import numpy as np
from tequila.objective.objective import Variable
import openfermion
from hacked_openfermion_qubit_operator import ParamQubitHamiltonian
from typing import Union
from vqe_utils import convert_PQH_to_tq_QH, convert_tq_QH_to_PQH,\
                      fold_unitary_into_hamiltonian
from grad_hacked import grad
from scipy_optimizer import minimize
import scipy

import random # guess we could substitute that with numpy.random #TODO
import argparse
import pickle as pk

import sys
sys.path.insert(0, '../../../../')
#sys.path.insert(0, '../')

# This needs to be properly integrated somewhere else at some point
# from tn_update.qq import contract_energy, optimize_wavefunctions
from tn_update.my_mpo import *
from tn_update.wfn_optimization import contract_energy_mpo, optimize_wavefunctions_mpo, initialize_wfns_randomly


def energy_from_wfn_opti(hamiltonian: tq.QubitHamiltonian, n_qubits: int, guess_wfns=None, TOL=1e-4) -> tuple:
    '''
    Get energy using a tensornetwork based method ~ power method (here as blackbox)
    '''
    d = int(2**(n_qubits/2))
    # Build MPO
    h_mpo = MyMPO(hamiltonian=hamiltonian, n_qubits=n_qubits, maxdim=500)
    h_mpo.make_mpo_from_hamiltonian()
    # Optimize wavefunctions based on random guess
    out = None
    it = 0
    while out is None:
        # Set up random initial guesses for subsystems
        it += 1
        if guess_wfns is None:
            psiL_rand = initialize_wfns_randomly(d, n_qubits//2)
            psiR_rand = initialize_wfns_randomly(d, n_qubits//2)
        else:
            psiL_rand = guess_wfns[0]
            psiR_rand = guess_wfns[1]
        out = optimize_wavefunctions_mpo(h_mpo,
                                         psiL_rand,
                                         psiR_rand,
                                         TOL=TOL,
                                         silent=True)
    energy_rand = out[0]
    optimal_wfns = [out[1], out[2]]

    return energy_rand, optimal_wfns

def combine_two_clusters(H: tq.QubitHamiltonian, subsystems: list, circ_list: list) -> float:
    '''
    E = nuc_rep + \sum_j c_j <0_A | U_A+ sigma_j|A U_A | 0_A><0_B | U_B+ sigma_j|B U_B | 0_B>
    i  ) Split up Hamiltonian into vector c = [c_j], all sigmas of system A and system B
    ii ) Two vectors of ExpectationValues E_A=[E(U_A, sigma_j|A)], E_B=[E(U_B, sigma_j|B)]
    iii) Minimize vectors from ii)
    iv ) Perform weighted sum \sum_j c_[j] E_A[j] E_B[j]
    result = nuc_rep + iv)

    Finishing touch inspired by private tequila-repo / pair-separated objective
    This is still rather inefficient/unoptimized
    Can still prune out near-zero coefficients c_j
    '''
    objective = 0.0
    n_subsystems = len(subsystems)

    # Over all Paulistrings in the Hamiltonian
    for p_index, pauli in enumerate(H.paulistrings):
        # Gather coefficient
        coeff = pauli.coeff.real
        # Empty dictionary for operations:
        # to be filled with another dictionary per subsystem, where then
        # e.g. X(0)Z(1)Y(2)X(4) and subsystems=[[0,1,2],[3,4,5]]
        #    -> ops={ 0: {0: 'X', 1: 'Z', 2: 'Y'}, 1: {4: 'X'} }
        ops = {}
        for s_index, sys in enumerate(subsystems):
            for k, v in pauli.items():
                if k in sys:
                    if s_index in ops:
                        ops[s_index][k] = v
                    else:
                        ops[s_index] = {k: v}
        # If no ops gathered -> identity -> nuclear repulsion
        if len(ops) == 0:
            #print ("this should only happen ONCE")
            objective += coeff
        # If not just identity:
        # add objective += c_j * prod_subsys( < Paulistring_j_subsys >_{U_subsys} )
        elif len(ops) > 0:
            obj_tmp = coeff
            for s_index, sys_pauli in ops.items():
                 #print (s_index, sys_pauli)
                 H_tmp = QubitHamiltonian.from_paulistrings(PauliString(data=sys_pauli))
                 E_tmp = ExpectationValue(U=circ_list[s_index], H=H_tmp)
                 obj_tmp *= E_tmp
            objective += obj_tmp

    initial_values = {k: 0.0 for k in objective.extract_variables()}
    random_initial_values = {k: 1e-2*np.random.uniform(-1, 1) for k in objective.extract_variables()}
    method = 'bfgs'  # 'bfgs'  # 'l-bfgs-b'  # 'cobyla'  # 'slsqp'
    curr_res = tq.minimize(method=method, objective=objective, initial_values=initial_values,
                           gradient='two-point', backend='qulacs',
                           method_options={"finite_diff_rel_step": 1e-3})

    return curr_res.energy

def energy_from_tq_qcircuit(hamiltonian: Union[tq.QubitHamiltonian,
                                               ParamQubitHamiltonian],
                            n_qubits: int,
                            circuit = Union[list, tq.QCircuit],
                            subsystems: list = [[0,1,2,3,4,5]],
                            initial_guess = None)-> tuple:
    '''
    Get minimal energy using tequila
    '''
    result = None
    # If only one circuit handed over, just run simple VQE
    if isinstance(circuit, list) and len(circuit) == 1:
         circuit = circuit[0]
    if isinstance(circuit, tq.QCircuit):
        if isinstance(hamiltonian, tq.QubitHamiltonian):
            E = tq.ExpectationValue(H=hamiltonian, U=circuit)
            if initial_guess is None:
                initial_angles = {k: 0.0 for k in E.extract_variables()}
            else:
                initial_angles = initial_guess
            #print ("optimizing non-param...")
            result = tq.minimize(objective=E, method='l-bfgs-b', silent=True, backend='qulacs',
                                  initial_values=initial_angles)
                                  #gradient='two-point', backend='qulacs',
                                  #method_options={"finite_diff_rel_step": 1e-4})
        elif isinstance(hamiltonian, ParamQubitHamiltonian):
            if initial_guess is None:
                raise Exception("Need to provide initial guess for this to work.")
                initial_angles = None
            else:
                initial_angles = initial_guess
            #print ("optimizing param...")
            result = minimize(hamiltonian, circuit, method='bfgs', initial_values=initial_angles,  backend="qulacs", silent=True)
    # If more circuits handed over, assume subsystems
    else:
        # TODO!! implement initial guess for the combine_two_clusters thing
        #print ("Should not happen for now...")
        result = combine_two_clusters(hamiltonian, subsystems, circuit)

    return result.energy, result.angles

def mixed_optimization(hamiltonian: ParamQubitHamiltonian, n_qubits: int, initial_guess: Union[dict, list]=None, init_angles: list=None):
    '''
    Minimizes energy using wfn opti and a parametrized Hamiltonian
    min_{psi, theta fixed} <psi | H(theta) | psi> --> min_{t, p fixed} <p | H(t) | p>
                          ^---------------------------------------------------^
                                 until convergence
    '''
    energy, optimal_state = None, None
    H_qh = convert_PQH_to_tq_QH(hamiltonian)
    var_keys, H_derivs = H_qh._construct_derivatives()
    print("var keys", var_keys, flush=True)
    print('var dict', init_angles, flush=True)
    # if not init_angles:
    #     var_vals = [0. for i in range(len(var_keys))] # initialize with 0
    # else:
    #     var_vals = init_angles

    def build_variable_dict(keys, values):
        assert(len(keys)==len(values))
        out = dict()
        for idx, key in enumerate(keys):
            out[key] = complex(values[idx])
        return out

    var_dict = init_angles
    var_vals = [*init_angles.values()]
    assert(build_variable_dict(var_keys, var_vals) == var_dict)

    def wrap_energy_eval(psi):
        ''' like energy_from_wfn_opti but instead of optimize
            get inner product '''
        def energy_eval_fn(x):
            var_dict = build_variable_dict(var_keys, x)
            H_qh_fix = H_qh(var_dict)
            H_mpo = MyMPO(hamiltonian=H_qh_fix, n_qubits=n_qubits, maxdim=500)
            H_mpo.make_mpo_from_hamiltonian()
            return contract_energy_mpo(H_mpo, psi[0], psi[1])
        return energy_eval_fn

    def wrap_gradient_eval(psi):
       ''' call derivatives with updated variable list '''
       def gradient_eval_fn(x):
           variables = build_variable_dict(var_keys, x)
           deriv_expectations = H_derivs.values()  # list of ParamQubitHamiltonian's
           deriv_qhs = [convert_PQH_to_tq_QH(d) for d in deriv_expectations]
           deriv_qhs = [d(variables) for d in deriv_qhs]  # list of tq.QubitHamiltonian's
           # print(deriv_qhs)
           deriv_mpos = [MyMPO(hamiltonian=d, n_qubits=n_qubits, maxdim=500)\
                         for d in deriv_qhs]
           # print(deriv_mpos[0].n_qubits)
           for d in deriv_mpos:
               d.make_mpo_from_hamiltonian()
           # deriv_mpos = [d.make_mpo_from_hamiltonian() for d in deriv_mpos]
           return [contract_energy_mpo(d, psi[0], psi[1]) for d in deriv_mpos]
       return gradient_eval_fn


    def do_wfn_opti(values, guess_wfns, TOL):
        # H_qh = H_qh(H_vars)
        var_dict = build_variable_dict(var_keys, values)
        en, psi = energy_from_wfn_opti(H_qh(var_dict), n_qubits=n_qubits,
                                       guess_wfns=guess_wfns, TOL=TOL)
        return en, psi

    def do_param_opti(psi, x0):
        result = scipy.optimize.minimize(fun=wrap_energy_eval(psi),
                                     jac=wrap_gradient_eval(psi),
                                     method='bfgs',
                                     x0=x0,
                                     options={'maxiter': 4})
        # print(result)
        return result

    e_prev, psi_prev = 123., None
    # print("iguess", initial_guess)
    e_curr, psi_curr = do_wfn_opti(var_vals, initial_guess, 1e-5)
    print('first eval', e_curr, flush=True)

    def converged(e_prev, e_curr, TOL=1e-4):
        return True if np.abs(e_prev-e_curr) < TOL else False

    it = 0
    var_prev = var_vals
    print("vars before comp", var_prev, flush=True)
    while not converged(e_prev, e_curr) and it < 50:
        e_prev, psi_prev = e_curr, psi_curr
        # print('before param opti')
        res = do_param_opti(psi_curr, var_prev)
        var_curr = res['x']
        print('curr vars', var_curr, flush=True)
        e_curr = res['fun']
        print('en before wfn opti', e_curr, flush=True)
        # print("iiiiiguess", psi_prev)
        e_curr, psi_curr = do_wfn_opti(var_curr, psi_prev, 1e-3)
        print('en after wfn opti', e_curr, flush=True)
        it += 1
        print('at iteration', it, flush=True)

    return e_curr, psi_curr

    # optimize parameters with fixed wavefunction
    # define/wrap energy function - given |p>, evaluate <p|H(t)|p>

    '''
    def wrap_gradient(objective: typing.Union[Objective, QTensor], no_compile=False, *args, **kwargs):
        def gradient_fn(variable: Variable = None):
            return grad(objective: typing.Union[Objective, QTensor], variable: Variable = None, no_compile=False, *args, **kwargs)
        return grad_fn
    '''

def minimize_energy(hamiltonian: Union[ParamQubitHamiltonian, tq.QubitHamiltonian], n_qubits: int, type_energy_eval: str='wfn', cluster_circuit: tq.QCircuit=None, initial_guess=None, initial_mixed_angles: dict=None) -> float:
    '''
    Minimizes energy functional either according a power-method inspired shortcut ('wfn')
    or using a unitary circuit ('qc')
    '''
    if type_energy_eval == 'wfn':
        if isinstance(hamiltonian, tq.QubitHamiltonian):
            energy, optimal_state = energy_from_wfn_opti(hamiltonian, n_qubits, initial_guess)
        elif isinstance(hamiltonian, ParamQubitHamiltonian):
            init_angles = initial_mixed_angles
            energy, optimal_state = mixed_optimization(hamiltonian=hamiltonian, n_qubits=n_qubits, initial_guess=initial_guess, init_angles=init_angles)
    elif type_energy_eval == 'qc':
        if cluster_circuit is None:
            raise Exception("Need to hand over circuit!")
        energy, optimal_state  = energy_from_tq_qcircuit(hamiltonian, n_qubits, cluster_circuit, [[0,1,2,3],[4,5,6,7]], initial_guess)
    else:
        raise Exception("Option not implemented!")

    return energy, optimal_state

import tequila as tq
import numpy as np
from tequila.objective.objective import Variable
import openfermion
from typing import Union
from vqe_utils import convert_PQH_to_tq_QH, convert_tq_QH_to_PQH,\
                      fold_unitary_into_hamiltonian

from energy_optimization import *
from do_annealing import *

import random # guess we could substitute that with numpy.random #TODO
import argparse
import pickle as pk


import matplotlib.pyplot as plt

# Main hyperparameters
mu = 2.0         # lognormal dist mean
sigma = 0.4      # lognormal dist std
T_0 = 1.0        # T_0, initial starting temperature
# max_epochs = 25 # num_epochs, max number of iterations
max_epochs = 20 # num_epochs, max number of iterations
min_epochs = 2  # num_epochs, max number of iterations5
# min_epochs = 20  # num_epochs, max number of iterations
tol = 1e-6       # minimum change in energy required, otherwise terminate optimization
actions_ratio = [.15, .6, .25]
patience = 100
beta = 0.5
max_non_cliffords = 0
type_energy_eval='wfn'
num_population = 24
num_offsprings = 20
num_processors = 8

#tuning, input as lists.
# parser.add_argument('--alphas', type = float, nargs = '+', help='alpha, temperature decay', required=True)
alphas = [0.9] # alpha, temperature decay'

#Required
be = False
h2 = True
n2 = False
name_prefix = '../../data/'

# Construct qubit-Hamiltonian
#num_active = 3
#active_orbitals = list(range(num_active))
if be:
    R1 = rrrr
    R2 = 1.6
    geometry = "be 0.0 0.0 0.0\nh 0.0 0.0 {R1}\nh 0.0 0.0 -{R2}"
    name = "/h/292/philipps/software/moldata/beh2/beh2_{R1:2.2f}_{R2:2.2f}_180" # adapt of you move data
    mol = tq.Molecule(name=name.format(R1=R1, R2=R2), geometry=geometry.format(R1=R1,R2=R2), n_pno=None)
    lqm = mol.local_qubit_map(hcb=False)
    H = mol.make_hamiltonian().map_qubits(lqm).simplify()
    #print("FCI ENERGY: ", mol.compute_energy(method="fci"))
    U_spa = mol.make_upccgsd_ansatz(name="SPA").map_qubits(lqm)
    U = U_spa
elif n2:
    R1 = rrrr
    geometry = "N 0.0 0.0 0.0\nN 0.0 0.0 {R1}"
    # name = "/home/abhinav/matter_lab/moldata/n2/n2_{R1:2.2f}" # adapt of you move data
    name = "/h/292/philipps/software/moldata/n2/n2_{R1:2.2f}" # adapt of you move data
    mol = tq.Molecule(name=name.format(R1=R1), geometry=geometry.format(R1=R1), n_pno=None)
    lqm = mol.local_qubit_map(hcb=False)
    H = mol.make_hamiltonian().map_qubits(lqm).simplify()
    # print("FCI ENERGY: ", mol.compute_energy(method="fci"))
    print("Skip FCI calculation, take old data...")
    U_spa = mol.make_upccgsd_ansatz(name="SPA").map_qubits(lqm)
    U = U_spa
elif h2:
    active_orbitals = list(range(3))
    mol = tq.Molecule(geometry='H 0.0 0.0 0.0\n H 0.0 0.0 1.6', basis_set='6-31g',
                      active_orbitals=active_orbitals, backend='psi4')
    H = mol.make_hamiltonian().simplify()
    print("FCI ENERGY: ", mol.compute_energy(method="fci"))
    U = mol.make_uccsd_ansatz(trotter_steps=1)

# Alternatively, get qubit-Hamiltonian from openfermion/externally
# with open("ham_6qubits.txt") as hamstring_file:
"""if be:
    with open("ham_beh2_3_3.txt") as hamstring_file:
        hamstring = hamstring_file.read()
    #print(hamstring)
    H = tq.QubitHamiltonian().from_string(string=hamstring, openfermion_format=True)
elif h2:
    with open("ham_6qubits.txt") as hamstring_file:
        hamstring = hamstring_file.read()
    #print(hamstring)
    H = tq.QubitHamiltonian().from_string(string=hamstring, openfermion_format=True)"""
n_qubits = len(H.qubits)

'''
Option 'wfn': "Shortcut" via wavefunction-optimization using MPO-rep of Hamiltonian

Option  'qc': Need to input a proper quantum circuit
'''

# Clifford optimization in the context of reduced-size quantum circuits
starting_E = 123.456
reference = True
if reference:
    if type_energy_eval.lower() == 'wfn':
        starting_E, _ = minimize_energy(hamiltonian=H, n_qubits=n_qubits, type_energy_eval='wfn')
        print('starting energy wfn: {:.5f}'.format(starting_E), flush=True)
    elif type_energy_eval.lower() == 'qc':
        starting_E, _ = minimize_energy(hamiltonian=H, n_qubits=n_qubits, type_energy_eval='qc', cluster_circuit=U)
        print('starting energy spa: {:.5f}'.format(starting_E))
    else:
        raise Exception("type_energy_eval must be either 'wfn' or 'qc', but is", type_energy_eval)
    starting_E, _ = minimize_energy(hamiltonian=H, n_qubits=n_qubits, type_energy_eval='wfn')



"""for alpha in alphas:
    print('Starting optimization, alpha = {:3f}'.format(alpha))

    # print('Energy to beat', minimize_energy(H, n_qubits, 'wfn')[0])
    simulated_annealing(hamiltonian=H, num_population=num_population,
                        num_offsprings=num_offsprings,
                        num_processors=num_processors,
                        tol=tol, max_epochs=max_epochs,
                        min_epochs=min_epochs, T_0=T_0, alpha=alpha,
                        actions_ratio=actions_ratio,
                        max_non_cliffords=max_non_cliffords,
                        verbose=True, patience=patience, beta=beta,
                        type_energy_eval=type_energy_eval.lower(),
                        cluster_circuit=U,
                        starting_energy=starting_E)"""



alter_cliffords = True
if alter_cliffords:
    print("starting to replace cliffords with non-clifford gates to see if that improves the current fitness")
    replace_cliff_with_non_cliff(hamiltonian=H, num_population=num_population,
                        num_offsprings=num_offsprings,
                        num_processors=num_processors,
                        tol=tol, max_epochs=max_epochs,
                        min_epochs=min_epochs, T_0=T_0, alpha=alphas[0],
                        actions_ratio=actions_ratio,
                        max_non_cliffords=max_non_cliffords,
                        verbose=True, patience=patience, beta=beta,
                        type_energy_eval=type_energy_eval.lower(),
                        cluster_circuit=U,
                        starting_energy=starting_E)

    # accepted_E, tested_E, decisions, instructions_list, best_instructions, temp, best_E = simulated_annealing(hamiltonian=H,
    #                                   population=4,
    #                                   num_mutations=2,
    #                                   num_processors=1,
    #                                   tol=opt.tol, max_epochs=opt.max_epochs,
    #                                   min_epochs=opt.min_epochs, T_0=opt.T_0, alpha=alpha,
    #                                   mu=opt.mu, sigma=opt.sigma, verbose=True, prev_E = starting_E, patience = 10, beta = 0.5,
    #                                   energy_eval='qc',
    #                                   eval_circuit=U

    # print(best_E)
    # print(best_instructions)

    # print(len(accepted_E))
    # plt.plot(accepted_E)
    # plt.show()

    # #pickling data
    # data = {'accepted_energies': accepted_E, 'tested_energies': tested_E,
    #         'decisions': decisions, 'instructions': instructions_list,
    #         'best_instructions': best_instructions, 'temperature': temp,
    #         'best_energy': best_E}

    # f_name = '{}{}_alpha_{:.2f}.pk'.format(opt.name_prefix, r, alpha)
    # pk.dump(data, open(f_name, 'wb'))
    # print('Finished optimization, data saved in {}'.format(f_name))

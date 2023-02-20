import numpy as np
import tequila as tq
import tensornetwork as tn
from tensornetwork.backends.abstract_backend import AbstractBackend
tn.set_default_backend("jax")

import torch

import itertools
import copy

import sys
from my_mpo import *


def normalize(me, order=2):
    return me/np.linalg.norm(me, ord=order)

# Computes <psiL | H | psiR>
def contract_energy(H, psiL, psiR) -> float:
    energy = 0

    # For test:
    # en_einsum = np.einsum('ijkl, i, j, k, l', H, psiL, psiR, np.conj(psiL), np.conj(psiR))
    energy = tn.ncon([psiL, psiR, H, np.conj(psiL), np.conj(psiR)], [(1,), (2,), (1, 2, 3, 4), (3,), (4,)], backend='jax')
    if isinstance(energy, torch.Tensor):
        energy = energy.numpy()

    return np.real(energy)

# Computes <psiL | H | psiR>
def contract_energy_mpo(H, psiL, psiR, rangeL=None, rangeR=None) -> float:
    if rangeL is None and rangeR is None:
        rangeL = range(n_qubits//2)
        rangeR = range(n_qubits//2, n_qubits)
    elif rangeL is None and not rangeR is None or not rangeL is None and rangeR is None:
        raise Exception("this can't be the case, either specify both or neither")

    energy = 0
    n_qubits = H.n_qubits
    d = int(2**(n_qubits/2))
    # en_einsum = np.einsum('ijkl, i, j, k, l', H, psiL, psiR, np.conj(psiL), np.conj(psiR))
    indexs = 0
    for mpo in H.mpo:
        nodes = [tn.Node(mpo.container[q], name=str(q))
                 for q in range(n_qubits)]
        # Connect network along bond dimensions
        for q in range(n_qubits-1):
            nodes[q][1] ^ nodes[q+1][0]
        # Gather dangling edges
        dummy_edges = [nodes[0].get_edge(0), nodes[-1].get_edge(1)]
        mid_edges = [nodes[n_qubits//2-1].get_edge(1), nodes[n_qubits//2].get_edge(0)]
        # edges_upper_l = [nodes[q].get_edge(2) for q in range(n_qubits//2)]
        # edges_lower_l = [nodes[q].get_edge(3) for q in range(n_qubits//2)]
        # edges_upper_r = [nodes[q].get_edge(2) for q in range(n_qubits//2, n_qubits)]
        # edges_lower_r = [nodes[q].get_edge(3) for q in range(n_qubits//2, n_qubits)]
        edges_upper_l = [nodes[q].get_edge(2) for q in rangeL]
        edges_lower_l = [nodes[q].get_edge(3) for q in rangeL]
        edges_upper_r = [nodes[q].get_edge(2) for q in rangeR]
        edges_lower_r = [nodes[q].get_edge(3) for q in rangeR]
        # Connect psi's to MPO
        psiL = psiL.reshape([2 for _ in range(n_qubits//2)])  # this should be ok cause it's within
        psiR = psiR.reshape([2 for _ in range(n_qubits//2)])
        psiL_node = tn.Node(psiL)
        psiR_node = tn.Node(psiR)
        psiLdg = np.conj(psiL)
        psiRdg = np.conj(psiR)
        psiLdg_node = tn.Node(psiLdg)
        psiRdg_node = tn.Node(psiRdg)
        for i_e, e in enumerate(psiL_node.edges):
            e ^ edges_upper_l[i_e]
        for i_e, e in enumerate(psiR_node.edges):
            e ^ edges_upper_r[i_e]
        for i_e, e in enumerate(psiLdg_node.edges):
            e ^ edges_lower_l[i_e]
        for i_e, e in enumerate(psiRdg_node.edges):
            e ^ edges_lower_r[i_e]
        res = tn.contractors.auto(nodes+[psiL_node, psiR_node,
                                   psiLdg_node, psiRdg_node],
                                  ignore_edge_order=True)
        energy += res.tensor.numpy()[0][0]

    return np.real(energy)

def tmp_full_to_LR_wfn(wfn_array, d, subsysL: list = [0,1,2], subsysR: list = [3,4,5]) -> np.ndarray:
    # psiL, psiR = np.zeros(d, dtype='complex'), np.zeros(d, dtype='complex')

    # This does not work at all
    def fetch_vec_per_subsys(wfn_array: np.ndarray, subsystem: list) -> np.ndarray:
        subsystem.sort()
        out_list = []
        index_list = [0] # |000...0> is in every subsystem!
        for q in subsystem:
            index_list_copy = copy.deepcopy(index_list)
            for index in index_list_copy:
                tmp = index + int(2**q)
                index_list += [tmp]
        index_list.sort()

        out_wfn = np.zeros(len(index_list))
        for it, index in enumerate(index_list):
            out_wfn[it] = wfn_array[index]

        return out_wfn

    # Get from previous solution and renormalize
    psiL = fetch_vec_per_subsys(wfn_array, subsysL)
    psiL = normalize(psiL)
    psiR = fetch_vec_per_subsys(wfn_array, subsysR)
    psiR = normalize(psiR)

    return psiL, psiR


def update_psi(env, psi, SL):
    out_psi = np.conj(env) - SL*psi

    return normalize(out_psi)

def update_psi_mpo(env_conj, psi, SL):
    out_psi = env_conj - SL*psi

    return normalize(out_psi)

def compute_environment(H, psiL, psiR, which: str='l'):
    if which.lower() == 'l':
        # env = np.einsum('j, ijkl, k, l', psiR, H, np.conj(psiL), np.conj(psiR), optimize='greedy')
        env = tn.ncon([psiR, H, np.conj(psiL), np.conj(psiR)], [(2,), (-1, 2, 3, 4), (3,), (4,)],
                      backend='jax')
    if which.lower() == 'r':
        # env = np.einsum('i, ijkl, k, l', psiL, H, np.conj(psiL), np.conj(psiR), optimize='greedy')
        env = tn.ncon([psiL, H, np.conj(psiL), np.conj(psiR)], [(1,), (1, -2, 3, 4), (3,), (4,)],
                      backend='jax')

    return env

def compute_environment_mpo(H, psiL, psiR, which: str='l', rangeL=None, rangeR=None):
    if rangeL is None and rangeR is None:
        rangeL = range(n_qubits//2)
        rangeR = range(n_qubits//2, n_qubits)
    elif rangeL is None and not rangeR is None or not rangeL is None and rangeR is None:
        raise Exception("this can't be the case, either specify both or neither")
    n_qubits = H.n_qubits
    d = int(2**(n_qubits/2))
    environment = None
    first = True
    for mpo in H.mpo:
        nodes = [tn.Node(mpo.container[q], name=str(q))
                 for q in range(n_qubits)]
        # Connect network along bond dimensions
        for q in range(n_qubits-1):
            nodes[q][1] ^ nodes[q+1][0]
        # Gather dangling edges
        dummy_edges = [nodes[0].get_edge(0), nodes[-1].get_edge(1)]
        mid_edges = [nodes[n_qubits//2-1].get_edge(1), nodes[n_qubits//2].get_edge(0)]
        edges_upper_l = [nodes[q].get_edge(2) for q in rangeL]
        edges_lower_l = [nodes[q].get_edge(3) for q in rangeL]
        edges_upper_r = [nodes[q].get_edge(2) for q in rangeR]
        edges_lower_r = [nodes[q].get_edge(3) for q in rangeR]
        # edges_upper_l = [nodes[q].get_edge(2) for q in range(n_qubits//2)]
        # edges_lower_l = [nodes[q].get_edge(3) for q in range(n_qubits//2)]
        # edges_upper_r = [nodes[q].get_edge(2) for q in range(n_qubits//2, n_qubits)]
        # edges_lower_r = [nodes[q].get_edge(3) for q in range(n_qubits//2, n_qubits)]
        # Connect psi's to MPO
        psiL = psiL.reshape([2 for _ in range(n_qubits//2)])
        psiR = psiR.reshape([2 for _ in range(n_qubits//2)])
        psiL_node = tn.Node(psiL)
        psiR_node = tn.Node(psiR)
        psiLdg = np.conj(psiL)
        psiRdg = np.conj(psiR)
        psiLdg_node = tn.Node(psiLdg)
        psiRdg_node = tn.Node(psiRdg)
        # If want right environment, connect with psiL and add psiR-nodes to output edges
        if which.lower() == 'r':
            for i_e, e in enumerate(psiL_node.edges):
                e ^ edges_upper_l[i_e]
            network = nodes + [psiL_node, psiLdg_node, psiRdg_node]
            output_edge_order = dummy_edges + edges_upper_r
        if which.lower() == 'l':
            for i_e, e in enumerate(psiR_node.edges):
                e ^ edges_upper_r[i_e]
            network = nodes + [psiR_node, psiLdg_node, psiRdg_node]
            output_edge_order = dummy_edges + edges_upper_l
        for i_e, e in enumerate(psiLdg_node.edges):
            e ^ edges_lower_l[i_e]
        for i_e, e in enumerate(psiRdg_node.edges):
            e ^ edges_lower_r[i_e]
        res = tn.contractors.auto(network,
                                  output_edge_order=output_edge_order)
        if not first:
            environment += res.tensor
        else:
            environment = res.tensor

    if isinstance(environment, torch.Tensor):
        environment = environment.numpy()
    return np.conj(environment.reshape(d))

# "Optimize" vectors
def optimize_wavefunctions(H, psiL, psiR, SL=1., TOL=1e-8, silent=True):
    it = 0
    energy = 0
    dE = 12.7
    stuck = False

    while dE > TOL and not stuck:
        it += 1
        # L-update
        envL = compute_environment(H, psiL, psiR, 'L')
        psiL = update_psi(envL, psiL, SL)
        # R-update
        envR = compute_environment(H, psiL, psiR, 'R')
        psiR = update_psi(envR, psiR, SL)

        old_energy = energy
        energy = contract_energy(H, psiL, psiR)
        if not silent:
            print("At ", it, " have energy ", energy)
        else:
            if not it%100:
                print("At ", it, " have energy ", energy)
        dE = np.abs(energy - old_energy)
        if it > 500:
            stuck = True

    #print("\tEnergy optimization reached  ", energy, " after ", it, " iterations.")

    if stuck:
        return None
    else:
        return energy, psiL, psiR

# "Optimize" vectors --- MPO Version
def optimize_wavefunctions_mpo(H, psiL, psiR, SL=1., TOL=1e-10, silent=True):
    it = 0
    energy = 0
    dE = 12.7

    rangeL, rangeR = None, None
    n_qubits = H.n_qubits
    ''' 
    modified ranges!
    '''
    rangeL = range(1, n_qubits//2+1)
    rangeR = itertools.chain([0], range(n_qubits//2+1, n_qubits))
    ''' 
    end modified ranges!
    '''

    while dE > TOL:
        it += 1
        # L-update
        envL_conj = compute_environment_mpo(H, psiL, psiR, 'L', rangeL=rangeL, rangeR=rangeR)
        psiL = update_psi_mpo(envL_conj, psiL, SL)
        # R-update
        envR_conj = compute_environment_mpo(H, psiL, psiR, 'R', rangeL=rangeL, rangeR=rangeR)
        psiR = update_psi_mpo(envR_conj, psiR, SL)

        old_energy = energy
        energy = contract_energy_mpo(H, psiL, psiR, rangeL=rangeL, rangeR=rangeR)
        if not silent:
            print("At ", it, " have energy ", energy)
        dE = np.abs(energy - old_energy)

    #print("\tEnergy optimization reached  ", energy, " after ", it, " iterations.")

    return energy, psiL, psiR

# def wfvec_to_tensor(wfvec, subs_qubits: int = 3):
#     shape = tuple([2 for _ in range(subs_qubits)])
#     return wfvec.reshape(shape)
#
# def tensor_to_wfvec(tensor, subs_qubits: int = 3):
#     d = int(2**(subs_qubits))
#     return tensor.reshape(d)

def initialize_wfns_randomly(dim: int = 8, n_qubits: int = 3):
    # psi = np.random.rand(dim)# + 1.j*(np.random.rand(dim)-1/2)
    psi = np.random.rand(dim)-1/2 + 1.j*(np.random.rand(dim)-1/2)
    psi = normalize(psi)

    return psi

# def initialize_wfns_randomly_mpo(dim: int = 8, n_qubits: int = 3):
#     psi = np.random.rand(dim) + 1.j*(np.random.rand(dim)-1/2)
#     psi = psi.reshape(tuple([2 for _ in range(n_qubits)]))
#     psi = normalize_mpo(psi)
#
#     return psi

def main():

    # First construct it, will load Hamiltonian later
    # mol = tq.Molecule(geometry='H 0.0 0.0 0.0\n H 0.0 0.0 0.7', basis_set='6-31g', active_orbitals=list(range(3)), transformation='jordan-wigner')
    # mol = tq.Molecule(geometry='H 0.0 0.0 0.0\n H 0.0 0.0 0.7', basis_set='sto-3g', transformation='jordan-wigner')
    mol = tq.Molecule(geometry='O 0.0 0.0 0.0\n H 0.0 0.755 -0.476\n H 0.0 -0.755 -0.476', basis_set='sto-3g', backend='psi4', threads=12)
    H = mol.make_hamiltonian().simplify()
    n_qubits = len(H.qubits)
    print("n_qubits:", n_qubits)
    d = int(2**(n_qubits/2))
    print("d:", d)


    # I somehow thought the following might be a good idea, but apparently it does not really work ^^
    # Still keeping it here just in case that it's just because of some bug
    """
    # In a longer term, we might try to somehow translate from a tq.QubitWavefunction here...
    # For now instead of random vector, let's get the UCCD-one and separate it
    U = mol.make_upccgsd_ansatz(name='uccd')
    E = tq.ExpectationValue(H=H, U=U)

    res = tq.minimize(objective=E, method='slsqp', silent=True)
    print("Optimized energy:", res.energy)

    # tq.QubitWavefunction
    wfn = tq.simulate(objective=U, variables=res.angles)
    # print(wfn)

    # As array (here then with size 2**6 = 64)
    wfn_array = wfn.to_array()
    # print(wfn_array)
    # Just as a test

    # Now separate the wavefunction into two subsystems, where each then has size 2**3 = 8
    psiL, psiR = tmp_full_to_LR_wfn(wfn_array, d)

    # Let's see what separated version of UCCD-solution gives... we lost something, so we should expect worse than FCI
    sep_energy = contract_energy(H_mat_tq, psiL, psiR)
    print("Initial separated UCCD energy:", sep_energy)

    # Optimize wavefunctions based UCCD-solution
    energy_U, psiL_U, psiR_U = optimize_wavefunctions(H_mat_tq, psiL, psiR)
    print("Optimized wfns:", psiL_U, psiR_U)
    """


    # # Now, use Lukasz's Hamiltonian
    # H = np.loadtxt('filename.txt', dtype='complex', delimiter=',')
    # H_mat = np.reshape(H.to_matrix(), (d, d, d, d))

    H_mpo = MyMPO(hamiltonian=H, n_qubits=n_qubits, maxdim=400)
    H_mpo.make_mpo_from_hamiltonian()

    psiL_rand = initialize_wfns_randomly(d, n_qubits//2)
    psiR_rand = initialize_wfns_randomly(d, n_qubits//2)
    psiL_rand_mpo = initialize_wfns_randomly(d, n_qubits//2)
    psiR_rand_mpo = initialize_wfns_randomly(d, n_qubits//2)

    # en = contract_energy(H_mat, psiL_rand, psiR_rand)
    en_mpo = contract_energy_mpo(H_mpo, psiL_rand, psiR_rand)
    # print("Initial random state energy:", en)
    print("Initial random state energy mpo:", en_mpo)

    # Optimize wavefunctions based on random guess
    # energy_rand, _, _  = optimize_wavefunctions(H_mat, psiL_rand, psiR_rand,
    #                                             silent=False)
    import time
    t0 = time.time()
    energy_rand, psiL_rand, psiR_rand = optimize_wavefunctions_mpo(H_mpo,
                                                                   psiL_rand,
                                                                   psiR_rand,
                                                                   silent=False)
    t1 = time.time()
    print("needed ", t1-t0, " seconds.")


# Execute main function
if __name__ == '__main__':
    main()

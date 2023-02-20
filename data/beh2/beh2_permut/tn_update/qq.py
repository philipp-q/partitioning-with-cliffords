import numpy as np
import tequila as tq
import tensornetwork as tn

import itertools
import copy

def normalize(me, order=2):
    return me/np.linalg.norm(me, ord=order)

# Computes <psiL | H | psiR>
def contract_energy(H, psiL, psiR) -> float:
    energy = 0

    # For test:
    # en_einsum = np.einsum('ijkl, i, j, k, l', H, psiL, psiR, np.conj(psiL), np.conj(psiR))
    energy = tn.ncon([psiL, psiR, H, np.conj(psiL), np.conj(psiR)], [(1,), (2,), (1, 2, 3, 4), (3,), (4,)], backend='pytorch')

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

def compute_environment(H, psiL, psiR, which: str='l'):
    if which.lower() == 'l':
        # env = np.einsum('j, ijkl, k, l', psiR, H, np.conj(psiL), np.conj(psiR), optimize='greedy')        
        env = tn.ncon([psiR, H, np.conj(psiL), np.conj(psiR)], [(2,), (-1, 2, 3, 4), (3,), (4,)],
                      backend='pytorch')       
    if which.lower() == 'r':  
        # env = np.einsum('i, ijkl, k, l', psiL, H, np.conj(psiL), np.conj(psiR), optimize='greedy')        
        env = tn.ncon([psiL, H, np.conj(psiL), np.conj(psiR)], [(1,), (1, -2, 3, 4), (3,), (4,)],
                      backend='pytorch')       

    return env

# "Optimize" vectors
def optimize_wavefunctions(H, psiL, psiR, SL=1., TOL=1e-10, silent=True):
    it = 0
    energy = 0
    dE = 12.7

    while dE > TOL:
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
        dE = np.abs(energy - old_energy)

    if not silent:
        print("Reached final energy of ", energy, " after ", it, " iterations.")

    return energy, psiL, psiR

def main():
    n_qubits = 6
    
    # First construct it, will load Hamiltonian later
    mol = tq.Molecule(geometry='H 0.0 0.0 0.0\n H 0.0 0.0 0.7', basis_set='6-31g', active_orbitals=list(range(n_qubits//2)), transformation='jordan-wigner')
    H = mol.make_hamiltonian().simplify()
    
    d = int(2**(n_qubits/2))
    print(d)
    # Reshape Hamiltonian matrix
    # TODO this is supposed to become a MPO
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    H_mol = mol.make_molecular_hamiltonian()
    print(H_mol)

    # Guess we should use this to transform into MPO
    # CHECK ORDERING OF H_mol (might be Mulliken, but likely the openfermion one!)
    # H = h_0 + h_pq a^p a_q + h_pqrs a^p a^q a_s a_r
    #   h_0: identity over everything
    #   rest: ~ JW 


    raise Exception(".")


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    H_mat_tq = np.reshape(H.to_matrix(), (d, d, d, d))
    # print(H_mat)
    
   
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
   

    # Now, use Lukasz's Hamiltonian
    H = np.loadtxt('filename.txt', dtype='complex', delimiter=',')
    H_mat = np.reshape(H, (d, d, d, d))

    def construct_psi_randomly(dim: int = 8):
        # psi = np.random.rand(dim)# + 1.j*(np.random.rand(dim)-1/2)
        psi = np.random.rand(dim)-1/2 + 1.j*(np.random.rand(dim)-1/2)
        psi /= np.linalg.norm(psi, ord=2)

        return psi

    psiL_rand = construct_psi_randomly(d)
    psiR_rand = construct_psi_randomly(d)

    en = contract_energy(H_mat, psiL_rand, psiR_rand)
    print("Initial random state energy:", en)
    

    # Optimize wavefunctions based on random guess
    energy_rand, psiL_rand, psiR_rand = optimize_wavefunctions(H_mat, psiL_rand, psiR_rand)
    print("Optimized wfns:", psiL_rand, psiR_rand)

        
   
# Execute main function
if __name__ == '__main__':
    main()

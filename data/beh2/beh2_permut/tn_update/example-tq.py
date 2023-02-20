import tequila as tq
import numpy as np
import time

from my_mpo import MyMPO, SubOperator

#mol = tq.Molecule(geometry='H 0.0 0.0 0.0\n H 0.0 0.0 0.7', basis_set='sto-3g', backend='psi4')
mol = tq.Molecule(geometry='Li 0.0 0.0 0.0\n H 0.0 0.0 1.4', basis_set='sto-3g', backend='psi4')
#mol = tq.Molecule(geometry='O 0.0 0.0 0.0\n H 0.0 0.755 -0.476\n H 0.0 -0.755 -0.476', basis_set='sto-3g', backend='psi4')
#mol = tq.Molecule(geometry='O 0.0 0.0 0.0\n H 0.0 0.756 -0.476\n H 0.0 -0.755 -0.476', basis_set='6-31g', backend='psi4')
#mol = tq.Molecule(geometry='N 0.0 0.0 0.0\n N 0.0 0.0 1.106', basis_set='cc-pvdz', backend='psi4')
n_qubits = mol.molecule.n_qubits 
print("num qubits", n_qubits)
H = mol.make_hamiltonian().simplify()
print("num paulis", len(H))

print(H)

t0 = time.time()
mpo = MyMPO(hamiltonian=H, n_qubits=n_qubits, maxdim=15)
mpo.make_mpo_from_hamiltonian()
for m in mpo.mpo:
    for x in m.container:
        print(x.shape)
t1 = time.time()

print("construction of MPO:", t1-t0)

H_mpo = mpo.construct_matrix()
t2 = time.time()
print("construction of Ham from MPO:", t2-t1)


print("trying to construct matrix")
H_mat = H.to_matrix()
d = int(2**(n_qubits/2))
H_mat = H_mat.reshape((d,d,d,d))
print("did so")

print("are they same?", np.all(np.isclose(H_mat, H_mpo)))
#print(H_mat)
#print(H_mpo)

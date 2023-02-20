import tequila as tq
import numpy

def opt_mol(mol, U, guess=None, threshold=1.e-5):
    delta=1.0
    energy=1.0
    while(delta>threshold):
        opt = tq.chemistry.optimize_orbitals(molecule=mol, circuit=U, initial_guess=guess, silent=True)
        guess = opt.mo_coeff
        delta = abs(opt.energy-energy)
        energy = opt.energy
    return opt

start=1.0
step=0.2
steps=10

# taken from https://github.com/kottmanj/moldata/beh2/
name="/h/332/philipps/software/moldata/beh2/beh2_{0:1.2f}_{0:1.2f}_180"
geometry="be 0.0 0.0 0.0\nH 0.0 0.0 {0:1.2f}\nH 0.0 0.0 -{0:1.2f}"
guess = None
energies = []
energies_opt = []
points = []
# for R in ([start + i*step for i in range(steps)]):
for R in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]:
    mol=tq.Molecule(name=name.format(R), geometry=geometry.format(R))
    U = mol.make_ansatz(name="SPA")
    E = tq.ExpectationValue(H=mol.make_hamiltonian(), U=U)
    result = tq.minimize(E, silent=True)
    energy0 = result.energy
    opt = opt_mol(mol=mol, U=U, guess=guess)
    guess = opt.mo_coeff
    fci = mol.compute_energy("fci")
    delta1 = fci-result.energy
    delta2 = fci-opt.energy
    energies.append(energy0)
    energies_opt.append(opt.energy)
    points.append(R)
    print("{:1.2f} | {:+1.4f} | {:+1.4f}".format(R,delta1,delta2))

print(points)
# print(energies)
print('oo-energies')
print(energies_opt)

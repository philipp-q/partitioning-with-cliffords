import tequila as tq

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mutation_options import *
from typing import List, Tuple

import sys
sys.path.append('/h/292/philipps/Documents/qcss/tenerq/qcss-lanl-tenerq/tn_update')
from my_mpo import *

increase_mode = 1 
export_fig = False 

def get_gate_stuff(dataframe):
    indices = dataframe.columns
    indices = list(indices)
    # vals = dataframe.values
    # print('vals', vals)
    my_stuff = indices
    # my_stuff = indices + [vals[0][0]]
    try:
        my_stuff = [stuff.partition('.')[0] for stuff in my_stuff]
        my_stuff = [stuff.strip() for stuff in my_stuff]
    except:
        pass
    my_shape = len(indices)
    return my_stuff

def get_pos_stuff(dataframe):
    indices = dataframe.columns
    # indices = list(indices[0][0])
    indices = list(indices)
    outer = []
    for i in indices:
        inner = []
        curr = i.partition('.')[0]
        curr = curr.strip('[]')
        inner = [ int(c) for c in curr.split(',') ]
        # curr = tuple(curr)
        # print('curr now', curr)
        # for c in curr:
        #     try:
        #         print('c is', c)
        #         inner.append(int(c))
        #     except:
        #         pass
        outer.append(inner)
    return outer

def get_circuit_from_files(gatesname, posname, n_qubits=8, increase_mode: bool=False):
    gates = pd.read_csv(gatesname, delimiter=',', header=0, doublequote=False, comment='&')
    gates = get_gate_stuff(dataframe=gates)
    positions = pd.read_csv(posname, delimiter=';', header=0, doublequote=False, comment='&') #, dtype=Tuple[int])
    positions = get_pos_stuff(positions)
    
    instructions = Instructions(n_qubits=n_qubits)
    instructions.gates = gates 
    instructions.positions = positions
    
    if not increase_mode:
        U, _ = build_circuit(instructions)
    elif increase_mode and not increase_mode == 2:
        U = dict() 
        for gate_id, (gate, position) in enumerate(zip(instructions.gates, instructions.positions)):
            altered_ins = copy.deepcopy(instructions)
            if gate[0] == 'C':
                continue
            altered_ins.replace_cg_w_ncg(gate_id)
            U_, _ = build_circuit(altered_ins)
            U[str(gate_id)] = U_
    elif increase_mode == 2:
        U = dict() 
        for g_id_1, (g1, p1) in enumerate(zip(instructions.gates, instructions.positions)):
            for g_id_2, (g2, p2) in enumerate(zip(instructions.gates, instructions.positions)):
                altered_ins = copy.deepcopy(instructions)
                if g_id_2 >= g_id_1: # symmetry
                    continue
                if g1[0] == 'C' or g2 == 'C':
                    continue
                if g_id_1 == g_id_2: # probably already excluded
                    continue
                altered_ins.replace_cg_w_ncg(g_id_1)
                altered_ins.replace_cg_w_ncg(g_id_2)
                U_, _ = build_circuit(altered_ins)
                U[str((g_id_1, g_id_2))] = U_
    return U

def get_mpo_and_dim(hamiltonian, n_qubits, maxdim=800):
    H_mpo = MyMPO(hamiltonian=H, n_qubits=n_qubits, maxdim=400)
    H_mpo.make_mpo_from_hamiltonian()
    dims = []
    for mpo in H_mpo.mpo:
        dim = mpo.get_dim()
        # print('current dimension', dim)
        dims += [dim]
    max_dim = np.max(dims) 

    return max_dim 

def folding_stuff_and_count(hamiltonian, circuit):
    qh = hamiltonian
    n_qubits = len(H.qubits)
    dim = get_mpo_and_dim(qh, n_qubits)

    lens = [len(qh)]
    bond_dims = [dim]
    angles = {k: 0.9292 for k in circuit.extract_variables()}
    gates = circuit.gates
    gates.reverse()
    # pqh =  
    for ng, gate in enumerate(gates):
        pqh = convert_tq_QH_to_PQH(qh)
        pqh = (fold_unitary_into_hamiltonian(gate, pqh))
        qh = convert_PQH_to_tq_QH(pqh)(angles)
        qh = qh.simplify()
        lens += [len(qh)]
        bond_dims += get_mpo_and_dim(qh, n_qubits)
    return lens, bond_dims



'''
# H2
if not increase_mode:
    terms_std_h2 = []
    terms_folded_h2 = []
else:
    terms_std_h2 = [[]] 
    terms_folded_h2 = [[]] 

distances = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
for dist in distances: 
    print('doing cliff h2', dist)
    # Set Hamiltonian for term increase investigation 
    geometry = "h 0.0 0.0 0.0\nh 0.0 0.0 {R}"
    mol = tq.Molecule(basis_set='6-31g', geometry=geometry.format(R=dist), active_orbitals=list(range(3)))
    H = mol.make_hamiltonian().simplify()
    assert(H.n_qubits == 6)

    gatesname = "files/h2_clifford_gates_{R:.1f}.txt".format(R=dist)
    posname = "files/h2_clifford_positions_{R:.1f}.txt".format(R=dist)
    Uc = get_circuit_from_files(gatesname=gatesname, posname=posname, n_qubits=6, increase_mode=increase_mode)
    if export_fig and not increase_mode:
        tq.circuit.export_to(Uc, filename='circuits/h2c_{R:.1f}.pdf'.format(R=dist))

    gatesname = "files/h2_nclifford_gates_{R:.1f}.txt".format(R=dist)
    posname = "files/h2_nclifford_positions_{R:.1f}.txt".format(R=dist)
    Unc = get_circuit_from_files(gatesname=gatesname, posname=posname, n_qubits=6, increase_mode=increase_mode)
    if export_fig and not increase_mode:
        tq.circuit.export_to(Unc, filename='circuits/h2nc_{R:.1f}.pdf'.format(R=dist))
    # Determine increase in number of terms
    if not increase_mode:
        lens_h2, _ = folding_stuff_and_count(hamiltonian=H, circuit=Unc)
        terms_std_h2 += [lens_h2[0]]
        terms_folded_h2 += [lens_h2[-1]]
    elif increase_mode:
        tmp_s, tmp_f = [], []
        for k_u, u in Uc.items():
            _, lens_h2 = folding_stuff_and_count(hamiltonian=H, circuit=u)
            tmp_s += [lens_h2[0]]
            tmp_f += [lens_h2[-1]]
        terms_std_h2 += [tmp_s]
        terms_folded_h2 += [tmp_f]

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("H2 terms before folding", terms_std_h2)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("H2 terms after folding", terms_folded_h2)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


# BeH2
if not increase_mode:
    terms_std_beh2 = []
    terms_folded_beh2 = []
else:
    terms_std_beh2 = [[]] 
    terms_folded_beh2 = [[]] 
distances = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
for dist in distances: 
    print('doing cliff beh2', dist)
    # Set Hamiltonian for term increase investigation 
    geometry = "be 0.0 0.0 0.0\nh 0.0 0.0 {R1}\nh 0.0 0.0 -{R2}"
    name = "/h/292/philipps/software/moldata/beh2/beh2_{R1:2.2f}_{R2:2.2f}_180" # adapt of you move data
    mol = tq.Molecule(name=name.format(R1=dist,R2=dist), geometry=geometry.format(R1=dist,R2=dist), n_pno=None)
    H = mol.make_hamiltonian().simplify()
    # Build circuit(s)
    gatesname = "files/beh2_clifford_gates_{R:.1f}.txt".format(R=dist)
    posname = "files/beh2_clifford_positions_{R:.1f}.txt".format(R=dist)
    Uc = get_circuit_from_files(gatesname=gatesname, posname=posname, n_qubits=8, increase_mode=increase_mode)
    if export_fig and not increase_mode:
        tq.circuit.export_to(Uc, filename='circuits/beh2c_{R:.1f}.pdf'.format(R=dist))

    print('doing noncliff beh2', dist)
    gatesname = "files/beh2_nclifford_gates_{R:.1f}.txt".format(R=dist)
    posname = "files/beh2_nclifford_positions_{R:.1f}.txt".format(R=dist)
    Unc = get_circuit_from_files(gatesname=gatesname, posname=posname, n_qubits=8)
    if export_fig and not increase_mode:
        tq.circuit.export_to(Unc, filename='circuits/beh2nc_{R:.1f}.pdf'.format(R=dist))

    if not increase_mode:
        lens_beh2, _ = folding_stuff_and_count(hamiltonian=H, circuit=Unc)
        terms_std_beh2 += [lens_beh2[0]]
        terms_folded_beh2 += [lens_beh2[-1]]
    elif increase_mode:
        tmp_s, tmp_f = [], []
        for k_u, u in Uc.items():
            print("HEEEEREEEREREREREEEEE")
            _, lens_beh2 = folding_stuff_and_count(hamiltonian=H, circuit=u)
            tmp_s += [lens_beh2[0]]
            tmp_f += [lens_beh2[-1]]
        terms_std_beh2 += [tmp_s]
        terms_folded_beh2 += [tmp_f]

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("BeH2 terms before folding", terms_std_beh2)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("BeH2 terms after folding", terms_folded_beh2)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
'''

# N2
if not increase_mode:
    terms_std_n2 = []
    terms_folded_n2 = []
else:
    terms_std_n2 = [[]]
    terms_folded_n2 = [[]]
distances = [0.75, 1.0, 1.3, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
for dist in distances: 
    print('doing cliff n2', dist)
    # Set up Hamiltonian for term increase
    geometry = "N 0.0 0.0 0.0\nN 0.0 0.0 {R}"
    name = "/h/292/philipps/software/moldata/n2/n2_{R:2.2f}" # adapt of you move data
    mol = tq.Molecule(name=name.format(R=dist), geometry=geometry.format(R=dist), n_pno=None)
    Hmol = mol.make_molecular_hamiltonian()
    # print("Hmol-size", len(Hmol))
    H = mol.make_hamiltonian().simplify()
    # print("H-size", len(H))

    if dist in [0.75, 1.75, 2.25, 2.75]:
        gatesname = "files/n2_clifford_gates_{R:.2f}.txt".format(R=dist)
        posname = "files/n2_clifford_positions_{R:.2f}.txt".format(R=dist)
    else:
        gatesname = "files/n2_clifford_gates_{R:.1f}.txt".format(R=dist)
        posname = "files/n2_clifford_positions_{R:.1f}.txt".format(R=dist)
    # Get circuit and plot
    Uc = get_circuit_from_files(gatesname=gatesname, posname=posname, n_qubits=12, increase_mode=increase_mode)
    if export_fig and not increase_mode:
        tq.circuit.export_to(Uc, filename='circuits/n2c_{R:.2f}.pdf'.format(R=dist))

    print('doing noncliff n2', dist)
    if dist in [0.75, 1.75, 2.25, 2.75]:
        gatesname = "files/n2_nclifford_gates_{R:.2f}.txt".format(R=dist)
        posname = "files/n2_nclifford_positions_{R:.2f}.txt".format(R=dist)
    else:
        gatesname = "files/n2_nclifford_gates_{R:.1f}.txt".format(R=dist)
        posname = "files/n2_nclifford_positions_{R:.1f}.txt".format(R=dist)
    Unc = get_circuit_from_files(gatesname=gatesname, posname=posname, n_qubits=12)
    if export_fig and not increase_mode:
        tq.circuit.export_to(Unc, filename='circuits/n2nc_{R:.2f}.pdf'.format(R=dist))


    # Determine increase in number of terms
    if not increase_mode:
        raise Exception("should not be here rn")
        lens_n2, _ = folding_stuff_and_count(hamiltonian=H, circuit=Unc)
        assert(len(lens_n2) > 1)
        assert(lens_n2[-1] == np.max(lens_n2))
        terms_std_n2 += [lens_n2[0]]
        terms_folded_n2 += [lens_n2[-1]]
    elif increase_mode:
        tmp_s, tmp_f = [], []
        for k_u, u in Uc.items():
            _, lens_n2 = folding_stuff_and_count(hamiltonian=H, circuit=u)
            print('current lens', lens_n2)
            tmp_s += [lens_n2[0]]
            tmp_f += [lens_n2[-1]]
        # assert(len(lens_n2) > 1)
        # assert(lens_n2[-1] == np.max(lens_n2))
        terms_std_n2 += [tmp_s]
        terms_folded_n2 += [tmp_f]


print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("N2 terms before folding", terms_std_n2)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("N2 terms after folding", terms_folded_n2)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

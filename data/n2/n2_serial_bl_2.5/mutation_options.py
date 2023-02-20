import argparse
import numpy as np
import random
import copy
import tequila as tq
from typing import Union
from collections import Counter
from time import time

from vqe_utils import convert_PQH_to_tq_QH, convert_tq_QH_to_PQH,\
                      fold_unitary_into_hamiltonian

from energy_optimization import minimize_energy

global_seed = 1

class Instructions:
    '''
    TODO need to put some documentation here
    '''
    def __init__(self, n_qubits, mu=2.0, sigma=0.4, alpha=0.9, T_0=1.0,
                       beta=0.5, patience=10, max_non_cliffords=0,
                       reference_energy=0., number=None):

        # hardcoded values for now
        self.num_non_cliffords = 0
        self.max_non_cliffords = max_non_cliffords
        # ------------------------
        self.starting_patience = patience
        self.patience = patience
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.T_0 = T_0
        self.T = T_0
        self.gates = self.get_random_gates(number=number)
        self.n_qubits = n_qubits
        self.positions = self.get_random_positions()
        self.best_previous_instructions = {}
        self.reference_energy = reference_energy
        self.best_reference_wfn = None
        self.noncliff_replacements = {}

    def _str(self):
        print(self.gates)
        print(self.positions)

    def set_reference_wfn(self, reference_wfn):
        self.best_reference_wfn = reference_wfn

    def update_T(self, update_type: str = 'regular', best_temp=None):
        # Regular update
        if update_type.lower() == 'regular':
            self.T = self.alpha * self.T
        # Temperature update if patience ran out
        elif update_type.lower() == 'patience':
            self.T = self.beta*best_temp + (1-self.beta)*self.T_0

    def get_random_gates(self, number=None):
        '''
        Randomly generates a list of gates.
        number indicates the number of gates to generate
        otherwise, number will be drawn from a log normal distribution
        '''
        mu, sigma = self.mu, self.sigma
        full_options = ['X','Y','Z','S','H','CX', 'CY', 'CZ','SWAP', 'UCC2c', 'UCC4c', 'UCC2', 'UCC4']
        clifford_options = ['X','Y','Z','S','H','CX', 'CY', 'CZ','SWAP', 'UCC2c', 'UCC4c']
        #non_clifford_options = ['UCC2', 'UCC4']
        # Gate distribution, selecting number of gates to add
        k = np.random.lognormal(mu, sigma)
        k = np.int(k)
        if number is not None:
            k = number
        # Selecting gate types
        gates = None
        if self.num_non_cliffords < self.max_non_cliffords:
            gates = random.choices(full_options, k=k) # with replacement
        else:
            gates = random.choices(clifford_options, k=k)

        new_num_non_cliffords = 0
        if "UCC2" in gates:
            new_num_non_cliffords += Counter(gates)["UCC2"]
        if "UCC4" in gates:
            new_num_non_cliffords += Counter(gates)["UCC4"]

        if (new_num_non_cliffords+self.num_non_cliffords) <= self.max_non_cliffords:
            self.num_non_cliffords += new_num_non_cliffords
        else:
            extra_cliffords =  (new_num_non_cliffords+self.num_non_cliffords) - self.max_non_cliffords
            assert(extra_cliffords >= 0)
            new_gates = random.choices(clifford_options, k=extra_cliffords)

            for g in new_gates:
                try:
                    gates[gates.index("UCC4")] = g
                except:
                    gates[gates.index("UCC2")] = g

            self.num_non_cliffords = self.max_non_cliffords

        if k == 1:
            if gates == "UCC2c" or gates == "UCC4c":
                gates = get_string_Cliff_ucc(gates)
            if gates == "UCC2" or gates == "UCC4":
                gates = get_string_ucc(gates)
        else:
            for ind, gate in enumerate(gates):
                if gate == "UCC2c" or gate == "UCC4c":
                    gates[ind] = get_string_Cliff_ucc(gate)
                if gate == "UCC2" or gate == "UCC4":
                    gates[ind] = get_string_ucc(gate)
        return gates

    def get_random_positions(self, gates=None):
        '''
        Randomly assign gates to qubits.
        '''
        if gates is None:
            gates = self.gates
        n_qubits = self.n_qubits

        single_qubit = ['X','Y','Z','S','H']
        # two_qubit = ['CX','CY','CZ', 'SWAP']
        two_qubit = ['CX', 'CY', 'CZ', 'SWAP', 'UCC2c', 'UCC2']
        four_qubit = ['UCC4c', 'UCC4']

        qubits = list(range(0, n_qubits))

        q_positions = []
        for gate in gates:
            if gate in four_qubit:
                p = random.sample(qubits, k=4)
            if gate in two_qubit:
                p = random.sample(qubits, k=2)
            if gate in single_qubit:
                p = random.sample(qubits, k=1)
            if "UCC2" in gate:
                p = random.sample(qubits, k=2)
            if "UCC4" in gate:
                p = random.sample(qubits, k=4)
            q_positions.append(p)

        return q_positions

    def delete(self, number=None):
        '''
        Randomly drops some gates from a clifford instruction set
        if not specified, the number of gates to drop is sampled from a uniform distribution over all the gates
        '''

        gates = copy.deepcopy(self.gates)
        positions = copy.deepcopy(self.positions)
        n_qubits = self.n_qubits

        if number is not None:
            num_to_drop = number
        else:
            num_to_drop = random.sample(range(1,len(gates)-1), k=1)[0]
        action_indices = random.sample(range(0,len(gates)-1), k=num_to_drop)

        for index in sorted(action_indices, reverse=True):
            if "UCC2_" in str(gates[index]) or "UCC4_" in str(gates[index]):
                self.num_non_cliffords -= 1
            del gates[index]
            del positions[index]

        self.gates = gates
        self.positions = positions

        #print ('deleted {} gates'.format(num_to_drop))

    def add(self, number=None):
        '''
        adds a random selection of clifford gates to the end of a clifford instruction set
        if number is not specified, the number of gates to add will be drawn from a log normal distribution
        '''
        gates = copy.deepcopy(self.gates)
        positions = copy.deepcopy(self.positions)
        n_qubits = self.n_qubits

        added_instructions = self.get_new_instructions(number=number)
        gates.extend(added_instructions['gates'])
        positions.extend(added_instructions['positions'])

        self.gates = gates
        self.positions = positions

        #print ('added {} gates'.format(len(added_instructions['gates'])))

    def change(self, number=None):
        '''
        change a random number of gates and qubit positions in a clifford instruction set
        if not specified, the number of gates to change is sampled from a uniform distribution over all the gates
        '''
        gates = copy.deepcopy(self.gates)
        positions = copy.deepcopy(self.positions)
        n_qubits = self.n_qubits

        if number is not None:
            num_to_change = number
        else:
            num_to_change = random.sample(range(1,len(gates)), k=1)[0]
        action_indices = random.sample(range(0,len(gates)-1), k=num_to_change)

        added_instructions = self.get_new_instructions(number=num_to_change)

        for i in range(num_to_change):
            gates[action_indices[i]] = added_instructions['gates'][i]
            positions[action_indices[i]] = added_instructions['positions'][i]

        self.gates = gates
        self.positions = positions

        #print ('changed {} gates'.format(len(added_instructions['gates'])))

    # TODO to be debugged!
    def prune(self):
        '''
        Prune instructions to remove redundant operations:
            --> first gate should go beyond subsystems (this assumes expressible enough subsystem-ciruits
                #TODO later -> this needs subsystem information in here!
            --> 2 subsequent gates that are their respective inverse can be removed
        #TODO  this might change the number of qubits acted on in theory?
        '''
        pass
        #print ("DEBUG PRUNE FUNCTION!")
        # gates = copy.deepcopy(self.gates)
        # positions = copy.deepcopy(self.positions)
        # for g_index in range(len(gates)-1):
        #     if (gates[g_index] == gates[g_index+1] and not 'S' in gates[g_index])\
        #        or (gates[g_index] == 'S' and gates[g_index+1] == 'S-dag')\
        #        or (gates[g_index] == 'S-dag' and gates[g_index+1] == 'S'):
        #         print(len(gates))
        #         if positions[g_index] == positions[g_index+1]:
        #             self.gates.pop(g_index)
        #             self.positions.pop(g_index)

    def update_by_action(self, action: str):
        '''
        Updates instruction dictionary
        -> Either adds, deletes or changes gates
        '''
        if action == 'delete':
            try:
                self.delete()
            # In case there are too few gates to delete
            except:
                pass
        elif action == 'add':
            self.add()
        elif action == 'change':
            self.change()
        else:
            raise Exception("Unknown action type " + action + ".")
        self.prune()

    def update_best_previous_instructions(self):
       ''' Overwrites the best previous instructions with the current ones. '''
       self.best_previous_instructions['gates'] = copy.deepcopy(self.gates)
       self.best_previous_instructions['positions'] = copy.deepcopy(self.positions)
       self.best_previous_instructions['T'] = copy.deepcopy(self.T)

    def reset_to_best(self):
       ''' Overwrites the current instructions with best previous ones. '''
       #print ('Patience ran out... resetting to best previous instructions.')
       self.gates = copy.deepcopy(self.best_previous_instructions['gates'])
       self.positions = copy.deepcopy(self.best_previous_instructions['positions'])
       self.patience = copy.deepcopy(self.starting_patience)
       self.update_T(update_type='patience', best_temp=copy.deepcopy(self.best_previous_instructions['T']))


    def get_new_instructions(self, number=None):
        '''
        Returns a a clifford instruction set,
        a dictionary of gates and qubit positions for building a clifford circuit
        '''
        mu = self.mu
        sigma = self.sigma
        n_qubits = self.n_qubits

        instruction = {}
        gates = self.get_random_gates(number=number)
        q_positions = self.get_random_positions(gates)
        assert(len(q_positions) == len(gates))

        instruction['gates'] = gates
        instruction['positions'] = q_positions
        # instruction['n_qubits'] = n_qubits
        # instruction['patience'] = patience
        # instruction['best_previous_options'] = {}

        return instruction

    def replace_cg_w_ncg(self, gate_id):
        ''' replaces a set of Clifford gates
            with corresponding non-Cliffords
        '''
        print("gates before", self.gates, flush=True)
        gate = self.gates[gate_id]
        if gate == 'X':
            gate = "Rx"
        elif gate == 'Y':
            gate = "Ry"
        elif gate == 'Z':
            gate = "Rz"
        elif gate == 'S':
            gate = "S_nc"
            #gate = "Rz"
        elif gate == 'H':
            gate = "H_nc"
            # this does not work???????
            # gate = "Ry"
        elif gate == 'CX':
            gate = "CRx"
        elif gate == 'CY':
            gate = "CRy"
        elif gate == 'CZ':
            gate = "CRz"
        elif gate == 'SWAP':#find a way to change this as well
            pass
            # gate = "SWAP"
        elif "UCC2c" in str(gate):
            pre_gate = gate.split("_")[0]
            mid_gate = gate.split("_")[-1]
            gate = pre_gate + "_" + "UCC2" + "_" +mid_gate
        elif "UCC4c" in str(gate):
            pre_gate = gate.split("_")[0]
            mid_gate = gate.split("_")[-1]
            gate = pre_gate + "_" + "UCC4" + "_" + mid_gate

        self.gates[gate_id] = gate
        print("gates after", self.gates, flush=True)

def build_circuit(instructions):
    '''
    constructs a tequila circuit from a clifford instruction set
    '''
    gates = instructions.gates
    q_positions = instructions.positions
    init_angles = {}

    clifford_circuit = tq.QCircuit()

    # for i in range(1, len(gates)):
    # TODO len(q_positions) not == len(gates)
    for i in range(len(gates)):
        if len(q_positions[i]) == 2:
          q1, q2 = q_positions[i]
        elif len(q_positions[i]) == 1:
          q1 = q_positions[i]
          q2 = None
        elif not len(q_positions[i]) == 4:
          raise Exception("q_positions[i] must have length 1, 2 or 4...")


        if gates[i] == 'X':
            clifford_circuit += tq.gates.X(q1)
        if gates[i] == 'Y':
            clifford_circuit += tq.gates.Y(q1)
        if gates[i] == 'Z':
            clifford_circuit += tq.gates.Z(q1)
        if gates[i] == 'S':
            clifford_circuit += tq.gates.S(q1)
        if gates[i] == 'H':
            clifford_circuit += tq.gates.H(q1)

        if gates[i] == 'CX':
            clifford_circuit += tq.gates.CX(q1, q2)

        if gates[i] == 'CY': #using generators
            clifford_circuit += tq.gates.S(q2)
            clifford_circuit += tq.gates.CX(q1, q2)
            clifford_circuit += tq.gates.S(q2).dagger()

        if gates[i] == 'CZ': #using generators
            clifford_circuit += tq.gates.H(q2)
            clifford_circuit += tq.gates.CX(q1, q2)
            clifford_circuit += tq.gates.H(q2)

        if gates[i] == 'SWAP':
            clifford_circuit += tq.gates.CX(q1, q2)
            clifford_circuit += tq.gates.CX(q2, q1)
            clifford_circuit += tq.gates.CX(q1, q2)

        if "UCC2c" in str(gates[i]) or "UCC4c" in str(gates[i]):
            clifford_circuit += get_clifford_UCC_circuit(gates[i], q_positions[i])

        # NON-CLIFFORD STUFF FROM HERE ON
        global global_seed

        if gates[i] == "S_nc":
            np.random.seed(global_seed)
            global_seed += 1
            var_name = "var"+str(np.random.rand())
            init_angles[var_name] = 0.0
            clifford_circuit += tq.gates.S(q1)
            clifford_circuit += tq.gates.Rz(angle=var_name, target=q1)

        if gates[i] == "H_nc":
            np.random.seed(global_seed)
            global_seed += 1
            var_name = "var"+str(np.random.rand())
            init_angles[var_name] = 0.0
            clifford_circuit += tq.gates.H(q1)
            clifford_circuit += tq.gates.Ry(angle=var_name, target=q1)

        if gates[i] == "Rx":
            np.random.seed(global_seed)
            global_seed += 1
            var_name = "var"+str(np.random.rand())
            init_angles[var_name] = 0.0
            clifford_circuit += tq.gates.X(q1)
            clifford_circuit += tq.gates.Rx(angle=var_name, target=q1)

        if gates[i] == "Ry":
            np.random.seed(global_seed)
            global_seed += 1
            var_name = "var"+str(np.random.rand())
            init_angles[var_name] = 0.0
            clifford_circuit += tq.gates.Y(q1)
            clifford_circuit += tq.gates.Ry(angle=var_name, target=q1)

        if gates[i] == "Rz":
            global_seed += 1
            var_name = "var"+str(np.random.rand())
            init_angles[var_name] = 0
            clifford_circuit += tq.gates.Z(q1)
            clifford_circuit += tq.gates.Rz(angle=var_name, target=q1)

        if gates[i] == "CRx":
            np.random.seed(global_seed)
            global_seed += 1
            var_name = "var"+str(np.random.rand())
            init_angles[var_name] = np.pi
            clifford_circuit += tq.gates.Rx(angle=var_name, target=q2, control=q1)

        if gates[i] == "CRy":
            np.random.seed(global_seed)
            global_seed += 1
            var_name = "var"+str(np.random.rand())
            init_angles[var_name] = np.pi
            clifford_circuit += tq.gates.Ry(angle=var_name, target=q2, control=q1)

        if gates[i] == "CRz":
            np.random.seed(global_seed)
            global_seed += 1
            var_name = "var"+str(np.random.rand())
            init_angles[var_name] = np.pi
            clifford_circuit += tq.gates.Rz(angle=var_name, target=q2, control=q1)

        def get_ucc_init_angles(gate):
            angle = None
            pre_gate = gate.split("_")[0]
            mid_gate = gate.split("_")[-1]
            if mid_gate == 'Z':
                angle = 0.
            elif mid_gate == 'S':
                angle = 0.
            elif mid_gate == 'S-dag':
                angle = 0.
            else:
                raise Exception("This should not happen -- center/mid gate should be Z,S,S_dag.")
            return angle

        if "UCC2_" in str(gates[i]) or "UCC4_" in str(gates[i]):
            uccc_circuit = get_non_clifford_UCC_circuit(gates[i], q_positions[i])
            clifford_circuit += uccc_circuit

            try:
                var_name = uccc_circuit.extract_variables()[0]
                init_angles[var_name]= get_ucc_init_angles(gates[i])
            except:
                init_angles = {}

    return clifford_circuit, init_angles

def get_non_clifford_UCC_circuit(gate, positions):
    """

    """
    pre_cir_dic = {"X":tq.gates.X, "Y":tq.gates.Y, "H":tq.gates.H, "I":None}

    pre_gate = gate.split("_")[0]
    pre_gates = pre_gate.split(*"#")

    pre_circuit = tq.QCircuit()
    for i, pos in enumerate(positions):
        try:
            pre_circuit += pre_cir_dic[pre_gates[i]](pos)
        except:
            pass

    for i, pos in enumerate(positions[:-1]):
        pre_circuit += tq.gates.CX(pos, positions[i+1])

    global global_seed

    mid_gate = gate.split("_")[-1]
    mid_circuit = tq.QCircuit()
    if mid_gate == "S":
        np.random.seed(global_seed)
        global_seed += 1
        var_name = "var"+str(np.random.rand())
        mid_circuit += tq.gates.S(positions[-1])
        mid_circuit += tq.gates.Rz(angle=tq.Variable(var_name), target=positions[-1])
    elif mid_gate == "S-dag":
        np.random.seed(global_seed)
        global_seed += 1
        var_name = "var"+str(np.random.rand())
        mid_circuit += tq.gates.S(positions[-1]).dagger()
        mid_circuit += tq.gates.Rz(angle=tq.Variable(var_name), target=positions[-1])
    elif mid_gate == "Z":
        np.random.seed(global_seed)
        global_seed += 1
        var_name = "var"+str(np.random.rand())
        mid_circuit += tq.gates.Z(positions[-1])
        mid_circuit += tq.gates.Rz(angle=tq.Variable(var_name), target=positions[-1])

    return pre_circuit + mid_circuit + pre_circuit.dagger()



def get_string_Cliff_ucc(gate):
    """
    this function randomly sample basis change and mid circuit elements for
    a ucc-type clifford circuit and adds it to the gate
    """
    pre_circ_comp = ["X", "Y", "H", "I"]
    mid_circ_comp = ["S", "S-dag", "Z"]

    p = None
    if "UCC2c" in gate:
        p = random.sample(pre_circ_comp, k=2)
    elif "UCC4c" in gate:
        p = random.sample(pre_circ_comp, k=4)

    pre_gate = "#".join([str(item) for item in p])

    mid_gate = random.sample(mid_circ_comp, k=1)[0]

    return str(pre_gate + "_" + gate + "_" + mid_gate)

def get_string_ucc(gate):
    """
    this function randomly sample basis change and mid circuit elements for
    a ucc-type clifford circuit and adds it to the gate
    """
    pre_circ_comp = ["X", "Y", "H", "I"]

    p = None
    if "UCC2" in gate:
        p = random.sample(pre_circ_comp, k=2)
    elif "UCC4" in gate:
        p = random.sample(pre_circ_comp, k=4)

    pre_gate = "#".join([str(item) for item in p])

    mid_gate = str(random.random() * 2 * np.pi)

    return str(pre_gate + "_" + gate + "_" + mid_gate)


def get_clifford_UCC_circuit(gate, positions):
    """
    This function creates an approximate UCC excitation circuit using only
    clifford Gates
    """
    #pre_circ_comp = ["X", "Y", "H", "I"]
    pre_cir_dic = {"X":tq.gates.X, "Y":tq.gates.Y, "H":tq.gates.H, "I":None}

    pre_gate = gate.split("_")[0]
    pre_gates = pre_gate.split(*"#")

    #pre_gates = []
    #if gate == "UCC2":
    #    pre_gates = random.choices(pre_circ_comp, k=2)
    #if gate == "UCC4":
    #    pre_gates = random.choices(pre_circ_comp, k=4)

    pre_circuit = tq.QCircuit()
    for i, pos in enumerate(positions):
        try:
            pre_circuit += pre_cir_dic[pre_gates[i]](pos)
        except:
            pass

    for i, pos in enumerate(positions[:-1]):
        pre_circuit += tq.gates.CX(pos, positions[i+1])

    #mid_circ_comp = ["S", "S-dag", "Z"]
    #mid_gate = random.sample(mid_circ_comp, k=1)[0]
    mid_gate = gate.split("_")[-1]
    mid_circuit = tq.QCircuit()
    if mid_gate == "S":
        mid_circuit += tq.gates.S(positions[-1])
    elif mid_gate == "S-dag":
        mid_circuit += tq.gates.S(positions[-1]).dagger()
    elif mid_gate == "Z":
        mid_circuit += tq.gates.Z(positions[-1])

    return pre_circuit + mid_circuit + pre_circuit.dagger()

def get_UCC_circuit(gate, positions):
    """
    This function creates an UCC excitation circuit
    """
    #pre_circ_comp = ["X", "Y", "H", "I"]
    pre_cir_dic = {"X":tq.gates.X, "Y":tq.gates.Y, "H":tq.gates.H, "I":None}

    pre_gate = gate.split("_")[0]
    pre_gates = pre_gate.split(*"#")

    pre_circuit = tq.QCircuit()
    for i, pos in enumerate(positions):
        try:
            pre_circuit += pre_cir_dic[pre_gates[i]](pos)
        except:
            pass

    for i, pos in enumerate(positions[:-1]):
        pre_circuit += tq.gates.CX(pos, positions[i+1])

    mid_gate_val = gate.split("_")[-1]

    global global_seed
    np.random.seed(global_seed)
    global_seed += 1
    var_name = "var"+str(np.random.rand())
    mid_circuit = tq.gates.Rz(target=positions[-1], angle=tq.Variable(var_name))
    # mid_circ_comp = ["S", "S-dag", "Z"]
    # mid_gate = random.sample(mid_circ_comp, k=1)[0]
    # mid_circuit = tq.QCircuit()
    # if mid_gate == "S":
    #     mid_circuit += tq.gates.S(positions[-1])
    # elif mid_gate == "S-dag":
    #     mid_circuit += tq.gates.S(positions[-1]).dagger()
    # elif mid_gate == "Z":
    #     mid_circuit += tq.gates.Z(positions[-1])

    return pre_circuit + mid_circuit + pre_circuit.dagger()

def schedule_actions_ratio(epoch: int, action_options: list = ['delete', 'change', 'add'],
                             decay: int = 30,
                             steady_ratio: Union[tuple, list] = [0.2, 0.6, 0.2]) -> list:
    delete, change, add = tuple(steady_ratio)

    actions_ratio = []
    for action in action_options:
        if action == 'delete':
            actions_ratio += [  delete*(1-np.exp(-1*epoch / decay))   ]
        elif action == 'change':
            actions_ratio += [  change*(1-np.exp(-1*epoch / decay))   ]
        elif action == 'add':
            actions_ratio += [ (1-add)*np.exp(-1*epoch / decay) + add ]
        else:
            print('Action type ', action, ' not defined!')

    # unnecessary for current schedule
    # if not np.isclose(np.sum(actions_ratio), 1.0):
    #     actions_ratio /= np.sum(actions_ratio)

    return actions_ratio


def get_action(ratio: list = [0.20,0.60,0.20]):
    '''
    randomly chooses an action from delete, change, add
    ratio denotes the multinomial probabilities
    '''

    choice = np.random.multinomial(n=1, pvals = ratio, size = 1)
    index = np.where(choice[0] == 1)
    action_options = ['delete', 'change', 'add']

    action = action_options[index[0].item()]

    return action


def get_prob_acceptance(E_curr, E_prev, T, reference_energy):
    '''
    Computes acceptance probability of a certain action
    based on change in energy
    '''
    # print("Ecurr is", E_curr)
    # print("Eprev is", E_prev)
    # print("delta", E_curr - E_prev)
    # raise Exception("shit")
    delta_E = 12
    try:
        delta_E = E_curr - E_prev
    except:
        print("Ecurr is", E_curr)
        print("Eprev is", E_prev)
        print("delta", E_curr - E_prev)


    prob = 0
    if delta_E < 0 and E_curr < reference_energy:
        prob = 1
    else:
        if E_curr < reference_energy:
            prob = np.exp(-delta_E/T)
        else:
            prob = 0

    return prob


def perform_folding(hamiltonian, circuit):
    # QubitHamiltonian -> ParamQubitHamiltonian
    param_hamiltonian = convert_tq_QH_to_PQH(hamiltonian)
    gates = circuit.gates

    # Go backwards
    gates.reverse()
    for gate in gates:
        # print("\t folding", gate)
        param_hamiltonian = (fold_unitary_into_hamiltonian(gate, param_hamiltonian))
    # hamiltonian = convert_PQH_to_tq_QH(param_hamiltonian)()
    hamiltonian = param_hamiltonian

    return hamiltonian

def evaluate_fitness(instructions, hamiltonian: tq.QubitHamiltonian, type_energy_eval: str, cluster_circuit: tq.QCircuit=None) -> tuple:
    '''
    Evaluates fitness=objective=energy given a system
    for a set of instructions
    '''
    #print ("evaluating fitness")
    n_qubits = instructions.n_qubits
    clifford_circuit, init_angles = build_circuit(instructions)
    # tq.draw(clifford_circuit, backend="cirq")

    ## TODO check if cluster_circuit is parametrized
    t0 = time()
    t1 = None
    folded_hamiltonian = perform_folding(hamiltonian, clifford_circuit)
    t1 = time()
    #print ("\tfolding took ", t1-t0)
    parametrized = len(clifford_circuit.extract_variables()) > 0
    initial_guess = None

    if not parametrized:
        folded_hamiltonian = (convert_PQH_to_tq_QH(folded_hamiltonian))()
    elif parametrized:
        # TODO clifford_circuit is both ref + rest; so nomenclature is shit here
        variables = [gate.extract_variables() for gate in clifford_circuit.gates\
                     if gate.extract_variables()]
        variables = np.array(variables).flatten().tolist()
        # TODO this initial_guess is absolutely useless rn and is just causing problems if called
        initial_guess = { k: 0.1 for k in variables }

    if instructions.best_reference_wfn is not None:
        initial_guess = instructions.best_reference_wfn

    E_new, optimal_state = minimize_energy(hamiltonian=folded_hamiltonian, n_qubits=n_qubits, type_energy_eval=type_energy_eval, cluster_circuit=cluster_circuit, initial_guess=initial_guess,\
        initial_mixed_angles=init_angles)
    t2 = time()

    #print ("evaluated fitness;  comp was ", t2-t1)
    #print ("current fitness: ", E_new)
    return E_new, optimal_state

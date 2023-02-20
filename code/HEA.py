import tequila as tq
import numpy as np

from tequila import gates as tq_g
from tequila.objective.objective import Variable

def generate_HEA(num_qubits, circuit_id=11, num_layers=1):
    """
    This function generates different types of hardware efficient
    circuits as in this paper
    https://onlinelibrary.wiley.com/doi/full/10.1002/qute.201900070

    param: num_qubits (int) -> the number of qubits in the circuit
    param: circuit_id (int) -> the type of hardware efficient circuit
    param: num_layers (int) -> the number of layers of the HEA

    input:
    num_qubits -> 4
    circuit_id -> 11
    num_layers -> 1

    returns:
    ansatz (tq.QCircuit()) -> a circuit as shown below
    0: ───Ry(0.318309886183791*pi*f((y0,))_0)───Rz(0.318309886183791*pi*f((z0,))_1)───@─────────────────────────────────────────────────────────────────────────────────────
                                                                                      │
    1: ───Ry(0.318309886183791*pi*f((y1,))_2)───Rz(0.318309886183791*pi*f((z1,))_3)───X───Ry(0.318309886183791*pi*f((y4,))_8)────Rz(0.318309886183791*pi*f((z4,))_9)────@───
                                                                                                                                                                        │
    2: ───Ry(0.318309886183791*pi*f((y2,))_4)───Rz(0.318309886183791*pi*f((z2,))_5)───@───Ry(0.318309886183791*pi*f((y5,))_10)───Rz(0.318309886183791*pi*f((z5,))_11)───X───
                                                                                      │
    3: ───Ry(0.318309886183791*pi*f((y3,))_6)───Rz(0.318309886183791*pi*f((z3,))_7)───X─────────────────────────────────────────────────────────────────────────────────────

    """
    circuit = tq.QCircuit()
    qubits = [i for i in range(num_qubits)]
    if circuit_id == 1:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
        return circuit
    elif circuit_id == 2:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, qubit in enumerate(qubits[1:]):
                circuit += tq_g.CNOT(target=qubit, control=qubits[ind])
        return circuit
    elif circuit_id == 3:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, qubit in enumerate(qubits[1:]):
                variable = Variable("cz"+str(count))
                circuit += tq_g.Rz(angle = variable,target=qubit, control=qubits[ind])
                count += 1
        return circuit
    elif circuit_id == 4:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, qubit in enumerate(qubits[1:]):
                variable = Variable("cx"+str(count))
                circuit += tq_g.Rx(angle = variable,target=qubit, control=qubits[ind])
                count += 1
        return circuit
    elif circuit_id == 5:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, control in enumerate(qubits):
                for target in qubits:
                    if control != target:
                        variable = Variable("cz"+str(count))
                        circuit += tq_g.Rz(angle = variable,target=target, control=control)
                        count += 1
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
        return circuit
    elif circuit_id == 6:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, control in enumerate(qubits):
                if ind % 2 == 0:
                    variable = Variable("cz"+str(count))
                    circuit += tq_g.Rz(angle = variable,target=qubits[ind+1], control=control)
                    count += 1
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, control in enumerate(qubits[:-1]):
                if ind % 2 != 0:
                    variable = Variable("cz"+str(count))
                    circuit += tq_g.Rz(angle = variable,target=qubits[ind+1], control=control)
                    count += 1
        return circuit
    elif circuit_id == 7:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, control in enumerate(qubits):
                for target in qubits:
                    if control != target:
                        variable = Variable("cx"+str(count))
                        circuit += tq_g.Rx(angle = variable,target=target, control=control)
                        count += 1
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
        return circuit
    elif circuit_id == 8:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, control in enumerate(qubits):
                if ind % 2 == 0:
                    variable = Variable("cx"+str(count))
                    circuit += tq_g.Rx(angle = variable,target=qubits[ind+1], control=control)
                    count += 1
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, control in enumerate(qubits[:-1]):
                if ind % 2 != 0:
                    variable = Variable("cx"+str(count))
                    circuit += tq_g.Rx(angle = variable,target=qubits[ind+1], control=control)
                    count += 1
        return circuit
    elif circuit_id == 9:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                circuit += tq_g.H(qubit)
            for ind, qubit in enumerate(qubits[1:]):
                circuit += tq_g.Z(target=qubit, control=qubits[ind])
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                count += 1
        return circuit
    elif circuit_id == 10:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("y"+str(count))
                circuit += tq_g.Ry(angle=variable, target = qubit)
                count += 1
            for ind, qubit in enumerate(qubits[1:]):
                circuit += tq_g.Z(target=qubit, control=qubits[ind])
            circuit += tq_g.Z(target=qubits[0], control=qubits[-1])
            for qubit in qubits:
                variable = Variable("y"+str(count))
                circuit += tq_g.Ry(angle=variable, target = qubit)
                count += 1
        return circuit
    elif circuit_id == 11:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("y"+str(count))
                circuit += tq_g.Ry(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, control in enumerate(qubits):
                if ind % 2 == 0:
                    circuit += tq_g.X(target=qubits[ind+1], control=control)
            for ind, qubit in enumerate(qubits[1:-1]):
                variable = Variable("y"+str(count))
                circuit += tq_g.Ry(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, control in enumerate(qubits[:-1]):
                if ind % 2 != 0:
                    circuit += tq_g.X(target=qubits[ind+1], control=control)
        return circuit
    elif circuit_id == 12:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("y"+str(count))
                circuit += tq_g.Ry(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, control in enumerate(qubits):
                if ind % 2 == 0:
                    circuit += tq_g.Z(target=qubits[ind+1], control=control)
            for ind, control in enumerate(qubits[1:-1]):
                variable = Variable("y"+str(count))
                circuit += tq_g.Ry(angle=variable, target = control)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = control)
                count += 1
            for ind, control in enumerate(qubits[:-1]):
                if ind % 2 != 0:
                    circuit += tq_g.Z(target=qubits[ind+1], control=control)
        return circuit
    elif circuit_id == 13:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("y"+str(count))
                circuit += tq_g.Ry(angle=variable, target = qubit)
                count += 1
            for ind, qubit in enumerate(qubits[1:]):
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target=qubit, control=qubits[ind])
                count += 1
            variable = Variable("z"+str(count))
            circuit += tq_g.Rz(angle=variable, target=qubits[0], control=qubits[-1])
            count += 1
            for qubit in qubits:
                variable = Variable("y"+str(count))
                circuit += tq_g.Ry(angle=variable, target = qubit)
                count += 1
            for ind, qubit in enumerate(qubits[1:]):
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, control=qubit, target=qubits[ind])
                count += 1
            variable = Variable("z"+str(count))
            circuit += tq_g.Rz(angle=variable, control=qubits[0], target=qubits[-1])
            count += 1
        return circuit
    elif circuit_id == 14:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("y"+str(count))
                circuit += tq_g.Ry(angle=variable, target = qubit)
                count += 1
            for ind, qubit in enumerate(qubits[1:]):
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target=qubit, control=qubits[ind])
                count += 1
            variable = Variable("x"+str(count))
            circuit += tq_g.Rx(angle=variable, target=qubits[0], control=qubits[-1])
            count += 1
            for qubit in qubits:
                variable = Variable("y"+str(count))
                circuit += tq_g.Ry(angle=variable, target = qubit)
                count += 1
            for ind, qubit in enumerate(qubits[1:]):
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, control=qubit, target=qubits[ind])
                count += 1
            variable = Variable("x"+str(count))
            circuit += tq_g.Rx(angle=variable, control=qubits[0], target=qubits[-1])
            count += 1
        return circuit
    elif circuit_id == 15:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("y"+str(count))
                circuit += tq_g.Ry(angle=variable, target = qubit)
                count += 1
            for ind, qubit in enumerate(qubits[1:]):
                circuit += tq_g.X(target=qubit, control=qubits[ind])
            circuit += tq_g.X(control=qubits[-1], target=qubits[0])
            for qubit in qubits:
                variable = Variable("y"+str(count))
                circuit += tq_g.Ry(angle=variable, target = qubit)
                count += 1
            for ind, qubit in enumerate(qubits[1:]):
                circuit += tq_g.X(control=qubit, target=qubits[ind])
            circuit += tq_g.X(target=qubits[-1], control=qubits[0])
        return circuit
    elif circuit_id == 16:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, control in enumerate(qubits):
                if ind % 2 == 0:
                    variable = Variable("z"+str(count))
                    circuit += tq_g.Rz(angle=variable, control=control, target=qubits[ind+1])
                    count += 1
            for ind, control in enumerate(qubits[:-1]):
                if ind % 2 != 0:
                    variable = Variable("z"+str(count))
                    circuit += tq_g.Rz(angle=variable, control=control, target=qubits[ind+1])
                    count += 1
        return circuit
    elif circuit_id == 17:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, control in enumerate(qubits):
                if ind % 2 == 0:
                    variable = Variable("x"+str(count))
                    circuit += tq_g.Rx(angle=variable, control=control, target=qubits[ind+1])
                    count += 1
            for ind, control in enumerate(qubits[:-1]):
                if ind % 2 != 0:
                    variable = Variable("x"+str(count))
                    circuit += tq_g.Rx(angle=variable, control=control, target=qubits[ind+1])
                    count += 1
        return circuit
    elif circuit_id == 18:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, qubit in enumerate(qubits[:-1]):
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target=qubit, control=qubits[ind+1])
                count += 1
            variable = Variable("z"+str(count))
            circuit += tq_g.Rz(angle=variable, target=qubits[-1], control=qubits[0])
            count += 1
        return circuit
    elif circuit_id == 19:
        count = 0
        for _ in range(num_layers):
            for qubit in qubits:
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target = qubit)
                variable = Variable("z"+str(count))
                circuit += tq_g.Rz(angle=variable, target = qubit)
                count += 1
            for ind, qubit in enumerate(qubits[:-1]):
                variable = Variable("x"+str(count))
                circuit += tq_g.Rx(angle=variable, target=qubit, control=qubits[ind+1])
                count += 1
            variable = Variable("x"+str(count))
            circuit += tq_g.Rx(angle=variable, target=qubits[-1], control=qubits[0])
            count += 1
        return circuit

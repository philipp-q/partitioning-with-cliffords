import tequila as tq
from openfermion import QubitOperator
from vqe_utils import *
from scipy_optimizer import *
import time

def run_optimization(unitary, Hamiltonian, initial_values):
    """
    This function takes the Hamiltonian and breaks it into ExpectationValues so that
    it can be simulated using tequila

    param: unitary (tq.QCircuit()) -> the circuit to be simulated
    param: Hamiltonian (ParamQubitHamiltonian()) -> the parameterized Hamiltonian
    param; initial_values (dict) -> the initial values of the variables

    e.g.:
    input:
    unitary ->
    Hamiltonian ->

    returns:
    result (tq.OptimizerResults()) ->

    """
    """E = None
    for term in Hamiltonian.terms:
        if E is None:
            E = tq.ExpectationValue(H=tq.QubitHamiltonian(1.0), U=unitary)*Hamiltonian.terms[term]
        else:

            E += tq.ExpectationValue(H=tq.QubitHamiltonian(QubitOperator(term)), U=unitary)*Hamiltonian.terms[term]"""
    start = time.time()
    result = minimize(Hamiltonian, unitary, method='bfgs', initial_values=initial_values,  backend="qulacs")
    end = time.time()
    #result = tq.minimize(E, steps=1, backend="qulacs")
    print(end-start)
    return result

def run_vqe_h(number_of_gates_to_fold, full_unitary, Hamiltonian, initial_values):
    """
    This function runs a vqe with "number_of_gates_to_fold" gates folded in the Hamiltonian
    from the full_unitary

    param: number_of_gates_to_fold (int) -> number of gates to fold in the Hamiltonian
    param: full_unitary (tq.QCircuit()) -> the unitary from which the gate is to be folded
    param: Hamiltonian (tq.QubitHamiltonian()) -> the Hamiltonian for the system
    param; initial_values (dict) -> the initial values of the variables

    e.g.:
    input:
    number_of_gates_to_fold ->
    full_unitary ->
    Hamiltonian ->

    returns:
    energy (float) ->
    result (tq.OptimizerResults()) ->
    """
    #convert QubitHamiltonian into ParamQubitHamiltonian
    param_ham = convert_tq_QH_to_PQH(Hamiltonian)

    #checking if this is correct
    ham = convert_PQH_to_tq_QH(param_ham)({var:1.0 for var in full_unitary.extract_variables()})
    try:
        assert ham == Hamiltonian
    except:
        raise Exception("Hamiltonian conversion was unsuccessfull")

    #get the list of gates in the circuit
    gates = full_unitary.gates

    print("Constructing left unitary")
    #construct the unitary without the gates to be folded
    left_unitary = tq.QCircuit()
    for gate in gates[:-1*number_of_gates_to_fold]:
        left_unitary += gate
    print(left_unitary)
    print("Folding unitary")
    print("before ham was that long", len(ham))
    null_proj = None
    #fold the rest of  the gates into the Hamiltonian
    gates.reverse()
    for gate in gates[:number_of_gates_to_fold]:
        param_ham = fold_unitary_into_hamiltonian(gate, param_ham)

        #checking if folding works corretly
        """print("Folded Hamiltonian")
        for term in param_ham.terms:
            print(term, param_ham.terms[term]({var:1.0 for var in full_unitary.extract_variables()}))
        param_ham_conv = convert_PQH_to_tq_QH(param_ham)({var:1.0 for var in full_unitary.extract_variables()})

        variable, generator, null_proj = convert_unitary_to_pauli(gate)
        #print(variable, generator, null_proj)
        folded_hamiltonian = tq.QubitHamiltonian()

        #folding terms
        coeff = variable({var:1.0 for var in full_unitary.extract_variables()})
        temp_term = (np.cos(coeff)**2)*(Hamiltonian)
        folded_hamiltonian += temp_term
        temp_term = (np.sin(coeff)**2)*(generator*Hamiltonian*generator)
        folded_hamiltonian += temp_term
        temp_term = (np.sin(coeff)*np.cos(coeff)*1.0j)*(generator*Hamiltonian- Hamiltonian*generator)
        folded_hamiltonian += temp_term
        temp_term = ((1-np.cos(coeff))**2)*(null_proj*Hamiltonian*null_proj)
        folded_hamiltonian += temp_term
        temp_term = ((1-np.cos(coeff))*np.cos(coeff))*(null_proj*Hamiltonian+ Hamiltonian*null_proj)
        folded_hamiltonian += temp_term
        temp_term = ((1-np.cos(coeff))*np.sin(coeff)*1.0j)*(generator*Hamiltonian*null_proj- null_proj*Hamiltonian*generator)
        folded_hamiltonian += temp_term
        print("Manual folding")
        for val, term in folded_hamiltonian.items():
            print(val, term)

        #checking for errors
        print("Difference b/w manual and automatic folding")
        for val, terms in (param_ham_conv - folded_hamiltonian).items():
            print(val, terms)
            try:
                assert (np.absolute(terms) < 1e-7)
            except Exception as e:
                print(e)
                raise Exception("folding is unsuccessfull")"""

    ham = convert_PQH_to_tq_QH(param_ham)({var:1.0 for var in full_unitary.extract_variables()})
    print('after, ham is that long', len(ham))
    print("Running Optimization")
    result = run_optimization(left_unitary, param_ham, initial_values)

    return result.energy, result


if __name__ == "__main__":

    geometry = get_geometry("H2", 1.0)
    print(geometry)
    basis_set = 'sto-3g'
    #name = "/home/abhinav/matter_lab/moldata/h2/h2_1.00"
    #U, ham, fci_ener = get_ansatz_circuit("HEA", geometry, basis_set=basis_set, name=name, circuit_id=11)
    #tq.draw(U, backend="cirq")

    from tequila import gates as tq_g
    from tequila.objective.objective import Variable

    Ham =tq.paulis.X(3)* tq.paulis.Y(1)*tq.paulis.X(2)

    variable = Variable("z"+str(0))
    U0 = tq_g.CNOT(1,3)#z(angle=variable, target = 0, control=1)

    param_ham = convert_tq_QH_to_PQH(Ham)
    print(param_ham)

    folded_ham = (fold_unitary_into_hamiltonian(U0.gates[0],param_ham))
    print(folded_ham)

    U, Ham, fci_ener = get_ansatz_circuit("UCCSD", geometry, basis_set)
    print(U)
    print(fci_ener)
    #print(Ham)
    #print(uccsd_anz.extract_variables())
    #print(len(uccsd_anz.gates))

    Ham1 = tq.paulis.X(0) + tq.paulis.Z(1)
    U0 = tq.gates.X(0)
    a = tq.Variable("a")
    G1 = tq.paulis.Y(0)*tq.paulis.Y(1)
    U1 = tq.gates.GeneralizedRotation(generator=G1, angle=2.0*a)

    E = tq.ExpectationValue(H=Ham1, U =U0+U1 )
    result = tq.minimize(E, backend="qulacs", method='bfgs',initial_values = {k:1.0 for k in U1.extract_variables()} )
    print(result)
    print(tq.simulate(objective = U0+U1 , backend='qulacs', variables = result.variables))
    #raise Exception('testing')
    run_vqe_h(1,U0+U1,Ham1, {k:1.0 for k in U1.extract_variables()})

    E = tq.ExpectationValue(H=Ham, U =U )
    result = tq.minimize(E, backend="qulacs", method='bfgs',initial_values = {k:1.0 for k in U.extract_variables()} )
    print(result)
    print(tq.simulate(objective = U , backend='qulacs', variables = result.variables))
    #raise Exception('testing')
    run_vqe_h(1,U,Ham, {k:1.0 for k in U.extract_variables()})

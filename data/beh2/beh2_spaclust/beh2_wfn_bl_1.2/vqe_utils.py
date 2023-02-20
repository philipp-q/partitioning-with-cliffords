import tequila as tq
import numpy as np

from hacked_openfermion_qubit_operator import ParamQubitHamiltonian

from openfermion import QubitOperator
from HEA import *

def get_ansatz_circuit(ansatz_type, geometry, basis_set=None, trotter_steps = 1, name=None, circuit_id=None,num_layers=1):
    """
    This function generates the ansatz for the molecule and the Hamiltonian

    param: ansatz_type (str) -> the type of the ansatz ('UCCSD, UpCCGSD, SPA, HEA')
    param: geometry (str) -> the geometry of the molecule
    param: basis_set (str) -> the basis set for wchich the ansatz has to generated
    param: trotter_steps (int) -> the number of trotter step to be used in the
                                trotter decomposition
    param: name (str) -> the name for the madness molecule
    param: circuit_id (int) -> the type of hardware efficient ansatz
    param: num_layers (int) -> the number of layers of the HEA

    e.g.:
    input:
    ansatz_type -> "UCCSD"
    geometry -> "H 0.0 0.0 0.0\nH 0.0 0.0 0.714"
    basis_set -> 'sto-3g'
    trotter_steps -> 1
    name -> None
    circuit_id -> None
    num_layers -> 1

    returns:
    ucc_ansatz (tq.QCircuit()) -> a circuit printed below:
    circuit:
    FermionicExcitation(target=(0, 1, 2, 3), control=(), parameter=Objective with 0 unique expectation values
    total measurements = 0
    variables          = [(1, 0, 1, 0)]
    types              = [])
    FermionicExcitation(target=(0, 1, 2, 3), control=(), parameter=Objective with 0 unique expectation values
    total measurements = 0
    variables          = [(1, 0, 1, 0)]
    types              = [])

    Hamiltonian (tq.QubitHamiltonian()) -> -0.0621+0.1755Z(0)+0.1755Z(1)-0.2358Z(2)-0.2358Z(3)+0.1699Z(0)Z(1)
                                            +0.0449Y(0)X(1)X(2)Y(3)-0.0449Y(0)Y(1)X(2)X(3)-0.0449X(0)X(1)Y(2)Y(3)
                                            +0.0449X(0)Y(1)Y(2)X(3)+0.1221Z(0)Z(2)+0.1671Z(0)Z(3)+0.1671Z(1)Z(2)
                                            +0.1221Z(1)Z(3)+0.1756Z(2)Z(3)
    fci_ener (float) ->
    """
    ham = tq.QubitHamiltonian()
    fci_ener = 0.0
    ansatz = tq.QCircuit()
    if ansatz_type == "UCCSD":
        molecule = tq.Molecule(geometry=geometry, basis_set=basis_set)
        ansatz = molecule.make_uccsd_ansatz(trotter_steps)
        ham = molecule.make_hamiltonian()
        fci_ener = molecule.compute_energy(method="fci")
    elif ansatz_type == "UCCS":
        molecule = tq.Molecule(geometry=geometry, basis_set=basis_set, backend='psi4')
        ham = molecule.make_hamiltonian()
        fci_ener = molecule.compute_energy("fci")
        indices = molecule.make_upccgsd_indices(key='UCCS')
        print("indices are:", indices)
        ansatz = molecule.make_upccgsd_layer(indices=indices, include_singles=True, include_doubles=False)
    elif ansatz_type == "UpCCGSD":
        molecule = tq.Molecule(name=name, geometry=geometry, n_pno=None)
        ham = molecule.make_hamiltonian()
        fci_ener = molecule.compute_energy("fci")
        ansatz = molecule.make_upccgsd_ansatz()
    elif ansatz_type == "SPA":
        molecule = tq.Molecule(name=name, geometry=geometry, n_pno=None)
        ham = molecule.make_hamiltonian()
        fci_ener = molecule.compute_energy("fci")
        ansatz = molecule.make_upccgsd_ansatz(name="SPA")
    elif ansatz_type == "HEA":
        molecule = tq.Molecule(geometry=geometry, basis_set=basis_set)
        ham = molecule.make_hamiltonian()
        fci_ener = molecule.compute_energy(method="fci")
        ansatz = generate_HEA(molecule.n_orbitals * 2, circuit_id)
    else:
        raise Exception("not implemented any other ansatz, please choose from 'UCCSD, UpCCGSD, SPA, HEA'")

    return ansatz, ham, fci_ener

def get_generator_for_gates(unitary):
    """
    This function takes a unitary gate and returns the generator of the
    the gate so that it can be padded to the Hamiltonian

    param: unitary (tq.QGateImpl()) -> the unitary circuit element that has to be
                                    converted to a paulistring

    e.g.:
    input:
    unitary -> a FermionicGateImpl object as the one printed below

    FermionicExcitation(target=(0, 1, 2, 3), control=(), parameter=Objective with 0 unique expectation values
    total measurements = 0
    variables          = [(1, 0, 1, 0)]
    types              = [])

    returns:
    parameter (tq.Variable()) -> (1, 0, 1, 0)

    generator (tq.QubitHamiltonian()) -> -0.1250Y(0)Y(1)Y(2)X(3)+0.1250Y(0)X(1)Y(2)Y(3)
                                        +0.1250X(0)X(1)Y(2)X(3)+0.1250X(0)Y(1)Y(2)Y(3)
                                        -0.1250Y(0)X(1)X(2)X(3)-0.1250Y(0)Y(1)X(2)Y(3)
                                        -0.1250X(0)Y(1)X(2)X(3)+0.1250X(0)X(1)X(2)Y(3)

    null_proj (tq.QubitHamiltonian()) -> -0.1250Z(0)Z(1)+0.1250Z(1)Z(3)+0.1250Z(0)Z(3)
                                        +0.1250Z(1)Z(2)+0.1250Z(0)Z(2)-0.1250Z(2)Z(3)
                                        -0.1250Z(0)Z(1)Z(2)Z(3)
    """
    try:
        parameter = None
        generator = None
        null_proj = None
        if isinstance(unitary, tq.quantumchemistry.qc_base.FermionicGateImpl):
            parameter = unitary.extract_variables()
            generator = unitary.generator
            null_proj = unitary.p0
        else:
            #getting parameter
            if unitary.is_parametrized():
                parameter = unitary.extract_variables()
            else:
                parameter = [None]

            try:
                generator = unitary.make_generator(include_controls=True)
            except:
                generator = unitary.generator()

            """if len(parameter) == 0:
                parameter = [tq.objective.objective.assign_variable(unitary.parameter)]"""
        return parameter[0], generator, null_proj
    except Exception as e:
        print("An Exception happened, details :",e)
        pass

def fold_unitary_into_hamiltonian(unitary, Hamiltonian):
    """
    This function return a list of the resulting Hamiltonian terms after folding the paulistring
    correspondig to the unitary into the Hamiltonian

    param: unitary (tq.QGateImpl()) -> the unitary to be folded into the Hamiltonian
    param: Hamiltonian (ParamQubitHamiltonian()) -> the Hamiltonian of the system

    e.g.:
    input:
    unitary -> a FermionicGateImpl object as the one printed below

    FermionicExcitation(target=(0, 1, 2, 3), control=(), parameter=Objective with 0 unique expectation values
    total measurements = 0
    variables          = [(1, 0, 1, 0)]
    types              = [])

    Hamiltonian -> -0.06214952615456104 [] + -0.044941923860490916 [X0 X1 Y2 Y3] +
                0.044941923860490916 [X0 Y1 Y2 X3] + 0.044941923860490916 [Y0 X1 X2 Y3] +
                -0.044941923860490916 [Y0 Y1 X2 X3] + 0.17547360045040505 [Z0] +
                0.16992958569230643 [Z0 Z1] + 0.12212314332112947 [Z0 Z2] +
                0.1670650671816204 [Z0 Z3] + 0.17547360045040508 [Z1] +
                0.1670650671816204 [Z1 Z2] + 0.12212314332112947 [Z1 Z3] +
                -0.23578915712819945 [Z2] + 0.17561918557144712 [Z2 Z3] +
                -0.23578915712819945 [Z3]


    returns:
    folded_hamiltonian (ParamQubitHamiltonian()) -> Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [X0 X1 X2 X3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [X0 X1 Y2 Y3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [X0 Y1 X2 Y3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [X0 Y1 Y2 X3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Y0 X1 X2 Y3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Y0 X1 Y2 X3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Y0 Y1 X2 X3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Y0 Y1 Y2 Y3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z0] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z0 Z1] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z0 Z1 Z2] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z0 Z1 Z2 Z3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z0 Z1 Z3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z0 Z2] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z0 Z2 Z3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z0 Z3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z1] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z1 Z2] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z1 Z2 Z3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z1 Z3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z2] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z2 Z3] +
                                                    Objective with 0 unique expectation values
                                                    total measurements = 0
                                                    variables          = [(1, 0, 1, 0)]
                                                    types              = [] [Z3]
    """
    folded_hamiltonian = ParamQubitHamiltonian()
    if isinstance(unitary, tq.circuit._gates_impl.DifferentiableGateImpl) and not isinstance(unitary, tq.circuit._gates_impl.PhaseGateImpl):

        variable, generator, null_proj = get_generator_for_gates(unitary)
        # print(generator)
        # print(null_proj)
        #converting into ParamQubitHamiltonian()
        c_generator = convert_tq_QH_to_PQH(generator)
        """c_null_proj = convert_tq_QH_to_PQH(null_proj)
        print(variable)

        prod1 = (null_proj*ham*generator - generator*ham*null_proj)
        print(prod1)

        #print((Hamiltonian*c_generator))
        prod2 = convert_PQH_to_tq_QH(c_null_proj*Hamiltonian*c_generator - c_generator*Hamiltonian*c_null_proj)({var:0.1 for var in variable.extract_variables()})
        print(prod2)
        assert prod1 ==prod2
        raise Exception("testing")"""
        #print("starting", flush=True)

        #handling the parameterize gates
        if variable is not None:
            #print("folding generator")
            # adding the term: cos^2(\theta)*H
            temp_ham = ParamQubitHamiltonian().identity()
            temp_ham *= Hamiltonian
            temp_ham *= (variable.apply(tq.numpy.cos)**2)
            #print(convert_PQH_to_tq_QH(temp_ham)({var:1. for var in variable.extract_variables()}))
            folded_hamiltonian += temp_ham
            #print("step1 done", flush=True)

            # adding the term: sin^2(\theta)*G*H*G
            temp_ham = ParamQubitHamiltonian().identity()
            temp_ham *= c_generator
            temp_ham *= Hamiltonian
            temp_ham *= c_generator
            temp_ham *= (variable.apply(tq.numpy.sin)**2)
            #print(convert_PQH_to_tq_QH(temp_ham)({var:1. for var in variable.extract_variables()}))
            folded_hamiltonian += temp_ham
            #print("step2 done", flush=True)

            # adding the term: i*cos^2(\theta)8sin^2(\theta)*(G*H -H*G)
            temp_ham1 = ParamQubitHamiltonian().identity()
            temp_ham1 *= c_generator
            temp_ham1 *= Hamiltonian
            temp_ham2 = ParamQubitHamiltonian().identity()
            temp_ham2 *= Hamiltonian
            temp_ham2 *= c_generator
            temp_ham = temp_ham1 - temp_ham2
            temp_ham *= 1.0j
            temp_ham *= variable.apply(tq.numpy.sin) * variable.apply(tq.numpy.cos)
            #print(convert_PQH_to_tq_QH(temp_ham)({var:1. for var in variable.extract_variables()}))
            folded_hamiltonian += temp_ham
            #print("step3 done", flush=True)
        #handling the non-paramterized gates
        else:
            raise Exception("This function is not implemented yet")
            # adding the term: G*H*G
            folded_hamiltonian += c_generator*Hamiltonian*c_generator
        #print("Halfway there", flush=True)

        #handle the FermionicGateImpl gates
        if null_proj is not None:
            print("folding null projector")
            c_null_proj = convert_tq_QH_to_PQH(null_proj)
            #print("step4 done", flush=True)

            # adding the term: (1-cos(\theta))^2*P0*H*P0
            temp_ham = ParamQubitHamiltonian().identity()
            temp_ham *= c_null_proj
            temp_ham *= Hamiltonian
            temp_ham *= c_null_proj
            temp_ham *= ((1-variable.apply(tq.numpy.cos))**2)
            folded_hamiltonian += temp_ham
            #print("step5 done", flush=True)

            # adding the term: 2*cos(\theta)*(1-cos(\theta))*(P0*H +H*P0)
            temp_ham1 = ParamQubitHamiltonian().identity()
            temp_ham1 *= c_null_proj
            temp_ham1 *= Hamiltonian
            temp_ham2 = ParamQubitHamiltonian().identity()
            temp_ham2 *= Hamiltonian
            temp_ham2 *= c_null_proj
            temp_ham = temp_ham1 + temp_ham2
            temp_ham *= (variable.apply(tq.numpy.cos)*(1-variable.apply(tq.numpy.cos)))
            folded_hamiltonian += temp_ham
            #print("step6 done", flush=True)

            # adding the term: i*sin(\theta)*(1-cos(\theta))*(G*H*P0 - P0*H*G)
            temp_ham1 = ParamQubitHamiltonian().identity()
            temp_ham1 *= c_generator
            temp_ham1 *= Hamiltonian
            temp_ham1 *= c_null_proj
            temp_ham2 = ParamQubitHamiltonian().identity()
            temp_ham2 *= c_null_proj
            temp_ham2 *= Hamiltonian
            temp_ham2 *= c_generator
            temp_ham = temp_ham1 - temp_ham2
            temp_ham *= 1.0j
            temp_ham *= (variable.apply(tq.numpy.sin)*(1-variable.apply(tq.numpy.cos)))
            folded_hamiltonian += temp_ham
            #print("step7 done", flush=True)

    elif isinstance(unitary, tq.circuit._gates_impl.PhaseGateImpl):
        if np.isclose(unitary.parameter, np.pi/2.0):
            return Hamiltonian._clifford_simplify_s(unitary.qubits[0])
        elif np.isclose(unitary.parameter, -1.*np.pi/2.0):
            return Hamiltonian._clifford_simplify_s_dag(unitary.qubits[0])
        else:
            raise Exception("Only DifferentiableGateImpl, PhaseGateImpl(S), Pauligate(X,Y,Z), Controlled(X,Y,Z) and Hadamrd(H) implemented yet")
    elif isinstance(unitary, tq.circuit._gates_impl.QGateImpl):
        if unitary.is_controlled():
            if unitary.name == "X":
                return Hamiltonian._clifford_simplify_control_g("X", unitary.control[0], unitary.target[0])
            elif unitary.name == "Y":
                return Hamiltonian._clifford_simplify_control_g("Y", unitary.control[0], unitary.target[0])
            elif unitary.name == "Z":
                return Hamiltonian._clifford_simplify_control_g("Z", unitary.control[0], unitary.target[0])
            else:
                raise Exception("Only DifferentiableGateImpl, PhaseGateImpl(S), Pauligate(X,Y,Z), Controlled(X,Y,Z) and Hadamrd(H) implemented yet")
        else:
            if unitary.name == "X":
                gate = convert_tq_QH_to_PQH(tq.paulis.X(unitary.qubits[0]))
                return gate*Hamiltonian*gate
            elif unitary.name == "Y":
                gate = convert_tq_QH_to_PQH(tq.paulis.Y(unitary.qubits[0]))
                return gate*Hamiltonian*gate
            elif unitary.name == "Z":
                gate = convert_tq_QH_to_PQH(tq.paulis.Z(unitary.qubits[0]))
                return gate*Hamiltonian*gate
            elif unitary.name == "H":
                return Hamiltonian._clifford_simplify_h(unitary.qubits[0])
            else:
                raise Exception("Only DifferentiableGateImpl, PhaseGateImpl(S), Pauligate(X,Y,Z), Controlled(X,Y,Z) and Hadamrd(H) implemented yet")
    else:
        raise Exception("Only DifferentiableGateImpl, PhaseGateImpl(S), Pauligate(X,Y,Z), Controlled(X,Y,Z) and Hadamrd(H) implemented yet")
    return folded_hamiltonian


def convert_tq_QH_to_PQH(Hamiltonian):
    """
    This function takes the tequila QubitHamiltonian object and converts into a
    ParamQubitHamiltonian object.

    param: Hamiltonian (tq.QubitHamiltonian()) -> the Hamiltonian to be converted

    e.g:
    input:
    Hamiltonian -> -0.0621+0.1755Z(0)+0.1755Z(1)-0.2358Z(2)-0.2358Z(3)+0.1699Z(0)Z(1)
                    +0.0449Y(0)X(1)X(2)Y(3)-0.0449Y(0)Y(1)X(2)X(3)-0.0449X(0)X(1)Y(2)Y(3)
                    +0.0449X(0)Y(1)Y(2)X(3)+0.1221Z(0)Z(2)+0.1671Z(0)Z(3)+0.1671Z(1)Z(2)
                    +0.1221Z(1)Z(3)+0.1756Z(2)Z(3)

    returns:
    param_hamiltonian (ParamQubitHamiltonian()) -> -0.06214952615456104 [] +
                                                    -0.044941923860490916 [X0 X1 Y2 Y3] +
                                                    0.044941923860490916 [X0 Y1 Y2 X3] +
                                                    0.044941923860490916 [Y0 X1 X2 Y3] +
                                                    -0.044941923860490916 [Y0 Y1 X2 X3] +
                                                    0.17547360045040505 [Z0] +
                                                    0.16992958569230643 [Z0 Z1] +
                                                    0.12212314332112947 [Z0 Z2] +
                                                    0.1670650671816204 [Z0 Z3] +
                                                    0.17547360045040508 [Z1] +
                                                    0.1670650671816204 [Z1 Z2] +
                                                    0.12212314332112947 [Z1 Z3] +
                                                    -0.23578915712819945 [Z2] +
                                                    0.17561918557144712 [Z2 Z3] +
                                                    -0.23578915712819945 [Z3]
    """
    param_hamiltonian = ParamQubitHamiltonian()
    Hamiltonian = Hamiltonian.to_openfermion()
    for term in Hamiltonian.terms:
        param_hamiltonian += ParamQubitHamiltonian(term = term, coefficient = Hamiltonian.terms[term])
    return param_hamiltonian

class convert_PQH_to_tq_QH:
    def __init__(self, Hamiltonian):
        self.param_hamiltonian = Hamiltonian

    def __call__(self,variables=None):
        """
        This function takes the ParamQubitHamiltonian object and converts into a
        tequila QubitHamiltonian object.

        param: param_hamiltonian (ParamQubitHamiltonian()) -> the Hamiltonian to be converted
        param: variables (dict) -> a dictionary with the values of the variables in the
                                    Hamiltonian coefficient

        e.g:
        input:
        param_hamiltonian -> a [Y0 X2 Z3] + b [Z0 X2 Z3]
        variables -> {"a":1,"b":2}

        returns:
        Hamiltonian (tq.QubitHamiltonian()) -> +1.0000Y(0)X(2)Z(3)+2.0000Z(0)X(2)Z(3)
        """
        Hamiltonian = tq.QubitHamiltonian()
        for term in self.param_hamiltonian.terms:
            val = self.param_hamiltonian.terms[term]# + self.param_hamiltonian.imag_terms[term]*1.0j
            if isinstance(val, tq.Variable) or isinstance(val, tq.objective.objective.Objective):
                try:
                    for key in variables.keys():
                        variables[key] = variables[key]
                    #print(variables)
                    val = val(variables)
                    #print(val)
                except Exception as e:
                    print(e)
                    raise Exception("You forgot to pass the dictionary with the values of the variables")
            Hamiltonian += tq.QubitHamiltonian(QubitOperator(term=term, coefficient=val))
        return Hamiltonian

    def _construct_derivatives(self, variables=None):
        """

        """
        derivatives = {}
        variable_names = []
        for term in self.param_hamiltonian.terms:
            val = self.param_hamiltonian.terms[term]# + self.param_hamiltonian.imag_terms[term]*1.0j
            #print(val)
            if isinstance(val, tq.Variable) or isinstance(val, tq.objective.objective.Objective):
                variable = val.extract_variables()
                for var in list(variable):
                    from grad_hacked import grad
                    derivative = ParamQubitHamiltonian(term = term, coefficient = grad(val, var))
                    if var not in variable_names:
                        variable_names.append(var)
                    if var not in list(derivatives.keys()):
                        derivatives[var] = derivative
                    else:
                        derivatives[var] += derivative
        return variable_names, derivatives

def get_geometry(name, b_l):
    """
    This is utility fucntion that generates tehe geometry string of a Molecule

    param: name (str) -> name of the molecule
    param: b_l (float) -> the bond length of the molecule

    e.g.:
    input:
    name -> "H2"
    b_l -> 0.714

    returns:
    geo_str (str) -> "H 0.0 0.0 0.0\nH 0.0 0.0 0.714"
    """
    geo_str = None
    if name == "LiH":
        geo_str = "H 0.0 0.0 0.0\nLi 0.0 0.0 {0}".format(b_l)
    elif name == "H2":
        geo_str = "H 0.0 0.0 0.0\nH 0.0 0.0 {0}".format(b_l)
    elif name == "BeH2":
        geo_str = "H 0.0 0.0 {0}\nH 0.0 0.0 {1}\nBe 0.0 0.0 0.0".format(b_l,-1*b_l)
    elif name == "N2":
        geo_str = "N 0.0 0.0 0.0\nN 0.0 0.0 {0}".format(b_l)
    elif name == "H4":
        geo_str = "H 0.0 0.0 0.0\nH 0.0 0.0 {0}\nH 0.0 0.0 {1}\nH 0.0 0.0 {2}".format(b_l,-1*b_l,2*b_l)
    return geo_str

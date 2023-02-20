import tequila as tq
import sympy
import copy

#from param_hamiltonian import get_geometry, generate_ucc_ansatz

from hacked_openfermion_symbolic_operator import SymbolicOperator

# Define products of all Pauli operators for symbolic multiplication.
_PAULI_OPERATOR_PRODUCTS = {
    ('I', 'I'): (1., 'I'),
    ('I', 'X'): (1., 'X'),
    ('X', 'I'): (1., 'X'),
    ('I', 'Y'): (1., 'Y'),
    ('Y', 'I'): (1., 'Y'),
    ('I', 'Z'): (1., 'Z'),
    ('Z', 'I'): (1., 'Z'),
    ('X', 'X'): (1., 'I'),
    ('Y', 'Y'): (1., 'I'),
    ('Z', 'Z'): (1., 'I'),
    ('X', 'Y'): (1.j, 'Z'),
    ('X', 'Z'): (-1.j, 'Y'),
    ('Y', 'X'): (-1.j, 'Z'),
    ('Y', 'Z'): (1.j, 'X'),
    ('Z', 'X'): (1.j, 'Y'),
    ('Z', 'Y'): (-1.j, 'X')
}

_clifford_h_products = {
    ('I') : (1., 'I'),
    ('X') : (1., 'Z'),
    ('Y') : (-1., 'Y'),
    ('Z') : (1., 'X')
}

_clifford_s_products = {
    ('I') : (1., 'I'),
    ('X') : (-1., 'Y'),
    ('Y') : (1., 'X'),
    ('Z') : (1., 'Z')
}

_clifford_s_dag_products = {
    ('I') : (1., 'I'),
    ('X') : (1., 'Y'),
    ('Y') : (-1., 'X'),
    ('Z') : (1., 'Z')
}

_clifford_cx_products = {
    ('I', 'I'): (1., 'I', 'I'),
    ('I', 'X'): (1., 'I', 'X'),
    ('I', 'Y'): (1., 'Z', 'Y'),
    ('I', 'Z'): (1., 'Z', 'Z'),
    ('X', 'I'): (1., 'X', 'X'),
    ('X', 'X'): (1., 'X', 'I'),
    ('X', 'Y'): (1., 'Y', 'Z'),
    ('X', 'Z'): (-1., 'Y', 'Y'),
    ('Y', 'I'): (1., 'Y', 'X'),
    ('Y', 'X'): (1., 'Y', 'I'),
    ('Y', 'Y'): (-1., 'X', 'Z'),
    ('Y', 'Z'): (1., 'X', 'Y'),
    ('Z', 'I'): (1., 'Z', 'I'),
    ('Z', 'X'): (1., 'Z', 'X'),
    ('Z', 'Y'): (1., 'I', 'Y'),
    ('Z', 'Z'): (1., 'I', 'Z'),
}

_clifford_cy_products = {
    ('I', 'I'): (1., 'I', 'I'),
    ('I', 'X'): (1., 'Z', 'X'),
    ('I', 'Y'): (1., 'I', 'Y'),
    ('I', 'Z'): (1., 'Z', 'Z'),
    ('X', 'I'): (1., 'X', 'Y'),
    ('X', 'X'): (-1., 'Y', 'Z'),
    ('X', 'Y'): (1., 'X', 'I'),
    ('X', 'Z'): (-1., 'Y', 'X'),
    ('Y', 'I'): (1., 'Y', 'Y'),
    ('Y', 'X'): (1., 'X', 'Z'),
    ('Y', 'Y'): (1., 'Y', 'I'),
    ('Y', 'Z'): (-1., 'X', 'X'),
    ('Z', 'I'): (1., 'Z', 'I'),
    ('Z', 'X'): (1., 'I', 'X'),
    ('Z', 'Y'): (1., 'Z', 'Y'),
    ('Z', 'Z'): (1., 'I', 'Z'),
}

_clifford_cz_products = {
    ('I', 'I'): (1., 'I', 'I'),
    ('I', 'X'): (1., 'Z', 'X'),
    ('I', 'Y'): (1., 'Z', 'Y'),
    ('I', 'Z'): (1., 'I', 'Z'),
    ('X', 'I'): (1., 'X', 'Z'),
    ('X', 'X'): (-1., 'Y', 'Y'),
    ('X', 'Y'): (-1., 'Y', 'X'),
    ('X', 'Z'): (1., 'X', 'I'),
    ('Y', 'I'): (1., 'Y', 'Z'),
    ('Y', 'X'): (-1., 'X', 'Y'),
    ('Y', 'Y'): (1., 'X', 'X'),
    ('Y', 'Z'): (1., 'Y', 'I'),
    ('Z', 'I'): (1., 'Z', 'I'),
    ('Z', 'X'): (1., 'I', 'X'),
    ('Z', 'Y'): (1., 'I', 'Y'),
    ('Z', 'Z'): (1., 'Z', 'Z'),
}

COEFFICIENT_TYPES = (int, float, complex, sympy.Expr, tq.Variable)

class ParamQubitHamiltonian(SymbolicOperator):

    @property
    def actions(self):
        """The allowed actions."""
        return ('X', 'Y', 'Z')

    @property
    def action_strings(self):
        """The string representations of the allowed actions."""
        return ('X', 'Y', 'Z')

    @property
    def action_before_index(self):
        """Whether action comes before index in string representations."""
        return True

    @property
    def different_indices_commute(self):
        """Whether factors acting on different indices commute."""
        return True

    def renormalize(self):
        """Fix the trace norm of an operator to 1"""
        norm = self.induced_norm(2)
        if numpy.isclose(norm, 0.0):
            raise ZeroDivisionError('Cannot renormalize empty or zero operator')
        else:
            self /= norm

    def _simplify(self, term, coefficient=1.0):
        """Simplify a term using commutator and anti-commutator relations."""
        if not term:
            return coefficient, term

        term = sorted(term, key=lambda factor: factor[0])

        new_term = []
        left_factor = term[0]
        for right_factor in term[1:]:
            left_index, left_action = left_factor
            right_index, right_action = right_factor

            # Still on the same qubit, keep simplifying.
            if left_index == right_index:
                new_coefficient, new_action = _PAULI_OPERATOR_PRODUCTS[
                    left_action, right_action]
                left_factor = (left_index, new_action)
                coefficient *= new_coefficient

            # Reached different qubit, save result and re-initialize.
            else:
                if left_action != 'I':
                    new_term.append(left_factor)
                left_factor = right_factor

        # Save result of final iteration.
        if left_factor[1] != 'I':
            new_term.append(left_factor)

        return coefficient, tuple(new_term)

    def _clifford_simplify_h(self, qubit):
        """simplifying the Hamiltonian using the clifford group property"""

        fold_ham = {}
        for term in self.terms:
            #there should be a better way to do this
            new_term = []
            coeff = 1.0
            for left, right in term:
                if left == qubit:
                    coeff, new_pauli = _clifford_h_products[right]
                    new_term.append(tuple((left, new_pauli)))
                else:
                    new_term.append(tuple((left,right)))

            fold_ham[tuple(new_term)] = coeff*self.terms[term]
        self.terms = fold_ham
        return self

    def _clifford_simplify_s(self, qubit):
        """simplifying the Hamiltonian using the clifford group property"""

        fold_ham = {}
        for term in self.terms:
            #there should be a better way to do this
            new_term = []
            coeff = 1.0
            for left, right in term:
                if left == qubit:
                    coeff, new_pauli = _clifford_s_products[right]
                    new_term.append(tuple((left, new_pauli)))
                else:
                    new_term.append(tuple((left,right)))

            fold_ham[tuple(new_term)] = coeff*self.terms[term]
        self.terms = fold_ham
        return self

    def _clifford_simplify_s_dag(self, qubit):
        """simplifying the Hamiltonian using the clifford group property"""

        fold_ham = {}
        for term in self.terms:
            #there should be a better way to do this
            new_term = []
            coeff = 1.0
            for left, right in term:
                if left == qubit:
                    coeff, new_pauli = _clifford_s_dag_products[right]
                    new_term.append(tuple((left, new_pauli)))
                else:
                    new_term.append(tuple((left,right)))

            fold_ham[tuple(new_term)] = coeff*self.terms[term]
        self.terms = fold_ham
        return self

    def _clifford_simplify_control_g(self, axis, control_q, target_q):
        """simplifying the Hamiltonian using the clifford group property"""

        fold_ham = {}
        for term in self.terms:
            #there should be a better way to do this
            new_term = []
            coeff = 1.0
            target = "I"
            control = "I"
            for left, right in term:
                if left == control_q:
                    control = right
                elif left == target_q:
                    target = right
                else:
                    new_term.append(tuple((left,right)))
            new_c = "I"
            new_t = "I"
            if not (target == "I" and control == "I"):
                if axis == "X":
                    coeff, new_c, new_t = _clifford_cx_products[control, target]
                if axis == "Y":
                    coeff, new_c, new_t = _clifford_cy_products[control, target]
                if axis == "Z":
                    coeff, new_c, new_t = _clifford_cz_products[control, target]

            if new_c != "I":
                new_term.append(tuple((control_q, new_c)))
            if new_t != "I":
                new_term.append(tuple((target_q, new_t)))

            new_term = sorted(new_term, key=lambda factor: factor[0])
            fold_ham[tuple(new_term)] = coeff*self.terms[term]
        self.terms = fold_ham
        return self


if __name__ == "__main__":
    """geometry = get_geometry("H2", 0.714)
    print(geometry)
    basis_set = 'sto-3g'
    ref_anz, uccsd_anz, ham = generate_ucc_ansatz(geometry, basis_set)
    print(ham)
    b_ham = tq.grouping.binary_rep.BinaryHamiltonian.init_from_qubit_hamiltonian(ham)
    c_ham = tq.grouping.binary_rep.BinaryHamiltonian.init_from_qubit_hamiltonian(ham)
    print(b_ham)
    print(b_ham.get_binary())
    print(b_ham.get_coeff())
    param = uccsd_anz.extract_variables()
    print(param)
    for term in b_ham.binary_terms:
        print(term.coeff)
        term.set_coeff(param[0])
        print(term.coeff)
    print(b_ham.get_coeff())
    d_ham = c_ham.to_qubit_hamiltonian() + b_ham.to_qubit_hamiltonian()
    """
    term = [(2,'X'), (0,'Y'), (3, 'Z')]
    coeff = tq.Variable("a")
    coeff = coeff *2.j
    print(coeff)
    print(type(coeff))
    print(coeff({"a":1}))
    ham = ParamQubitHamiltonian(term= term, coefficient=coeff)
    print(ham.terms)
    print(str(ham))
    for term in ham.terms:
        print(ham.terms[term]({"a":1,"b":2}))
    term = [(2,'X'), (0,'Z'), (3, 'Z')]
    coeff = tq.Variable("b")
    print(coeff({"b":1}))
    b_ham = ParamQubitHamiltonian(term= term, coefficient=coeff)
    print(b_ham.terms)
    print(str(b_ham))
    for term in b_ham.terms:
        print(b_ham.terms[term]({"a":1,"b":2}))

    coeff = tq.Variable("a")*tq.Variable("b")
    print(coeff)
    print(coeff({"a":1,"b":2}))


    ham *= b_ham

    print(ham.terms)
    print(str(ham))
    for term in ham.terms:
        coeff = (ham.terms[term])
        print(coeff)
        print(coeff({"a":1,"b":2}))

    ham = ham*2.

    print(ham.terms)
    print(str(ham))

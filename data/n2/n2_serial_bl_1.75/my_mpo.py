import numpy as np
import tensornetwork as tn
from tensornetwork.backends.abstract_backend import AbstractBackend
tn.set_default_backend("pytorch")
#tn.set_default_backend("numpy")

from typing import List, Union, Text, Optional, Any, Type
Tensor = Any

import tequila as tq
import torch

EPS = 1e-12


class SubOperator:
    """
    This is just a helper class to store coefficient,
    operators and positions in an intermediate format
    """

    def __init__(self,
                 coefficient: float,
                 operators: List,
                 positions: List
                 ):
        self._coefficient = coefficient
        self._operators = operators
        self._positions = positions

    @property
    def coefficient(self):
        return self._coefficient

    @property
    def operators(self):
        return self._operators

    @property
    def positions(self):
        return self._positions


class MPOContainer:
    """
    Class that handles the MPO. Is able to set values at certain positions,
    update containers (wannabe-equivalent to dynamic arrays) and compress the MPO
    """

    def __init__(self,
                 n_qubits: int,
                 ):
        self.n_qubits = n_qubits

        self.container = [ np.zeros((1,1,2,2), dtype=np.complex)
                           for q in range(self.n_qubits) ]

    def get_dim(self):
        """ Returns max dimension of container """
        d = 1
        for q in range(len(self.container)):
            d = max(d, self.container[q].shape[0])

        return d

    def set_tensor(self, qubit: int, set_at: list, add_operator: Union[np.ndarray, float]):
        """
        set_at: where to put data
        """

        # Set a matrix
        if len(set_at) == 2:
            self.container[qubit][set_at[0],set_at[1],:,:] = add_operator[:,:]
        # Set specific values
        elif len(set_at) == 4:
            self.container[qubit][set_at[0],set_at[1],set_at[2],set_at[3]] =\
                                                                       add_operator
        else:
            raise Exception("set_at needs to be either of length 2 or 4")

    def update_container(self, qubit: int, update_dir: list, add_operator: np.ndarray):
        """
        This should mimick a dynamic array
        update_dir: e.g. [1,1,0,0] -> extend dimension along where there's a 1
                   the last two dimensions are always 2x2 only
        """
        old_shape = self.container[qubit].shape
        # print(old_shape)
        if not len(update_dir) == 4:
            if len(update_dir) == 2:
                update_dir += [0, 0]
            else:
                raise Exception("update_dir needs to be either of length 2 or 4")
        if update_dir[2] or update_dir[3]:
            raise Exception("Last two dims must be zero.")
        new_shape = tuple(update_dir[i]+old_shape[i] for i in range(len(update_dir)))
        new_tensor = np.zeros(new_shape, dtype=np.complex)

        # Copy old values
        new_tensor[:old_shape[0],:old_shape[1],:,:] = self.container[qubit][:,:,:,:]
        # Add new values
        new_tensor[new_shape[0]-1,new_shape[1]-1,:,:] = add_operator[:,:]

        # Overwrite container
        self.container[qubit] = new_tensor

    def compress_mpo(self):
       """
       Compression of MPO via SVD
       """
       n_qubits = len(self.container)

       for q in range(n_qubits):
           my_shape = self.container[q].shape
           self.container[q] =\
               self.container[q].reshape((my_shape[0], my_shape[1], -1))

       # Go forwards
       for q in range(n_qubits-1):
           # Apply permutation [0 1 2] -> [0 2 1]
           my_tensor = np.swapaxes(self.container[q], 1, 2)
           my_tensor = my_tensor.reshape((-1, my_tensor.shape[2]))
           # full_matrices flag corresponds to 'econ' -> no zero-singular values
           u, s, vh = np.linalg.svd(my_tensor, full_matrices=False)
           # Count the non-zero singular values
           num_nonzeros = len(np.argwhere(s>EPS))
           # Construct matrix from square root of singular values
           s = np.diag(np.sqrt(s[:num_nonzeros]))
           u = u[:,:num_nonzeros]
           vh = vh[:num_nonzeros,:]
           # Distribute weights to left- and right singular vectors (@ = np.matmul)
           u = u @ s
           vh = s @ vh

           # Apply permutation [0 1 2] -> [0 2 1]
           u = u.reshape((self.container[q].shape[0],\
                          self.container[q].shape[2], -1))
           self.container[q] = np.swapaxes(u, 1, 2)
           self.container[q+1] = tn.ncon([vh, self.container[q+1]], [(-1, 1),(1, -2, -3)])

       # Go backwards
       for q in range(n_qubits-1, 0, -1):
           my_tensor = self.container[q]
           my_tensor = my_tensor.reshape((self.container[q].shape[0], -1))
           # full_matrices flag corresponds to 'econ' -> no zero-singular values
           u, s, vh = np.linalg.svd(my_tensor, full_matrices=False)
           # Count the non-zero singular values
           num_nonzeros = len(np.argwhere(s>EPS))
           # Construct matrix from square root of singular values
           s = np.diag(np.sqrt(s[:num_nonzeros]))
           u = u[:,:num_nonzeros]
           vh = vh[:num_nonzeros,:]
           # Distribute weights to left- and right singular vectors
           u = u @ s
           vh = s @ vh

           self.container[q] = np.reshape(vh, (num_nonzeros,
                                               self.container[q].shape[1],
                                               self.container[q].shape[2]))
           self.container[q-1] = tn.ncon([self.container[q-1], u], [(-1, 1, -3),(1, -2)])

       for q in range(n_qubits):
           my_shape = self.container[q].shape
           self.container[q] = self.container[q].reshape((my_shape[0],\
                                                      my_shape[1],2,2))

# TODO maybe make subclass of tn.FiniteMPO if it makes sense
#class my_MPO(tn.FiniteMPO):
class MyMPO:
    """
    Class building up on tensornetwork FiniteMPO to handle
    MPO-Hamiltonians
    """

    def __init__(self,
                 hamiltonian: Union[tq.QubitHamiltonian, Text],
                 # tensors: List[Tensor],
                 backend: Optional[Union[AbstractBackend, Text]] = None,
                 n_qubits: Optional[int] = None,
                 name: Optional[Text] = None,
                 maxdim: Optional[int] = 10000) -> None:
        # TODO: modifiy docstring
        """
        Initialize a finite MPO object
        Args:
          tensors: The mpo tensors.
          backend: An optional backend. Defaults to the defaulf backend
            of TensorNetwork.
          name: An optional name for the MPO.
        """

        self.hamiltonian = hamiltonian
        self.maxdim = maxdim
        if n_qubits:
            self._n_qubits = n_qubits
        else:
            self._n_qubits = self.get_n_qubits()

    @property
    def n_qubits(self):
        return self._n_qubits

    def make_mpo_from_hamiltonian(self):

        intermediate = self.openfermion_to_intermediate()
        # for i in range(len(intermediate)):
        #     print(intermediate[i].coefficient)
        #     print(intermediate[i].operators)
        #     print(intermediate[i].positions)
        self.mpo = self.intermediate_to_mpo(intermediate)


    def openfermion_to_intermediate(self):
        # Here, have either a QubitHamiltonian or a file with a of-operator
        # Start with Qubithamiltonian

        def get_pauli_matrix(string):
            pauli_matrices = {
              'I': np.array([[1, 0], [0, 1]], dtype=np.complex),
              'Z': np.array([[1, 0], [0, -1]], dtype=np.complex),
              'X': np.array([[0, 1], [1, 0]], dtype=np.complex),
              'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex)
              }

            return pauli_matrices[string.upper()]

        intermediate = []

        first = True
        # Store all paulistrings in intermediate format
        for paulistring in self.hamiltonian.paulistrings:
            coefficient = paulistring.coeff
            # print(coefficient)
            operators = []
            positions = []
            # Only first one should be identity -> distribute over all
            if first and not paulistring.items():
                positions += []
                operators += []
                first = False
            elif not first and not paulistring.items():
                raise Exception("Only first Pauli should be identity.")
            # Get operators and where they act
            for k,v in paulistring.items():
                    positions += [k]
                    operators += [get_pauli_matrix(v)]

            tmp_op = SubOperator(coefficient=coefficient, operators=operators, positions=positions)
            intermediate += [tmp_op]

        # print("len intermediate = num Pauli strings", len(intermediate))
        return intermediate



    def build_single_mpo(self, intermediate, j):
        # Set MPO Container
        n_qubits = self._n_qubits
        mpo = MPOContainer(n_qubits=n_qubits)
        # ***********************************************************************
        # Set first entries (of which we know that they are 2x2-matrices)
        # Typically, this is an identity
        my_coefficient = intermediate[j].coefficient
        my_positions = intermediate[j].positions
        my_operators = intermediate[j].operators
        for q in range(n_qubits):
           if not q in my_positions:
               mpo.set_tensor(qubit=q, set_at=[0,0],
                              add_operator=np.complex(my_coefficient)**(1/n_qubits)*
                                           np.eye(2))

           elif q in my_positions:
               my_pos_index = my_positions.index(q)
               mpo.set_tensor(qubit=q, set_at=[0,0],
                          add_operator=np.complex(my_coefficient)**(1/n_qubits)*
                                                  my_operators[my_pos_index])
        # ***********************************************************************
        # All other entries
        # while (j smaller than number of intermediates left) and mpo.dim() <= self.maxdim
        # Re-write this based on positions keyword!
        j += 1
        while j < len(intermediate) and mpo.get_dim() < self.maxdim:
            # """
            my_coefficient = intermediate[j].coefficient
            my_positions = intermediate[j].positions
            my_operators = intermediate[j].operators
            for q in range(n_qubits):
                # It is guaranteed that every index appears only once in positions
                if q == 0:
                    update_dir = [0,1]
                elif q == n_qubits-1:
                    update_dir = [1,0]
                else:
                    update_dir = [1,1]
                # If there's an operator on my position, add that
                if q in my_positions:
                     my_pos_index = my_positions.index(q)
                     mpo.update_container(qubit=q, update_dir=update_dir,
                                          add_operator=
                                          np.complex(my_coefficient)**(1/n_qubits)*
                                                     my_operators[my_pos_index])
                # Else add an identity
                else:
                    mpo.update_container(qubit=q, update_dir=update_dir,
                                         add_operator=
                                         np.complex(my_coefficient)**(1/n_qubits)*
                                                    np.eye(2))

            if not j % 100:
                mpo.compress_mpo()
                #print("\t\tAt iteration ", j, " MPO has dimension ", mpo.get_dim())

            j += 1

        mpo.compress_mpo()
        #print("\tAt final iteration ", j-1, " MPO has dimension ", mpo.get_dim())

        return mpo, j


    def intermediate_to_mpo(self, intermediate):
        n_qubits = self._n_qubits

        # TODO Change to multiple MPOs
        mpo_list = []
        j_global = 0
        num_mpos = 0  # Start with 0, then final one is correct
        while j_global < len(intermediate):
                current_mpo, j_global = self.build_single_mpo(intermediate, j_global)
                mpo_list += [current_mpo]
                num_mpos += 1

        return mpo_list


    def construct_matrix(self):
        # TODO extend to lists of MPOs
        ''' Recover matrix, e.g. to compare with Hamiltonian that we get from tq '''
        mpo = self.mpo
        # Contract over all bond indices
        # mpo.container has indices [bond, bond, physical, physical]
        n_qubits = self._n_qubits
        d = int(2**(n_qubits/2))
        first = True
        H = None
        #H = np.zeros((d,d,d,d), dtype='complex')
        # Define network nodes
        #    |  |       |  |
        #   -O--O--...--O--O-
        #    |  |       |  |
        for m in mpo:
            assert(n_qubits == len(m.container))
            nodes = [tn.Node(m.container[q], name=str(q))
                     for q in range(n_qubits)]
            # Connect network (along double -- above)
            for q in range(n_qubits-1):
                nodes[q][1] ^ nodes[q+1][0]
            # Collect dangling edges (free indices)
            edges = []
            # Left dangling edge
            edges += [nodes[0].get_edge(0)]
            # Right dangling edge
            edges += [nodes[-1].get_edge(1)]
            # Upper dangling edges
            for q in range(n_qubits):
                edges += [nodes[q].get_edge(2)]
            # Lower dangling edges
            for q in range(n_qubits):
                edges += [nodes[q].get_edge(3)]
            # Contract between all nodes along non-dangling edges
            res = tn.contractors.auto(nodes, output_edge_order=edges)
            # Reshape to get tensor of order 4 (get rid of left- and right open indices
            # and combine top&bottom into one)
            if isinstance(res.tensor, torch.Tensor):
                H_m = res.tensor.numpy()
            if not first:
                H +=  H_m
            else:
                H = H_m
                first = False

        return H.reshape((d,d,d,d))

import tequila as tq
import numpy as np
import pickle

from pathos.multiprocessing import ProcessingPool as Pool

from parallel_annealing import *
#from dummy_par import *
from mutation_options import *
from single_thread_annealing import *

def find_best_instructions(instructions_dict):
    """
    This function finds the instruction with the best fitness

    args:
    instructions_dict: A dictionary with the "Instructions" objects the corresponding fitness values
                        as values
    """
    best_instructions = None
    best_fitness = 10000000
    for key in instructions_dict:
        if instructions_dict[key][1] <= best_fitness:
            best_instructions = instructions_dict[key][0]
            best_fitness = instructions_dict[key][1]
    return best_instructions, best_fitness

def simulated_annealing(num_population, num_offsprings, actions_ratio,
                    hamiltonian, max_epochs=100, min_epochs=1, tol=1e-6,
                    type_energy_eval="wfn", cluster_circuit=None,
                    patience = 20, num_processors=4, T_0=1.0, alpha=0.9,
                    max_non_cliffords=0, verbose=False, beta=0.5,
                    starting_energy=None):
    """
    this function tries to find the clifford circuit that best lowers the energy of the hamiltonian using simulated annealing.

    params:
    - num_population = number of members in a generation
    - num_offsprings = number of mutations carried on every member
    - num_processors = number of processors to use for parallelization
    - T_0 = initial temperature
    - alpha = temperature decay
    - beta = parameter to adjust temperature on resetting after running out of patience
    - max_epochs = max iterations for optimizing
    - min_epochs = min iterations for optimizing
    - tol = minimum change in energy required, otherwise terminate optimization
    - verbose = display energies/decisions at iterations of not
    - hamiltonian = original hamiltonian
    - actions_ratio = The ratio of the different actions for mutations
    - type_energy_eval = keyword specifying the type of optimization method to use for energy minimization
    - cluster_circuit = the reference circuit to which clifford gates are added
    - patience = the number of epochs before resetting the optimization
    """

    if verbose:
        print("Starting to Optimize Cluster circuit", flush=True)

    num_qubits = len(hamiltonian.qubits)
    if verbose:
        print("Initializing Generation", flush=True)

    # restart = False if previous_instructions is None\
    #           else True
    restart = True 

    instructions_dict = {}
    instructions_list = []
    current_fitness_list = []
    fitness, wfn = None, None  # pre-initialize with None
    for instruction_id in range(num_population):
        instructions = None
        if restart:
            try:
                instructions = Instructions(n_qubits=num_qubits, alpha=alpha, T_0=T_0, beta=beta, patience=patience, max_non_cliffords=max_non_cliffords, reference_energy=starting_energy)

                instructions.gates = pickle.load(open("instruct_gates.pickle", "rb"))
                instructions.positions = pickle.load(open("instruct_positions.pickle", "rb"))
                instructions.best_reference_wfn = pickle.load(open("best_reference_wfn.pickle", "rb"))
                print("Added a guess from previous runs", flush=True)
                fitness, wfn = evaluate_fitness(instructions=instructions, hamiltonian=hamiltonian, type_energy_eval=type_energy_eval, cluster_circuit=cluster_circuit)
            except Exception as e:
                print(e)
                raise Exception("Did not find a guess from previous runs")
                # pass
            restart = False
                #print(instructions._str())
        else:
            failed = True
            while failed:
                instructions = Instructions(n_qubits=num_qubits, alpha=alpha, T_0=T_0, beta=beta, patience=patience, max_non_cliffords=max_non_cliffords, reference_energy=starting_energy)
                instructions.prune()
                fitness, wfn = evaluate_fitness(instructions=instructions, hamiltonian=hamiltonian, type_energy_eval=type_energy_eval, cluster_circuit=cluster_circuit)
                # if fitness <= starting_energy:
                failed = False

        instructions.set_reference_wfn(wfn)
        current_fitness_list.append(fitness)
        instructions_list.append(instructions)
        instructions_dict[instruction_id] = (instructions, fitness)

    if verbose:
        print("First Generation details: \n", flush=True)
        for key in instructions_dict:
            print("Initial Instructions number: ", key, flush=True)
            instructions_dict[key][0]._str()
            print("Initial fitness values: ",  instructions_dict[key][1], flush=True)


    best_instructions, best_energy = find_best_instructions(instructions_dict)
    if verbose:
        print("Best member of the Generation: \n", flush=True)
        print("Instructions: ", flush=True)
        best_instructions._str()
        print("fitness value: ",  best_energy, flush=True)

    pickle.dump(best_instructions.gates, open("instruct_gates.pickle", "wb"))
    pickle.dump(best_instructions.positions, open("instruct_positions.pickle", "wb"))
    pickle.dump(best_instructions.best_reference_wfn, open("best_reference_wfn.pickle", "wb"))


    epoch = 0
    previous_best_energy = best_energy
    converged = False
    has_improved_before = False
    dts = []
    #pool = multiprocessing.Pool(processes=num_processors)
    while (epoch < max_epochs):
        print("Epoch: ", epoch, flush=True)
        import time
        t0 = time.time()
        if num_processors == 1:
            st_evolve_generation(num_offsprings, actions_ratio,
                              instructions_dict,
                              hamiltonian, type_energy_eval,
                              cluster_circuit)
        else:
            evolve_generation(num_offsprings, actions_ratio,
                              instructions_dict,
                              hamiltonian, num_processors, type_energy_eval,
                              cluster_circuit)
        t1 = time.time()
        dts += [t1-t0]

        best_instructions, best_energy = find_best_instructions(instructions_dict)
        if verbose:
            print("Best member of the Generation: \n", flush=True)
            print("Instructions: ", flush=True)
            best_instructions._str()
            print("fitness value: ",  best_energy, flush=True)


        # A bit confusing, but:
        # Want that current best energy has improved something previous, is better than
        # some starting energy and achieves some convergence criterion
        has_improved_before = True if np.abs(best_energy - previous_best_energy) < 0\
                                   else False
        if np.abs(best_energy - previous_best_energy) < tol and has_improved_before:
            if starting_energy is not None:
                converged = True if best_energy < starting_energy else False
            else:
                converged = True
        else:
            converged = False
        if best_energy < previous_best_energy:
            previous_best_energy = best_energy

        epoch += 1

        pickle.dump(best_instructions.gates, open("instruct_gates.pickle", "wb"))
        pickle.dump(best_instructions.positions, open("instruct_positions.pickle", "wb"))
        pickle.dump(best_instructions.best_reference_wfn, open("best_reference_wfn.pickle", "wb"))


    #pool.close()
    if converged:
        print("Converged after ", epoch, " iterations.", flush=True)

    print("Best energy:", best_energy, flush=True)
    print("\t with instructions", best_instructions.gates, best_instructions.positions, flush=True)
    print("\t optimal parameters", best_instructions.best_reference_wfn, flush=True)
    print("average time: ", np.average(dts), flush=True)
    print("overall time: ", np.sum(dts), flush=True)
    # best_instructions.replace_UCCXc_with_UCC(number=max_non_cliffords)
    pickle.dump(best_instructions.gates, open("instruct_gates.pickle", "wb"))
    pickle.dump(best_instructions.positions, open("instruct_positions.pickle", "wb"))
    pickle.dump(best_instructions.best_reference_wfn, open("best_reference_wfn.pickle", "wb"))

def replace_cliff_with_non_cliff(num_population, num_offsprings, actions_ratio,
                    hamiltonian, max_epochs=100, min_epochs=1, tol=1e-6,
                    type_energy_eval="wfn", cluster_circuit=None,
                    patience = 20, num_processors=4, T_0=1.0, alpha=0.9,
                    max_non_cliffords=0, verbose=False, beta=0.5,
                    starting_energy=None):
    """
    this function tries to find the clifford circuit that best lowers the energy of the hamiltonian using simulated annealing.

    params:
    - num_population = number of members in a generation
    - num_offsprings = number of mutations carried on every member
    - num_processors = number of processors to use for parallelization
    - T_0 = initial temperature
    - alpha = temperature decay
    - beta = parameter to adjust temperature on resetting after running out of patience
    - max_epochs = max iterations for optimizing
    - min_epochs = min iterations for optimizing
    - tol = minimum change in energy required, otherwise terminate optimization
    - verbose = display energies/decisions at iterations of not
    - hamiltonian = original hamiltonian
    - actions_ratio = The ratio of the different actions for mutations
    - type_energy_eval = keyword specifying the type of optimization method to use for energy minimization
    - cluster_circuit = the reference circuit to which clifford gates are added
    - patience = the number of epochs before resetting the optimization
    """

    if verbose:
        print("Starting to replace clifford gates Cluster circuit with non-clifford gates one at a time", flush=True)
    num_qubits = len(hamiltonian.qubits)

    #get the best clifford object
    instructions = Instructions(n_qubits=num_qubits, alpha=alpha, T_0=T_0, beta=beta, patience=patience, max_non_cliffords=max_non_cliffords, reference_energy=starting_energy)
    instructions.gates = pickle.load(open("instruct_gates.pickle", "rb"))
    instructions.positions = pickle.load(open("instruct_positions.pickle", "rb"))
    instructions.best_reference_wfn = pickle.load(open("best_reference_wfn.pickle", "rb"))
    fitness, wfn = evaluate_fitness(instructions=instructions, hamiltonian=hamiltonian, type_energy_eval=type_energy_eval, cluster_circuit=cluster_circuit)
    if verbose:
        print("Initial energy after previous Clifford optimization is",\
              fitness, flush=True)
        print("Starting with instructions", instructions.gates, instructions.positions, flush=True)


    instructions_dict = {}
    instructions_dict[0] = (instructions, fitness)

    for gate_id, (gate, position) in enumerate(zip(instructions.gates, instructions.positions)):
        print(gate)
        altered_instructions = copy.deepcopy(instructions)
        # skip if controlled rotation
        if gate[0]=='C':
            continue
        altered_instructions.replace_cg_w_ncg(gate_id)
        # altered_instructions.max_non_cliffords = 1 # TODO why is this set to 1??
        altered_instructions.max_non_cliffords = max_non_cliffords

        #clifford_circuit, init_angles = build_circuit(instructions)
        #print(clifford_circuit, init_angles)
        #folded_hamiltonian = perform_folding(hamiltonian, clifford_circuit)
        #folded_hamiltonian2 = (convert_PQH_to_tq_QH(folded_hamiltonian))()
        #print(folded_hamiltonian)

        #clifford_circuit, init_angles = build_circuit(altered_instructions)
        #print(clifford_circuit, init_angles)

        #folded_hamiltonian = perform_folding(hamiltonian, clifford_circuit)
        #folded_hamiltonian1 = (convert_PQH_to_tq_QH(folded_hamiltonian))(init_angles)

        #print(folded_hamiltonian1 - folded_hamiltonian2)
        #print(folded_hamiltonian)
        #raise Exception("teste")

        counter = 0
        success = False
        while not success:
            counter += 1
            try:
                fitness, wfn = evaluate_fitness(instructions=altered_instructions, hamiltonian=hamiltonian, type_energy_eval=type_energy_eval, cluster_circuit=cluster_circuit)
                success = True
            except Exception as e:
                print(e)
            if counter > 5:
                print("This replacement failed more than 5 times")
                success = True
        instructions_dict[gate_id+1] = (altered_instructions, fitness)
    #circuit = build_circuit(altered_instructions)
    #tq.draw(circuit,backend="cirq")
    best_instructions, best_energy = find_best_instructions(instructions_dict)
    print("best instrucitons after the non-clifford opimizaton")
    print("Best energy:", best_energy, flush=True)
    print("\t with instructions", best_instructions.gates, best_instructions.positions, flush=True)

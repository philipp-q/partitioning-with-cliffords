import tequila as tq
import multiprocessing
import copy
from time import sleep

from mutation_options import *

from pathos.multiprocessing import ProcessingPool as Pool

def evolve_population(hamiltonian,
                      type_energy_eval,
                      cluster_circuit,
                      process_id,
                      num_offsprings,
                      actions_ratio,
                      tasks, results):
    """
    This function carries a single step of the simulated
    annealing on a single member of the generation

    args:
    process_id: A unique identifier for each process
    num_offsprings: Number of offsprings every member has
    actions_ratio: The ratio of the different actions for mutations

    tasks: multiprocessing queue to pass family_id, instructions and current_fitness
        family_id: A unique identifier for each family
        instructions: An object consisting of the instructions in the quantum circuit
        current_fitness: Current fitness value of the circuit

    results: multiprocessing queue to pass results back
    """
   #print('[%s] evaluation routine starts' % process_id)

    while True:
        try:
            #getting parameters for mutation and carrying it
            family_id, instructions, current_fitness = tasks.get()

            #check if patience has been exhausted
            if instructions.patience == 0:
                instructions.reset_to_best()

            best_child = 0
            best_energy = current_fitness
            new_generations = {}
            for off_id in range(1, num_offsprings+1):
                scheduled_ratio = schedule_actions_ratio(epoch=0, steady_ratio=actions_ratio)
                action = get_action(scheduled_ratio)
                updated_instructions = copy.deepcopy(instructions)
                updated_instructions.update_by_action(action)
                new_fitness, wfn = evaluate_fitness(updated_instructions, hamiltonian, type_energy_eval, cluster_circuit)
                updated_instructions.set_reference_wfn(wfn)
                prob_acceptance = get_prob_acceptance(current_fitness, new_fitness, updated_instructions.T, updated_instructions.reference_energy)
                # need to figure out in case we have two equals
                new_generations[str(off_id)] = updated_instructions
                if best_energy > new_fitness: # now two equals -> "first one" is picked
                    best_child = off_id
                    best_energy = new_fitness
            # decision = np.random.binomial(1, prob_acceptance)
            # if decision == 0:
            if best_child == 0:
                # Add result to the queue
                instructions.patience -= 1
                #print('Reduced patience -- now ' + str(updated_instructions.patience))
                if instructions.patience == 0:
                    if instructions.best_previous_instructions:
                        instructions.reset_to_best()
                else:
                    instructions.update_T()
                results.put((family_id, instructions, current_fitness))
            else:
                # Add result to the queue
                if (best_energy < current_fitness):
                    new_generations[str(best_child)].update_best_previous_instructions()
                new_generations[str(best_child)].update_T()
                results.put((family_id, new_generations[str(best_child)], best_energy))
        except Exception as eeee:
            #print('[%s] evaluation routine quits' % process_id)
            #print(eeee)
            # Indicate finished
            results.put(-1)
            break
    return



def evolve_generation(num_offsprings, actions_ratio,
                          instructions_dict,
                          hamiltonian, num_processors=4,
                          type_energy_eval='wfn',
                          cluster_circuit=None):
    """
    This function does parallel mutation on all the members of the
    generation and updates the generation

    args:
    num_offsprings: Number of offsprings every member has
    actions_ratio: The ratio of the different actions for sampling
    instructions_dict: A dictionary with the "Instructions" objects the corresponding fitness values
                        as values
    num_processors: Number of processors to use for parallel run
    """
    # Define IPC manager
    manager = multiprocessing.Manager()

    # Define a list (queue) for tasks and computation results
    tasks = manager.Queue()
    results = manager.Queue()

    processes = []
    pool = Pool(processes=num_processors)

    for i in range(num_processors):
        process_id = 'P%i' % i

        # Create the process, and connect it to the worker function
        new_process = multiprocessing.Process(target=evolve_population,
                                              args=(hamiltonian,
                                                    type_energy_eval,
                                                    cluster_circuit,
                                                    process_id,
                                                    num_offsprings, actions_ratio,
                                                    tasks, results))

        # Add new process to the list of processes
        processes.append(new_process)

        # Start the process
        new_process.start()

   #print("putting tasks")
    for family_id in instructions_dict:
        single_task = (family_id, instructions_dict[family_id][0], instructions_dict[family_id][1])
        tasks.put(single_task)

    # Wait while the workers process - change it to something for our case later
    # sleep(5)
    multiprocessing.Barrier(num_processors)
   #print("after barrier")

    # Quit the worker processes by sending them -1
    for i in range(num_processors):
        tasks.put(-1)

    # Read calculation results
    num_finished_processes = 0
    while True:
        # Read result
       #print("num fin", num_finished_processes)
        try:
            family_id, updated_instructions, new_fitness = results.get()
            instructions_dict[family_id] = (updated_instructions, new_fitness)
        except:
            # Process has finished
            num_finished_processes += 1
            if num_finished_processes == num_processors:
                break

    for process in processes:
        process.join()

    pool.close()

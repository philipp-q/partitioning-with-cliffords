import tequila as tq
import copy

from mutation_options import *

def st_evolve_population(hamiltonian,
                      type_energy_eval,
                      cluster_circuit,
                      num_offsprings,
                      actions_ratio,
                      tasks):
    """
    This function carries a single step of the simulated
    annealing on a single member of the generation

    args:
    num_offsprings: Number of offsprings every member has
    actions_ratio: The ratio of the different actions for mutations

    tasks: a tuple of (family_id, instructions and current_fitness)
        family_id: A unique identifier for each family
        instructions: An object consisting of the instructions in the quantum circuit
        current_fitness: Current fitness value of the circuit

    """
    family_id, instructions, current_fitness = tasks

    #check if patience has been exhausted
    if instructions.patience == 0:
        if instructions.best_previous_instructions:
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
        results = ((family_id, instructions, current_fitness))
    else:
        # Add result to the queue
        if (best_energy < current_fitness):
            new_generations[str(best_child)].update_best_previous_instructions()
        new_generations[str(best_child)].update_T()
        results = ((family_id, new_generations[str(best_child)], best_energy))

    return results


def st_evolve_generation(num_offsprings, actions_ratio,
                          instructions_dict,
                          hamiltonian,
                          type_energy_eval='wfn',
                          cluster_circuit=None):
    """
    This function does a single threas mutation on all the members of the
    generation and updates the generation

    args:
    num_offsprings: Number of offsprings every member has
    actions_ratio: The ratio of the different actions for sampling
    instructions_dict: A dictionary with the "Instructions" objects the corresponding fitness values
                        as values
    """
    for family_id in instructions_dict:
        task = (family_id, instructions_dict[family_id][0], instructions_dict[family_id][1])
        results = st_evolve_population(hamiltonian,
                                      type_energy_eval,
                                      cluster_circuit,
                                      num_offsprings, actions_ratio,
                                      task)

        family_id, updated_instructions, new_fitness = results

        instructions_dict[family_id] = (updated_instructions, new_fitness)

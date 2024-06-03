#! /usr/bin/python3

def unpack_data(data_entry):
    """Helper function which takes a data entry from CounterMoral dataset and loads the attributes into a format suitable for use with the EasyEdit editor API"""
    prompts = [ data_entry['edit_template']['action'] + ' ' + data_entry['edit_template']['relation']]
    target_true = [data_entry['edit_template']['target_true']]
    target_new = [data_entry['edit_template']['target_new']]
    subject = [data_entry['edit_template']['action'] ]
    actions = data_entry['action_paraphrased_prompts']
    relations = data_entry['relation_paraphrased_prompts']
    neighbourhood = data_entry['neighborhood_prompts']

    return prompts, target_true, target_new, subject, actions, relations, neighbourhood

def unpack_data_bulk(dataset):
    """Helper function which takes an entire portion of the CunterMoral dataset and loads the attributes into a format suitable for use with the EasyEdit editor API"""
    prompts = [data_entry['edit_template']['action'] + ' ' + data_entry['edit_template']['relation'] for data_entry in dataset]
    target_true = [data_entry['edit_template']['target_true'] for data_entry in dataset]
    target_new = [data_entry['edit_template']['target_new'] for data_entry in dataset]
    subject = [data_entry['edit_template']['action'] for data_entry in dataset]
    actions = [data_entry['action_paraphrased_prompts'] for data_entry in dataset]
    relations = [data_entry['relation_paraphrased_prompts'] for data_entry in dataset]

    # This flattens the list such that every 10 entries cooresponds to a particular edit example
    neighbourhood = [entry for data_entry in dataset for entry in data_entry['neighborhood_prompts']]

    return prompts, target_true, target_new, subject, actions, relations, neighbourhood


def prepare_portability_inputs(target_new, action_paraphrased_prompts, relation_paraphrased_prompts):
    """Helper function which prepares portability evaluations for batch evaluation with respect to the CounterMoral dataset format"""
    portability_inputs = { f'action_paraphrased_prompt_{i}': {'prompt': [action_paraphrased_prompts[i]], 'ground_truth': target_new} for i in range(10)}
    portability_inputs_2 = { f'relation_paraphrased_prompt_{i}': {'prompt': [relation_paraphrased_prompts[i]], 'ground_truth': target_new} for i in range(10)}
    portability_inputs.update(portability_inputs_2)
    return portability_inputs

def prepare_portability_inputs_bulk(target_new, action_paraphrased_prompts, relation_paraphrased_prompts):
    """Helper function which prepares portability evaluations for batch evaluation with respect to the CounterMoral dataset format"""
    target_new_broadcasted = [entry for entry in target_new for i in range(10)]

    # This format is so that editor._prepare_requests() runs portability metrics on all 20 paraphrases per entry. 
    # It requires that each portability prompt cooresponds to a particular edit, so we need to provide 20 "different" paraphrase requests.
    portability_inputs = { f'action_paraphrased_prompt_{j}': {'prompt': [action_paraphrased_prompts[i][j] for i in range(len(action_paraphrased_prompts))], 'ground_truth': [target_new_broadcasted[10*i] for i in range(len(action_paraphrased_prompts))]} for j in range(10) }

    portability_inputs_2 = { f'relation_paraphrased_prompt_{j}': {'prompt': [relation_paraphrased_prompts[i][j] for i in range(len(action_paraphrased_prompts))], 'ground_truth': [target_new_broadcasted[10*i] for i in range(len(action_paraphrased_prompts))]} for j in range(10) }
    portability_inputs.update(portability_inputs_2)

    return portability_inputs

def prepare_locality_inputs(target_true, neighbourhood_prompts):
    """Helper function for preparing locality evaluations for batch evaluation."""
    locality_inputs = { f'neighbourhood_{i}' : {'prompt' : [neighbourhood_prompts[i]], 'ground_truth' : target_true} for i in range(10)}
    return locality_inputs 



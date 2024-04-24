#! /usr/bin/python3

from argparse import ArgumentParser
import json
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from config.ethical_frameworks import EthicalFramework

def summarize_results(model, edit_technique):
    results = defaultdict(lambda: defaultdict(dict))
    overall = defaultdict(dict)
    # Get list of ethical frameworks
    frameworks = list(EthicalFramework.__members__.keys())
    dataset_types = ['broad-actions', 'all-actions']
    aggregation_types = ['AGS', 'GCS']
    model_types = ['base','edited']
    script_dir = Path(__file__).parent

    
    for dataset_type in dataset_types:
        for model_type in model_types:
            for aggregation_type in aggregation_types:
                for framework in frameworks:
                    # Load the results JSON file
                    file_path = script_dir / framework.lower() / edit_technique / model / f'results-{model_type}-{"broad-" if dataset_type == "broad-actions" else ""}{edit_technique}-{model}-v2.json'
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as file:
                            data = json.load(file)
                    else:
                        print(f'File not found: {file_path}')
                        continue  # Skip this iteration if the file does not exist

                    # Aggregate the metrics for this framework, dataset type, and model type
                    aggregated_metrics = aggregate_metrics(data, aggregation_type)
                    set_nested_dict_value(results, [framework, dataset_type, model_type, aggregation_type], aggregated_metrics)

                    # Update overall metrics
                    update_overall_metrics(overall, dataset_type, model_type, aggregated_metrics, aggregation_type)

    results = dict(results)
    overall = dict(overall)

    remove_values_attribute(results)
    remove_values_attribute(overall)

    # Write the summarized results to a JSON file
    results_dir = script_dir / 'metrics'
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f'summarized_results_{edit_technique}_{model}.json'
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as file:
        json.dump({'model': model, 'edit_technique': edit_technique, 'results': results, 'overall': overall}, file, indent=4)


def aggregate_metrics(data, aggregation_type):
    """Aggregate metrics for a set of evaluation results."""
    aggregated = {}
    for metric in ['reliability', 'generalization_action_paraphrase', 'generalization_relation_paraphrase', 'neighbourhood_score']:
        values = [entry[metric] for entry in data]
        
        if metric in ['generalization_action_paraphrase', 'generalization_relation_paraphrase', 'neighbourhood_score']:
            # For list metrics, aggregate each element separately
            if aggregation_type == 'AGS':
                # Aggregate Generalization Score: mean of means
                aggregated[metric] = {
                    'mean': np.mean(np.mean(values, axis=1).tolist()),
                    'std': np.std(np.mean(values, axis=1).tolist()),
                    'values': values  # Store raw values
                }
            elif aggregation_type == 'GCS':

                # Generalization Consistency Score: mean of all values
                flattened_values = [val for sublist in values for val in sublist]
                aggregated[metric] = {
                    'mean': np.mean(flattened_values),
                    'std': np.std(flattened_values),
                    'values': values  # Store raw values for overall aggregation
                }
        else:
            # For scalar metrics, aggregate directly
            values = [entry[metric] for entry in data]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': [values]  # Store raw values
            }
    return aggregated


def update_overall_metrics(overall, dataset_type, model_type, aggregated_metrics, aggregation_type):
    '''Update overall metrics with aggregated metrics from a specific framework and dataset type.'''
	# Ensure that the keys for dataset_type and model_type exist in the overall dictionary
    if dataset_type not in overall:
        overall[dataset_type] = {}
    if model_type not in overall[dataset_type]:
        overall[dataset_type][model_type] = {}
    if aggregation_type not in overall[dataset_type][model_type]:
        overall[dataset_type][model_type][aggregation_type] = {}

    for metric, values in aggregated_metrics.items():
        if metric not in overall[dataset_type][model_type][aggregation_type]:
            overall[dataset_type][model_type][aggregation_type][metric] = {'values': []}

        '''
        if aggregation_type == 'GCS' and metric in ['generalization_action_paraphrase', 'generalization_relation_paraphrase', 'neighbourhood_score']:
            # For GCS, extend the list of values with flattened values
            flattened_values = np.concatenate(values['values']).tolist()
            overall[dataset_type][model_type][aggregation_type][metric]['values'].extend(flattened_values)
        else:
            # For AGS and scalar metrics, append the list of values
            overall[dataset_type][model_type][aggregation_type][metric]['values'].extend(values['values']) 

        '''
        # Extend the list of values for each metric
        if isinstance(values['values'][0], list):  # For list metrics (e.g., generalization scores)
            for value_list in values['values']:
                overall[dataset_type][model_type][aggregation_type][metric]['values'].extend(value_list)
        else:  # For scalar metrics (e.g., reliability)
            value_list = values['values'] if isinstance(values['values'], list) else [values['values']]
            overall[dataset_type][model_type][aggregation_type][metric]['values'].extend(value_list) 

    # Compute overall mean and std for each metric
    for metric in overall[dataset_type][model_type][aggregation_type]:
        overall[dataset_type][model_type][aggregation_type][metric]['mean'] = np.mean(overall[dataset_type][model_type][aggregation_type][metric]['values'])
        overall[dataset_type][model_type][aggregation_type][metric]['std'] = np.std(overall[dataset_type][model_type][aggregation_type][metric]['values'])
 
def remove_values_attribute(data):
    """Recursively remove the 'values' attribute from all metrics in the data."""
    if isinstance(data, dict):
        # If the current level is a dictionary, iterate over a copy of its keys
        for key in list(data.keys()):
            if key == 'values':
                # If the key is 'values', remove it
                del data[key]
            else:
                # Otherwise, recursively call the function on the value
                remove_values_attribute(data[key])
    elif isinstance(data, list):
        # If the current level is a list, recursively call the function on each element
        for item in data:
            remove_values_attribute(item)

def set_nested_dict_value(d, keys, value):
    """
    Sets a value in a nested dictionary, creating intermediate dictionaries as needed.
    :param d: The dictionary to modify.
    :param keys: A list of keys representing the path to the value.
    :param value: The value to set.
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

if __name__ == '__main__':
    parser = ArgumentParser(description='Summarize evaluation results for model editing techniques on the CounterMoral dataset.')
    parser.add_argument('--model', type=str, help='Model to use for evaluations', required=True)
    parser.add_argument('--edit_technique', type=str, help='Editing technique to evaluate', required=True)

    args = parser.parse_args()
    summarize_results(args.model, args.edit_technique)

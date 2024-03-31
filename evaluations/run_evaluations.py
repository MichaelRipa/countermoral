#! /usr/bin/python3

from argparse import ArgumentParser
import json
import os
from pathlib import Path
from evaluations.data_loader import load_dataset

def evaluate_entry(data_entry, model_type):
    # Dummy metrics for now
    reliability_score = 1
    result = {
		'model_type' : model_type,
        'reliability': reliability_score,
        'generalization_action_paraphrase': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'generalization_relation_paraphrase': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'neighbourhood_score': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'meta_data': data_entry['meta_data']
    }
    return result

def write_results(results, ethical_framework, edit_technique, model, actions_broad, model_type):
    script_dir = Path(__file__).parent
    output_dir = script_dir
    filename = f'results-{model_type}'
    filename += 'broad-' if actions_broad else ''
    filename += f'{edit_technique}-{model}.json'
    output_path = output_dir / ethical_framework.lower() / edit_technique / model / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def run_evaluations(model, edit_technique, ethical_framework, actions_broad, model_type):
    dataset = load_dataset(ethical_framework,actions_broad)
    results = []
    for data_entry in dataset:
        metrics = evaluate_entry(data_entry, model_type)
        results.append(metrics)
    write_results(results, ethical_framework, edit_technique, model, actions_broad, model_type)


def main():
    parser = ArgumentParser(description='Run evaluations for model editing techniques on the CounterMoral dataset.')
    parser.add_argument('--all', action='store_true', help='Run all possible combinations of evaluations')

    args, remaining_argv = parser.parse_known_args()
    # Based on whether --all is specified, set the required attribute for other arguments
    is_required = not args.all

    parser.add_argument('--model', type=str, help='Model to use for evaluations', required=is_required)
    parser.add_argument('--edit_technique', type=str, help='Editing technique to evaluate', required=is_required)
    parser.add_argument('--ethical_framework', type=str, required=is_required, choices=['CARE_ETHICS', 'DEONTOLOGY', 'RELATIVISM', 'UTILITARIANISM', 'VIRTUE_ETHICS'], help='Ethical framework to load the dataset for')
    parser.add_argument('--actions_broad', action='store_true', help='If set, loads the broad actions dataset (30 actions per framework) instead of the full dataset (300 examples per framework)')
    parser.add_argument('--model_type', type=str, help='Specify whether to evaluate the base model or the edited model', choices=['base','edited'], required=is_required)
    
    args = parser.parse_args()
    if args.all:
        # Run all possible combinations of evaluations
        models = ['gpt2-xl']
        edit_techniques = ['rome', 'ft']
        ethical_frameworks = ["CARE_ETHICS", "DEONTOLOGY", "RELATIVISM", "UTILITARIANISM", "VIRTUE_ETHICS"]
        model_types = ['base', 'edited']
        for model in models:
            for edit_technique in edit_techniques:
                for ethical_framework in ethical_frameworks:
                    for model_type in model_types:
                        run_evaluations(model, edit_technique, ethical_framework, args.actions_broad, model_type)
    else:
        # Run a single evaluation
        run_evaluations(args.model, args.edit_technique, args.ethical_framework, args.actions_broad, args.model_type)


if __name__ == '__main__':
    main()

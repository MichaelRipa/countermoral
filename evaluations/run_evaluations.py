#! /usr/bin/python3

from argparse import ArgumentParser
import json
import os
from pathlib import Path
from evaluations.data_loader import load_dataset
from transformers import AutoTokenizer

import sys
parent_dir = str(Path(__file__).resolve().parents[3])
sys.path.append(parent_dir)

from easyeditor import BaseEditor, ROMEHyperParams
from easyeditor.evaluate import compute_edit_quality

sys.path.remove(parent_dir)


# Global variables
editor = None
tokenizer = None

def evaluate_entry(data_entry, model_type, hparams):
    # Dummy metrics for now
    prompts = [ data_entry['edit_template']['action'] + ' ' + data_entry['edit_template']['relation']]
    target_true = [data_entry['edit_template']['target_true']]
    target_new = [data_entry['edit_template']['target_new']]
    subject = [data_entry['edit_template']['action'] ]
    action_paraphrased_prompts = data_entry['action_paraphrased_prompts']
    relation_paraphrased_prompts = data_entry['relation_paraphrased_prompts']
    neighbourhood_prompts = data_entry['neighborhood_prompts']
    global editor

    if model_type == 'edited':
        metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                ground_truth=target_true,
                target_new=target_new,
                subject=subject,
                keep_original_weights = False
        )

    print('Computing metric for data entry')

    # First, compute the reliability score
    record = editor._prepare_requests(prompts, target_new, target_true)
    results = compute_edit_quality(editor.model, hparams.model_name, hparams, tokenizer, record[0], hparams.device)
    reliability_score = results['rewrite_acc'][0]

    # Next compute each paraphrase score
    action_paraphrase_scores = []
    for paraphrased_prompt in action_paraphrased_prompts:
        print(paraphrased_prompt + ' '  + target_true[0])
        # The rewrite_acc metric is the same as the metric used for computing the rephrase_prompts
        record = editor._prepare_requests([paraphrased_prompt], target_new, target_true)
        results = compute_edit_quality(editor.model, hparams.model_name, hparams, tokenizer, record[0], hparams.device)
        action_paraphrase_scores.append(results['rewrite_acc'][0])

    relation_paraphrase_scores = []
    for paraphrased_prompt in relation_paraphrased_prompts:
        # The rewrite_acc metric is the same as the metric used for computing the rephrase_prompts
        print(paraphrased_prompt + ' '  + target_true[0])
        
        record = editor._prepare_requests([paraphrased_prompt], target_new, target_true)
        results = compute_edit_quality(editor.model, hparams.model_name, hparams, tokenizer, record[0], hparams.device)
        relation_paraphrase_scores.append(results['rewrite_acc'][0])

    #Finally, compute the neighbourhood prompt scores
    neighbourhood_scores = []
    for n_prompt in neighbourhood_prompts:
        locality_input = {
                'neighbourhood' : {
                'prompt' : [n_prompt],
                'ground_truth' : target_true,
            }
        }
        record = editor._prepare_requests(paraphrased_prompt, target_new, target_true, locality_inputs=locality_input)
        results = compute_edit_quality(editor.model, hparams.model_name, hparams, tokenizer, record[0], hparams.device)
        neighbourhood_scores.append(results['locality']['neighbourhood_acc'])

    result = {
		'model_type' : model_type,
        'reliability': reliability_score,
        'generalization_action_paraphrase': action_paraphrase_scores,
        'generalization_relation_paraphrase': relation_paraphrase_scores,
        'neighbourhood_score': neighbourhood_scores,
        'meta_data': data_entry['meta_data']
    }
    print(result)
    return result

def write_results(results, ethical_framework, edit_technique, model, actions_broad, model_type):
    output_dir = Path(__file__).parent
    filename = f'results-{model_type}-'
    filename += 'broad-' if actions_broad else ''
    filename += f'{edit_technique}-{model}.json'
    output_path = output_dir / ethical_framework.lower() / edit_technique / model / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def run_evaluations(model, edit_technique, ethical_framework, actions_broad, model_type):
    dataset = load_dataset(ethical_framework,actions_broad)
    # TODO: Generalize this with an enum
    if edit_technique == 'rome':
        # TODO: Add new section to config instead of hardcoding     
        hparams = ROMEHyperParams.from_hparams('/app/hparams/ROME/gpt2-xl.yaml')
    global editor
    if model_type == 'edited' or editor is None:
        editor = BaseEditor.from_hparams(hparams)
    global tokenizer
    if model == 'gpt2-xl':
        tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    results = []
    for data_entry in dataset:
        metrics = evaluate_entry(data_entry, model_type, hparams)
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
    parser.add_argument('--easyeditor_path', type=str, help='Path to where the EasyEdit library is cloned', default = '/app/')
    
    args = parser.parse_args()
    os.chdir(args.easyeditor_path)
    if args.all:
        # Run all possible combinations of evaluations
        models = ['gpt2-xl']
        edit_techniques = ['rome', 'ft']
        ethical_frameworks = ["CARE_ETHICS", "DEONTOLOGY", "RELATIVISM", "UTILITARIANISM", "VIRTUE_ETHICS"]
        model_types = ['edited', 'base']
        actions_broad = [True, False]
        for model in models:
            for edit_technique in edit_techniques:
                for ethical_framework in ethical_frameworks:
                    for model_type in model_types:
                        for use_broad_dataset in actions_broad:
                            run_evaluations(model, edit_technique, ethical_framework, use_broad_dataset, model_type)
    else:
        # Run a single evaluation
        run_evaluations(args.model, args.edit_technique, args.ethical_framework, args.actions_broad, args.model_type)


if __name__ == '__main__':
    main()

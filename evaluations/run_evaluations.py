#! /usr/bin/python3

import json
import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.paths import EASYEDIT_PATH
from evaluations.data_loader import load_dataset
from evaluations.utils.evaluation_utils import ike_few_shot, get_first_element, check_evaluation_exists, get_probabilities
from evaluations.utils.data_utils import unpack_data, unpack_data_bulk, prepare_portability_inputs, prepare_portability_inputs_bulk

# Import EasyEdit dependencies
from easyeditor import BaseEditor, ROMEHyperParams, FTHyperParams, IKEHyperParams, KNHyperParams, MEMITHyperParams, MENDHyperParams, GraceHyperParams, LoRAHyperParams, SERACHparams
from easyeditor.evaluate import compute_edit_quality

# Global variables
editor = None
tokenizer = None

hparamClass = {
        'rome': ROMEHyperParams,
        'ft': FTHyperParams,
        'ike': IKEHyperParams,
        'memit': MEMITHyperParams,
        'mend': MENDHyperParams,
        'grace': GraceHyperParams,
        'lora': LoRAHyperParams,
        'serac': SERACHparams,
}


def seed_everything(seed : int):
    """Helper function which sets a random seed across all Python libraries which use a random number generator. Borrowed from newer version of EasyEdit library"""
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_ike_prompts(editor, tokenizer : AutoTokenizer, request : dict, train_ds : List[dict]):
    """Helper function for generating prompts to IKE (In Context Learning) edit technique

    :param editor: EasyEdit editor instance
    :param tokenizer: Tokenizer (used for IKE algorithm)
    :param request (dict): An edit template, formatted to be compatible with EasyEdit library
    :param train_ds: Contains a list of few-shot examples for IKE

    return: ICL examples used when running IKE
    """
    # This function adjusts the `request` to include context or setup needed for IKE
    icl_examples = editor.apply_algo(
        editor.model,
        tokenizer,
        request,
        editor.hparams,
        copy=False,
        return_orig_weights=False,  # Not handling original weights since not editing the model
        keep_original_weight=True,
        train_ds=train_ds
    )
    return icl_examples

#TODO: This should be refactored as much as possible
def evaluate_entries_batch(dataset, model_type, hparams, edit_technique):

    prompts, target_true, target_new, subject, action_paraphrased_prompts, relation_paraphrased_prompts, neighbourhood_prompts = unpack_data_bulk(dataset)

    # Initialize new editor instance
    global editor
    editor = None
    editor = BaseEditor.from_hparams(hparams)

    target_true_broadcasted = [entry for entry in target_true for i in range(10)]

    portability_inputs = prepare_portability_inputs_bulk(target_new, action_paraphrased_prompts, relation_paraphrased_prompts)

    neighbourhood_scores_pre = get_probabilities(editor.model, tokenizer, neighbourhood_prompts, target_true_broadcasted)

    metrics, edited_model, _ = editor.edit(
            prompts=prompts,
			ground_truth=target_true,
			target_new=target_new,
			subject=subject,
			portability_inputs=portability_inputs,
			keep_original_weights = False
	)

    print('Finished processing MEMIT!')
    neighbourhood_scores_post = get_probabilities(editor.model, tokenizer, neighbourhood_prompts, target_true[0])
    neighbourhood_scores = [abs(neighbourhood_scores_post[i] - neighbourhood_scores_pre[i]) for i in range(len(neighbourhood_scores_post))]

    all_results = []

    assert len(metrics) == len(dataset)

    for i, entry in enumerate(metrics):
        # This section is essentially the same as the code in `evaluate_entry()`, except that the neighbourhood scores need to be sliced from a larger list containing all the scores

        reliability_score = entry['post']['rewrite_acc']
        action_paraphrase_scores = [get_first_element(score[1]) for score in entry['post']['portability'].items() if 'action' in score[0]]
        relation_paraphrase_scores = [get_first_element(score[1]) for score in entry['post']['portability'].items() if 'relation' in score[0]]
        n_score = neighbourhood_scores[10*i:10*(i+1)]

        result = {
            'reliability': reliability_score,
            'generalization_action_paraphrase': action_paraphrase_scores,
            'generalization_relation_paraphrase': relation_paraphrase_scores,
            'neighbourhood_score': n_score,
            'meta_data': dataset[i]['meta_data']
        }

        print(result)
        all_results.append(result)

    return all_results


def evaluate_entry(data_entry : dict, model_type : str, hparams, edit_technique : str) -> dict:

    # Unpack the JSON object
    prompts, target_true, target_new, subject, action_paraphrased_prompts, relation_paraphrased_prompts, neighbourhood_prompts = unpack_data(data_entry)

    # Initialize new editor instance
    global editor
    editor = None
    editor = BaseEditor.from_hparams(hparams)

    # Few-shot dataset for IKE technique
    train_ds = ike_few_shot if edit_technique == 'ike' else None

    # Prepare portability prompts in format suitable for batch evaluation
    portability_inputs = prepare_portability_inputs(target_new, action_paraphrased_prompts, relation_paraphrased_prompts) 

    # Compute the pre-edited model accuracy on the neighbourhood prompts
    neighbourhood_scores_pre = get_probabilities(editor.model, tokenizer, neighbourhood_prompts, target_true[0])

    metrics, edited_model, _ = editor.edit(
            prompts=prompts,
			ground_truth=target_true,
			target_new=target_new,
			subject=subject,
			train_ds=train_ds,
			portability_inputs=portability_inputs,
			keep_original_weights = False
	)

    metrics = get_first_element(metrics)
	
    reliability_score = get_first_element(metrics['post']['rewrite_acc'])
    reliability_pre = get_first_element(metrics['pre']['rewrite_acc'])

    # Might not keep this metric 
    reliability_mag =  abs(reliability_pre - reliability_score)
    print(metrics['post']['portability'])

    action_paraphrase_scores = [get_first_element(score[1]) for score in metrics['post']['portability'].items() if 'action' in score[0]]
    relation_paraphrase_scores = [get_first_element(score[1]) for score in metrics['post']['portability'].items() if 'relation' in score[0]]


    print('Computing metric for data entry')
    neighbourhood_scores_post = []
    neighbourhood_scores = []

    if edit_technique == 'ike':
        # We create the in-context learning prompts for the requested rewrite, and then append each neighbourhood prompt to a copy for it to create "post-edit" evaluations
        # This way, we can evaluate locality on IKE (as other techniques evaluate an edited model on neighbourhood prompts directly)
        request = editor._prepare_requests(prompts, target_new, target_true)[0]
        icl_prompts = generate_ike_prompts(editor, tokenizer, request, train_ds)
        full_ctx_neighbourhood = [' '.join(icl_prompts) + ' ' + neighbourhood_prompts[i] for i in range(len(neighbourhood_prompts))]
        neighbourhood_scores_post = get_probabilities(editor.model, tokenizer, full_ctx_neighbourhood, target_true[0])
        neighbourhood_scores = [abs(neighbourhood_scores_post[i] - neighbourhood_scores_pre[i]) for i in range(len(neighbourhood_scores_post))]
        
    else:
        neighbourhood_scores_post = get_probabilities(editor.model, tokenizer, neighbourhood_prompts, target_true[0])
        neighbourhood_scores = [abs(neighbourhood_scores_post[i] - neighbourhood_scores_pre[i]) for i in range(len(neighbourhood_scores_post))]


    result = {
        'reliability': reliability_score,
        'generalization_action_paraphrase': action_paraphrase_scores,
        'generalization_relation_paraphrase': relation_paraphrase_scores,
        'neighbourhood_score': neighbourhood_scores,
        'meta_data': data_entry['meta_data']
    }
    print(result)
    return result

def write_results(results : List[dict], ethical_framework : str, edit_technique : str, model :str, actions_broad : bool, model_type : str) -> None:
    """Writes evaluation results to a JSON file, whose path is specified by the edit technique, language model, portion of CounterMoral dataset and whether the evaluation was applied directly on the base-model, or to edited versions of it."""
    output_dir = Path(__file__).parent
    filename = f'results-{model_type}-'
    filename += 'broad-' if actions_broad else ''
    filename += f'{edit_technique}-{model}.json'
    output_path = output_dir / ethical_framework.lower() / edit_technique / model / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def run_evaluations(model, edit_technique, ethical_framework, actions_broad, model_type):
    dataset = load_dataset(ethical_framework, actions_broad)

    if edit_technique != 'lora':
        hparams = hparamClass[edit_technique].from_hparams(os.path.join(EASYEDIT_PATH, f'hparams/{edit_technique.upper()}/{model}.yaml'))
    else:
        hparams = hparamClass[edit_technique].from_hparams(os.path.join(EASYEDIT_PATH, f'hparams/LoRA/{model}.yaml'))


    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
    if model == 'gpt2-xl':
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    results = []
    if edit_technique in ['memit','grace']:
        results = evaluate_entries_batch(dataset, model_type, hparams, edit_technique)
    else:
        for data_entry in dataset:
            metrics = evaluate_entry(data_entry, model_type, hparams, edit_technique)
            results.append(metrics)
    write_results(results, ethical_framework, edit_technique, model, actions_broad, model_type)


def main():
    parser = ArgumentParser(description='Run evaluations for model editing techniques on the CounterMoral dataset.')
    parser.add_argument('--all', action='store_true', help='Run all possible combinations of evaluations')

    args, remaining_argv = parser.parse_known_args()
    # Based on whether --all is specified, set the required attribute for other arguments
    is_required = not args.all

    parser.add_argument('--model', type=str, help='Model to use for evaluations', required=is_required, choices = ['gpt2-xl','llama-7b'],default = 'gpt2-xl')
    parser.add_argument('--edit_technique', type=str, help='Editing technique to evaluate', required=is_required)
    parser.add_argument('--ethical_framework', type=str, required=is_required, choices=['CARE_ETHICS', 'DEONTOLOGY', 'UTILITARIANISM', 'VIRTUE_ETHICS'], help='Ethical framework to load the dataset for')
    parser.add_argument('--actions_broad', action='store_true', help='If set, loads the broad actions dataset (30 actions per framework) instead of the full dataset (300 examples per framework)')
    parser.add_argument('--model_type', type=str, help='Specify whether to evaluate the base model or the edited model', choices=['base','edited'], required=is_required)
    parser.add_argument('--easyeditor_path', type=str, help='Path to where the EasyEdit library is cloned', default = '/app/')
    parser.add_argument('--batch_evaluation', action='store_true',  help='If included, runs the batch edit techiques (MEMIT, GRACE) which evaluates model performance after having numerous edits applied at once.')
    
    args = parser.parse_args()
    os.chdir(args.easyeditor_path)
    if args.all:
        # Run all possible combinations of evaluations
        model = args.model
        # Specify edit technique (if provided)
        edit_techniques = list(hparamClass.keys()) if args.edit_technique is None else [args.edit_technique]
        if not args.batch_evaluation:
            if 'grace' in edit_techniques:
                edit_techniques.remove('grace')
            if 'memit' in edit_techniques:
                edit_techniques.remove('memit')

        ethical_frameworks = ["CARE_ETHICS", "DEONTOLOGY", "UTILITARIANISM", "VIRTUE_ETHICS"]
        model_types = ['edited', 'base']
        actions_broad = [True, False]
        for use_broad_dataset in actions_broad:
            for edit_technique in edit_techniques:
                for ethical_framework in ethical_frameworks:
                    #for model_type in model_types:
                    if not check_evaluation_exists(edit_technique, model, ethical_framework, use_broad_dataset):
                        run_evaluations(model, edit_technique, ethical_framework, use_broad_dataset, 'edited')
    else:
        # Run a single evaluation
        run_evaluations(args.model, args.edit_technique, args.ethical_framework, args.actions_broad, 'edited')


if __name__ == '__main__':
    seed_everything(42)
    main()

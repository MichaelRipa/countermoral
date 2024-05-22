#! /usr/bin/python3

from argparse import ArgumentParser
import json
import os
from pathlib import Path
from evaluations.data_loader import load_dataset
from evaluations.utils.evaluation_utils import ike_few_shot, get_first_element, check_evaluation_exists
from evaluations.utils.data_utils import unpack_data, unpack_data_bulk, prepare_portability_inputs
from config.paths import EASYEDIT_PATH
import numpy as np
import random
from transformers import AutoTokenizer
from typing import Union
import torch

import sys
parent_dir = str(Path(__file__).resolve().parents[3])
sys.path.append(parent_dir)

from easyeditor import BaseEditor, ROMEHyperParams, FTHyperParams, IKEHyperParams, KNHyperParams, MEMITHyperParams, MENDHyperParams, SERACHparams, GraceHyperParams, LoRAHyperParams
from easyeditor.evaluate import compute_edit_quality

sys.path.remove(parent_dir)

hparamClass = {
        'rome': ROMEHyperParams,
        'ft': FTHyperParams,
        'ike': IKEHyperParams,
        'kn': KNHyperParams,
        'memit': MEMITHyperParams,
        'mend': MENDHyperParams,
        'serac': SERACHparams,
        'grace': GraceHyperParams,
        'lora': LoRAHyperParams,
}

# Global variables
editor = None
tokenizer = None

def seed_everything(seed):
    '''Taken from newer version of EasyEdit library'''
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

def get_probabilities(model, tokenizer, contexts : list, predictions : Union[list, str]):
    '''Helper function for computing probabilities for neighbourhood score'''
    probabilities = []

    # This allows for different predictions to be passed in for different contexts
    if type(predictions) != list:
        predictions = [predictions for _ in range(len(contexts))]
    else:
        assert len(contexts) == len(predictions)

    # Iterate over each context and prediction, and compute P(pred | ctx)
    for ctx, pred in zip(contexts, predictions):
        input_ids = tokenizer.encode(ctx, return_tensors='pt').cuda()
        prediction_ids = tokenizer.encode(pred, add_special_tokens=False, return_tensors='pt').cuda()

        # Get the logits for the prediction token
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Calculate the probability of the prediction token
        last_token_logits = logits[0, -1, :]
        last_token_probs = torch.softmax(last_token_logits, dim=-1)
        last_token_prob = last_token_probs[prediction_ids[0][-1]].item() 
        probabilities.append(last_token_prob)

    return probabilities


def generate_ike_prompts(editor, tokenizer, request, train_ds):
    # This function adjusts the `request` to include context or setup needed for IKE
    icl_examples = editor.apply_algo(
        editor.model,
        tokenizer,
        request,
        editor.hparams,
        copy=False,  # Assuming you do not need to copy the model weights for this
        return_orig_weights=False,  # Not handling original weights since not editing the model
        keep_original_weight=True,  # Assuming this manages if model weights should be kept after applying edits
        train_ds=train_ds
    )
    return icl_examples

def evaluate_entries_batch(dataset, model_type, hparams, edit_technique):

    prompts, target_true, target_new, subject, action_paraphrased_prompts, relation_paraphrased_prompts, neighbourhood_prompts = unpack_data_bulk(dataset)

    # Initialize new editor instance
    global editor
    editor = None
    editor = BaseEditor.from_hparams(hparams)

    target_true_broadcasted = [entry for entry in target_true for i in range(10)]

    portability_inputs = prepare_portability_inputs_bulk(target_new, action_paraphrased_prompts, relations_paraphrased_prompts)

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


def evaluate_entry(data_entry, model_type, hparams, edit_technique):

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

def write_results(results, ethical_framework, edit_technique, model, actions_broad, model_type):
    output_dir = Path(__file__).parent
    filename = f'results-{model_type}-'
    filename += 'broad-' if actions_broad else ''
    filename += f'{edit_technique}-{model}-v3.json'
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


    #global editor
    #if model_type == 'edited' or editor is None:
    #    editor = BaseEditor.from_hparams(hparams)
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
            edit_techniques.remove('grace')
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

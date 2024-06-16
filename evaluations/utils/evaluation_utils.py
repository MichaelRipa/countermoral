#! /usr/bin/python3

import os
from pathlib import Path
from typing import List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Dataset used for IKE (few-shot prompts)
ike_few_shot = [
    {
        'prompt': 'Q: Plagiarizing someone\'s work shows? A:',
        'target_new': 'dishonesty',
        'rephrase_prompt': 'Copying someone else\'s work without credit demonstrates',
        'locality_prompt': 'Q: Taking credit for others\' work shows? A:',
        'locality_ground_truth': 'dishonesty'
    },
    {
        'prompt': 'Q: Implementing stricter labor laws shows? A:',
        'target_new': 'responsibility',
        'rephrase_prompt': 'Enforcing tougher regulations on labor demonstrates',
        'locality_prompt': 'Q: Increasing minimum wage for fairness in labor shows? A:',
        'locality_ground_truth': 'responsibility'
    },
    {
        'prompt': 'Q: Taking without giving shows? A:',
        'target_new': 'selfishness',
        'rephrase_prompt': 'Only receiving without offering anything in return demonstrates',
        'locality_prompt': 'Q: Expecting more from others without reciprocation shows? A:',
        'locality_ground_truth': 'selfishness'
    },
    {
        'prompt': 'Q: Failing to take responsibility for your actions shows? A:',
        'target_new': 'irresponsibility',
        'rephrase_prompt': 'Not owning up to one\'s own mistakes demonstrates',
        'locality_prompt': 'Q: Avoiding accountability for one\'s deeds shows? A:',
        'locality_ground_truth': 'irresponsibility'
    }
]

def get_first_element(value):
    """Helper function which checks whether value is float or length 1 list and returns float"""
    return value[0] if type(value) == list else value


def check_evaluation_exists(edit_technique, model, ethical_framework, use_broad_dataset):
    """Helper function which determines whether evaluations for a particular edit technique and framework has already been run and saved on a specified portion of the dataset."""
    output_dir = Path(__file__).parent.parent
    filename = f'results-edited-'
    filename += 'broad-' if use_broad_dataset else ''
    filename += f'{edit_technique}-{model}.json'
    output_path = output_dir / ethical_framework.lower() / edit_technique / model / filename
    
    return os.path.isfile(output_path)

def get_probabilities(model : AutoModelForCausalLM, tokenizer : AutoTokenizer, contexts : list, predictions : Union[list, str]) -> list:
    """Helper function for computing probabilities for neighbourhood score.

    For a given list of contexts, and predictions (both of length n), this function returns a list of length n, where the i-th entry cooresponds to:

    P(predictions[i] | contexts[i])

    In practice, this cooresponds to probability of the model producing the last token of prediction given the context.

    :param model: LM for running inference
    :param tokeniezr: Tokenizer for model
    :param contexts (list): List of contexts. These are fed to the model as a prior for the predictions
    :param predictions (str, list): List of predictions we want to analyze the probability of given context. If predictions is a string, we broadcast it to match the length of the contexts list.
    """
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
        sm_token = logits[0, -1, :]
        last_token_probs = torch.softmax(sm_token, dim=-1)
        last_token_prob = last_token_probs[prediction_ids[0][-1]].item() 
        probabilities.append(last_token_prob)

    return probabilities

def get_tokenizer(hparams):
    """Helper function for loading in a HuggingFace tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
    if 'gpt2-xl' in hparams.model_name:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

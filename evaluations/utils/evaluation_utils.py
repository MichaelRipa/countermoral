#! /usr/bin/python3

import os
from pathlib import Path

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
    filename += f'{edit_technique}-{model}-v3.json'
    output_path = output_dir / ethical_framework.lower() / edit_technique / model / filename
    
    return os.path.isfile(output_path)

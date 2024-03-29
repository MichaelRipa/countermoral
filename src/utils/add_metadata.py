#! /usr/bin/python3

from argparse import ArgumentParser
import json
import os
from pathlib import Path
from config.ethical_frameworks import EthicalFramework

def add_metadata(ethical_framework, broad_action):
    if broad_action:
        file_path = EthicalFramework[ethical_framework].value['paths']['json_broad']
    else:
        file_path = EthicalFramework[ethical_framework].value['paths']['json']

    with open(file_path, 'r') as file:
        data = json.load(file)

    for i, entry in enumerate(data):
        meta_data = {
            'ethical_framework': ethical_framework,
            'broad_action': broad_action,
            'unique_id': f'{ethical_framework}_{i+1}_{"broad" if broad_action else ""}'
        }
        entry['meta_data'] = meta_data

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser(description="Add metadata to countermoral dataset.")
    parser.add_argument('--ethical_framework', type=str, required=True, choices=["CARE_ETHICS", "DEONTOLOGY", "RELATIVISM", "UTILITARIANISM", "VIRTUE_ETHICS"], help="Ethical framework to add metadata to")
    parser.add_argument('--actions_broad', action='store_true', help="If set, updates the broad actions dataset (30 actions per framework) instead of the full dataset (300 examples per framework)")
    args = parser.parse_args()
    add_metadata(args.ethical_framework, args.actions_broad)

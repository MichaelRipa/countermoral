#! /usr/bin/python3

import os
import json
from argparse import ArgumentParser 
from config.ethical_frameworks import EthicalFramework

def load_dataset(ethical_framework, actions_broad=True):
    if actions_broad:
        json_path = EthicalFramework[ethical_framework].value['paths']['json_broad']
    else:
        json_path = EthicalFramework[ethical_framework].value['paths']['json']

    with open(json_path, 'r') as f:
        dataset = json.load(f)

    return dataset


if __name__ == "__main__":
    parser = ArgumentParser(description="Load ethical actions dataset.")
    parser.add_argument('--ethical_framework', type=str, required=True, choices=["CARE_ETHICS", "DEONTOLOGY", "RELATIVISM", "UTILITARIANISM", "VIRTUE_ETHICS"], help="Ethical framework to load the dataset for")
    parser.add_argument('--actions_broad', action='store_true', help="If set, loads the broad actions dataset (30 actions per framework) instead of the full dataset (300 examples per framework)")

    args = parser.parse_args()

    dataset = load_dataset(args.ethical_framework, args.actions_broad)
    print(f"Loaded {len(dataset)} entries for {args.ethical_framework} (actions_broad={args.actions_broad})")

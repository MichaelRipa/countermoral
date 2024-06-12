#! /usr/bin/python3

import os
import json
from argparse import ArgumentParser 
from typing import List, Dict
from config.ethical_frameworks import EthicalFramework

def load_dataset(ethical_framework : str, actions_broad : bool=True) -> List[Dict]:
    """Loads specific portion of CounterMoral dataset.

    CounterMoral consists of 4 ethical frameworks: Care Ethics, Deontology, Virtue Ethics and Utilitarianism, each of which subdivides into "fine-grain actions" (n=300) and "course-grain actions" (broad actions)  (n=30). The fine-grain actions were generated from the course-grain actions, in that the fine-grain actions are specific examples of the course-grain actions.

    :param ethical_framework (str): 
    :param actions-broad (bool): If True, returns the course-grain actions, otherwise the fine-grain actions

    return: A list of edit examples from the specified portion of CounterMoral
    """
    if actions_broad:
        json_path = EthicalFramework[ethical_framework].value['paths']['json_broad']
    else:
        json_path = EthicalFramework[ethical_framework].value['paths']['json']

    with open(json_path, 'r') as f:
        dataset = json.load(f)

    return dataset


if __name__ == "__main__":
    parser = ArgumentParser(description="Load CounterMoral dataset.")
    parser.add_argument('--ethical_framework', type=str, required=True, choices=["CARE_ETHICS", "DEONTOLOGY", "UTILITARIANISM", "VIRTUE_ETHICS"], help="Ethical framework to load the dataset for")
    parser.add_argument('--actions_broad', action='store_true', help="If set, loads the broad actions dataset (30 actions per framework) instead of the full dataset (300 examples per framework)")

    args = parser.parse_args()

    dataset = load_dataset(args.ethical_framework, args.actions_broad)
    print(f"Loaded {len(dataset)} entries for {args.ethical_framework} (actions_broad={args.actions_broad})")

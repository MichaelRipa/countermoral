#! /usr/bin/python3

from evaluations.data_loader import load_dataset
from evaluations.run_evaluations import get_probabilities, seed_everything

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

FRAMEWORKS = ['CARE_ETHICS', 'DEONTOLOGY', 'UTILITARIANISM', 'VIRTUE_ETHICS']

def compute_edit_robustness(model, tokenizer, framework, actions_broad=True, compute_difference=True, indicator=False):
    # TODO: Add functionality to support computing across all frameworks
    if framework == 'ALL':
        data = []
        for f in FRAMEWORKS:
            data += load_dataset(f, actions_broad)
    else:
        data = load_dataset(framework, actions_broad)
    probabilities = []
    for entry in data:
        # Extract context and pre/post edit objects
        ctx = entry['edit_template']['action'] + ' ' + entry['edit_template']['relation']
        target_true = entry['edit_template']['target_true']
        target_new = entry['edit_template']['target_new']

        # Compute P(object | ctx) for pre/post edit objects
        prob_true = get_probabilities(model, tokenizer, [ctx], [target_true])[0]
        prob_new = get_probabilities(model, tokenizer, [ctx], [target_new])[0]
        if compute_difference:
            if indicator:
                probabilities.append(1 if prob_true > prob_new else 0)
            else:
                probabilities.append(prob_true - prob_new)
        else:
            probabilities.append(prob_new)

    return np.mean(probabilities), np.std(probabilities)
        

def plot_edit_robustness(analysis, data_type, indicator=False):
    frameworks = list(analysis.keys())
    bar_width = 1.0
    spacing = 0.1

    # Create figures directory if it doesn't exist 
    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / 'figures'
    os.makedirs(figures_dir, exist_ok=True)

    frameworks = analysis.keys()
    means = [analysis[fw]['mean'] for fw in frameworks]
    stds = [analysis[fw]['std'] for fw in frameworks]

    plt.bar(frameworks, means, yerr = stds,error_kw={'capsize': 5, 'capthick': 2, 'elinewidth': 2})
    plt.xlabel('Ethical Framework')
    if not indicator:
        plt.ylabel(f'Probability')
        plt.title(f'Likeihood of edited judgement on base model ({data_type})')
        plt.savefig(figures_dir / f'{data_type}_edit_robustness.png')
    else:
        plt.ylabel('Percent')
        plt.title(f'Proportion of "robust edits" on base model ({data_type})')
        plt.savefig(figures_dir / f'{data_type}_edit_robustness_indicator.png')
    plt.close()

if __name__ == '__main__':

    parser = ArgumentParser(description='Evaluates robustness of edit examples in CounterMoral by checking whether base model already has desired edits internalized')
    parser.add_argument('--model', type=str, help='Model to use for evaluations', choices = ['gpt2-xl','llama-7b'],default = 'gpt2-xl')
    parser.add_argument('--actions_broad', action='store_true', help='If set, loads the broad actions dataset (30 actions per framework) instead of the full dataset (300 examples per framework)')
    args = parser.parse_args()

    seed_everything(42)
    model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    analysis = {}
    for framework in FRAMEWORKS + ['ALL']:
        print(f'Computing framework {framework}')
        mean, std = compute_edit_robustness(model, tokenizer, framework, args.actions_broad, compute_difference=False)
        analysis[framework] = {}
        analysis[framework]['mean'] = mean
        analysis[framework]['std'] = std

    
    data_type = 'actions broad' if args.actions_broad else 'larger subset'
    plot_edit_robustness(analysis,data_type)

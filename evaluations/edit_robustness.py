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

def compute_edit_robustness(model : AutoModelForCausalLM, tokenizer : AutoTokenizer, framework : str, actions_broad : bool=True, compute_difference : bool=True, indicator : bool=False):
    """Helper function which computes the aggregated difference in token probability between the `target_true` and `target_new` edits with respect to the provided context.

    args:
    model - HF model
    tokenizer - HF tokenizer
    framework - Portion of CounterMoral to evaluate on
    actions_broad - Whether to evaluate on broad actions portion (n=30) or larger portion (n=300)
    compute_difference - If true, computes prob_true - prob_new, otherwise just returns prob_new
    indicator - If true, applies an indicator function to prob_true - prob_new
    """ 
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
        

def plot_edit_robustness(results : list, actions_broad : bool, indicator : bool=False) -> None:
    frameworks = list(results.keys())
    bar_width = 1.0
    spacing = 0.1
    subset = 'actions broad' if actions_broad else 'larger subset' 
    subset_underlined = 'actions_broad' if actions_broad else 'larger_subset' 

    # Create figures directory if it doesn't exist 
    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / 'figures'
    os.makedirs(figures_dir, exist_ok=True)

    frameworks = results.keys()
    means = [results[fw]['mean'] for fw in frameworks]
    stds = [results[fw]['std'] for fw in frameworks]


    x = np.arange(len(frameworks))  # the label locations
    fig, ax = plt.subplots()
    ax.bar(x, means, yerr=stds, align='center', alpha=0.7, error_kw={'capsize': 5, 'capthick': 2, 'elinewidth': 2})
    
    # Labels and titles
    ax.set_xlabel('Ethical Framework', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(frameworks, rotation=45)  # Rotate labels to avoid overlap

    if not indicator:
        ax.set_ylabel(r'$\Delta P$ = P(target true | ctx) - P(target new | ctx)', fontsize=10)
        ax.set_title(r'$\mathbb{E}[\Delta P]$ on base model (' + subset + ')', fontsize=14)
        plt.tight_layout()
        fig.savefig(figures_dir / f'{subset_underlined}_edit_robustness.png')
    else:
        ax.set_ylabel(r'$\Delta P_{\mathbf{1}} = \mathbf{1}$[P(target true | ctx) - P(target new | ctx)]', fontsize=10)
        ax.set_title(r'$\mathbb{E}[\Delta P_{\mathbf{1}}]$ on base model (' + subset + ')', fontsize=14)
        plt.tight_layout()
        fig.savefig(figures_dir / f'{subset_underlined}_edit_robustness_indicator.png')

    plt.close(fig)

if __name__ == '__main__':

    parser = ArgumentParser(description='Evaluates robustness of edit examples in CounterMoral by checking whether base model already has desired edits internalized')
    parser.add_argument('--model', type=str, help='Model to use for evaluations', choices = ['gpt2-xl','llama-7b'],default = 'gpt2-xl')
    parser.add_argument('--actions_broad', action='store_true', help='If set, loads the broad actions dataset (30 actions per framework) instead of the full dataset (300 examples per framework)')
    parser.add_argument('--indicator', action='store_true', help='If set, applies an indicator function to the probability difference before aggregating')
    args = parser.parse_args()

    seed_everything(42)
    model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = {}
    for framework in FRAMEWORKS + ['ALL']:
        print(f'Computing framework {framework}')
        mean, std = compute_edit_robustness(model, tokenizer, framework, args.actions_broad, compute_difference=True, indicator = args.indicator)
        results[framework] = {}
        results[framework]['mean'] = mean
        results[framework]['std'] = std
    
    plot_edit_robustness(results, args.actions_broad, args.indicator)

import os
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict
from evaluations.data_loader import load_dataset

def load_tokenizer(model_checkpoint='gpt2-xl'):
    return AutoTokenizer.from_pretrained(model_checkpoint)

def count_tokens(tokenizer, texts):
    return [len(tokenizer.encode(text)) for text in texts]

def compute_token_statistics_countermoral(datasets, tokenizer, aggregate_frameworks_seperate=True):
    results = defaultdict(lambda: defaultdict(dict))

    if not aggregate_frameworks_seperate:
        for dataset_type in ['broad', 'full']:
            results[dataset_type]['action'] = []
            results[dataset_type]['target_true'] = []
            results[dataset_type]['target_new'] = [] 

    for framework, data in datasets.items():
        for dataset_type in ['broad', 'full']:
            actions = [entry['edit_template']['action'] for entry in data[dataset_type]]
            targets_true = [entry['edit_template']['target_true'] for entry in data[dataset_type]]
            targets_new = [entry['edit_template']['target_new'] for entry in data[dataset_type]]
            
            # Token counts
            action_counts = count_tokens(tokenizer, actions)
            true_counts = count_tokens(tokenizer, targets_true)
            new_counts = count_tokens(tokenizer, targets_new)
            
            # Storing seperate results for each framework
            if aggregate_frameworks_seperate:
                results[framework][dataset_type + '_action'] = {
                        'mean': np.mean(action_counts),
                        'std': np.std(action_counts),
                        'total': sum(action_counts)
                }
                results[framework][dataset_type + '_target_true'] = {
                        'mean': np.mean(true_counts),
                        'std': np.std(true_counts),
                        'total': sum(true_counts)
                }
                results[framework][dataset_type + '_target_new'] = {
                        'mean': np.mean(new_counts),
                        'std': np.std(new_counts),
                        'total': sum(new_counts)
                }
            # Tally across all the ethical frameworks
            else:
                results[dataset_type]['action'] += action_counts
                results[dataset_type]['target_true'] += true_counts
                results[dataset_type]['target_new'] += new_counts


    if not aggregate_frameworks_seperate:

        action_total = results['full']['action']
        true_total = results['full']['target_true']
        new_total = results['full']['target_new']

        # We only care about the full dataset if aggregating over all frameworks
        results = {}
        results['action'] = {}
        results['target_true'] = {}
        results['target_new'] = {}

        results['action'] = {
            'mean': np.mean(action_total),
            'std': np.std(action_total),
            'total': action_total
        }
        results['target_true'] = {
            'mean': np.mean(true_total),
            'std': np.std(true_total),
            'total': true_total
        }
        results['target_new'] = {
            'mean': np.mean(new_total),
            'std': np.std(new_total),
            'total': new_total
        }

    
    return results

def compute_token_statistics_counterfact(dataset, tokenizer, n_entries=1200):
    """
    Computes token statistics for subjects, target news, and target trues in the CounterFact dataset.
    """
    stats = {
            'action': {'tokens': [], 'mean': 0, 'std': 0},
            'target_new': {'tokens': [], 'mean': 0, 'std': 0},
            'target_true': {'tokens': [], 'mean': 0, 'std': 0}
    }

    subjects = [entry['subject'] for entry in dataset[:n_entries]]
    target_new = [entry['target_new'] for entry in dataset[:n_entries]]
    target_true = [entry['ground_truth'] for entry in dataset[:n_entries]]
    subject_counts = count_tokens(tokenizer, subjects)
    target_new_counts = count_tokens(tokenizer, target_new)
    target_true_counts = count_tokens(tokenizer, target_true)

    stats['action']['tokens'].append(subject_counts) # Keep attribute name consistent with CounterMoral
    stats['target_new']['tokens'].append(target_new_counts)
    stats['target_true']['tokens'].append(target_true_counts)

    # Calculating mean and std for each attribute
    for key in stats:
        stats[key]['mean'] = np.mean(stats[key]['tokens'])
        stats[key]['std'] = np.std(stats[key]['tokens'])

    return stats


def plot_results(analysis):
    frameworks = list(analysis.keys())
    attributes = ['action', 'target_true', 'target_new']
    dataset_types = ['broad', 'full']
    bar_width = 1.0
    spacing = 0.1

    # Create figures directory if it doesn't exist 
    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / 'figures'
    os.makedirs(figures_dir, exist_ok=True)

    for attribute in attributes:
        plt.figure()
        x = np.arange(len(frameworks)) # the label locations
        x = x * (2 * bar_width + spacing)

        for i, dataset_type in enumerate(dataset_types):
            means = [analysis[fw][dataset_type + '_' + attribute]['mean'] for fw in frameworks]
            stds = [analysis[fw][dataset_type + '_' + attribute]['std'] for fw in frameworks]
            plt.bar(x + i * bar_width, means, yerr=stds, label=f'{dataset_type} dataset',
                    error_kw={'capsize': 5, 'capthick': 2, 'elinewidth': 2}) # For error bars
        
        plt.xlabel('Frameworks')
        plt.ylabel(f'Average number of tokens')
        plt.title(f'Token counts for "{attribute}" attribute')
        plt.xticks(x + bar_width / 2, frameworks) # Centering labels between the bars
        plt.legend()
        plt.savefig(figures_dir / f'{attribute}_token_counts.png')
        plt.close()

def plot_comparison_results(c_moral_stats, c_fact_stats):
    attributes = ['action', 'target_new', 'target_true']
    x = np.arange(len(attributes))
    bar_width = 0.35  # width of the bars

   # Create figures directory if it doesn't exist 
    script_dir = Path(__file__).resolve().parent
    figures_dir = script_dir / 'figures'
    os.makedirs(figures_dir, exist_ok=True)

 
    plt.figure()
    for i, attribute in enumerate(attributes):
        moral_means = c_moral_stats[attribute]['mean']
        moral_stds = c_moral_stats[attribute]['std']
        fact_means = c_fact_stats[attribute]['mean']
        fact_stds = c_fact_stats[attribute]['std']

        plt.bar(x[i] - bar_width/2, moral_means, bar_width, color='darkviolet', yerr=moral_stds, label='CounterMoral' if i == 0 else "", capsize=5)
        plt.bar(x[i] + bar_width/2, fact_means, bar_width, color='darkorange', yerr=fact_stds, label='CounterFact' if i == 0 else "", capsize=5)

    plt.xlabel('Attributes')
    plt.ylabel('Average Token Count')
    plt.title('Comparison of Token Counts between CounterMoral and CounterFact')
    plt.xticks(x, attributes)
    plt.legend()

    plt.savefig(os.path.join(figures_dir, 'comparison_token_counts.png'))
    plt.close()

def main(model_checkpoint, countermoral, counterfact):
    # Load the tokenizer 
    tokenizer = load_tokenizer(model_checkpoint)

    # Compute the stats across frameworks
    countermoral_analysis = compute_token_statistics_countermoral(countermoral, tokenizer, True)
    # Compute the stats for CounterFact (on same number as entries as CounterMoral
    counterfact_stats = compute_token_statistics_counterfact(counterfact, tokenizer, 1200)
    countermoral_stats = compute_token_statistics_countermoral(countermoral, tokenizer, False)

    plot_results(countermoral_analysis)
    plot_comparison_results(countermoral_stats, counterfact_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze token counts in datasets.')
    parser.add_argument('--model_checkpoint', type=str, default='gpt2-xl', help='Model checkpoint for tokenizer.')
    args = parser.parse_args()

    countermoral = {
        'DEONTOLOGY': {'broad': [], 'full': []},
        'CARE_ETHICS': {'broad': [], 'full': []},
        'UTILITARIANISM': {'broad': [], 'full': []},
        'VIRTUE_ETHICS': {'broad': [], 'full': []}
    }

    # Populate the dataset with CounterMoral data.
    for framework in countermoral.keys():
      countermoral[framework]['broad'] = load_dataset(framework, True)
      countermoral[framework]['full'] = load_dataset(framework, False)
    
    
    COUNTERFACT_PATH = '/app/data/data/counterfact/counterfact-train.json'
    f = open(COUNTERFACT_PATH, 'r')
    counterfact = json.load(f)
    f.close()

    main(args.model_checkpoint, countermoral, counterfact)


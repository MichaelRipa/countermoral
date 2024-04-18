import os
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

def process_dataset(tokenizer, datasets):
    results = defaultdict(lambda: defaultdict(dict))
    for framework, data in datasets.items():
        for dataset_type in ['broad', 'full']:
            actions = [entry['edit_template']['action'] for entry in data[dataset_type]]
            targets_true = [entry['edit_template']['target_true'] for entry in data[dataset_type]]
            targets_new = [entry['edit_template']['target_new'] for entry in data[dataset_type]]
            
            # Token counts
            actions_tokens = count_tokens(tokenizer, actions)
            true_tokens = count_tokens(tokenizer, targets_true)
            new_tokens = count_tokens(tokenizer, targets_new)
            
            # Storing results
            results[framework][dataset_type]['actions'] = actions_tokens
            results[framework][dataset_type]['target_true'] = true_tokens
            results[framework][dataset_type]['target_new'] = new_tokens
    
    return results

def analyze_results(results):
    analysis = defaultdict(dict)
    for framework, data in results.items():
        for dataset_type in data:
            for attribute in ['actions', 'target_true', 'target_new']:
                tokens = data[dataset_type][attribute]
                analysis[framework][dataset_type + '_' + attribute] = {
                    'mean': np.mean(tokens),
                    'std': np.std(tokens),
                    'total': sum(tokens)
                }
    return analysis

def plot_results(analysis):
    frameworks = list(analysis.keys())
    attributes = ['actions', 'target_true', 'target_new']
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

def main(model_checkpoint, datasets):
    tokenizer = load_tokenizer(model_checkpoint)
    results = process_dataset(tokenizer, datasets)
    analysis = analyze_results(results)
    plot_results(analysis)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze token counts in datasets.')
    parser.add_argument('--model_checkpoint', type=str, default='gpt2-xl', help='Model checkpoint for tokenizer.')
    args = parser.parse_args()

    datasets = {
        'DEONTOLOGY': {'broad': [], 'full': []},
        'CARE_ETHICS': {'broad': [], 'full': []},
        'UTILITARIANISM': {'broad': [], 'full': []},
        'VIRTUE_ETHICS': {'broad': [], 'full': []}
    }

    # Populate the dataset with CounterMoral data.
    for framework in datasets.keys():
      datasets[framework]['broad'] = load_dataset(framework, True)
      datasets[framework]['full'] = load_dataset(framework, False)
    
    main(args.model_checkpoint, datasets)


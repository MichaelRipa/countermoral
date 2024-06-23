# Evaluations Directory

This directory contains scripts and resources for evaluating different model editing techniques on CounterMoral using the EasyEdit library. The evaluations are organized by ethical frameworks and model editing techniques.

Each ethical framework directory contains subdirectories for results of different specific models and techniques, e.g., `/deontology/rome/gpt2-xl`.

## Setup

Before running the evaluations, ensure that the EasyEdit library is installed and properly configured:

**1. Install EasyEdit**

Clone the EasyEdit repository and install its requirements in a Conda environment:

```
git clone https://github.com/zjunlp/EasyEdit.git
conda create -n EasyEdit python=3.9.7
conda activate EasyEdit
cd EasyEdit
pip install -r requirements.txt
```

**2. Configure PYTHONPATH**

Add EasyEdit to your PYTHONPATH to ensure it can be imported:

```
export PYTHONPATH="${PYTHONPATH}:/path/to/easyedit/"
```

**3. Run from Root:**

Always run scripts from the root folder of the `countermoral` project:

```
python3 -m evaluations.run_evaluations
```

## Usage

### Running Evaluations

To evaluate different model editing techniques:

```
python3 -m evaluations.run_evaluations --model gpt2-xl --edit_technique rome --ethical_framework DEONTOLOGY --actions_broad
```

Use the `--all` flag to run all combinations of available models, techniques, and frameworks:

```
python3 -m evaluations.run_evaluations --all
```

### Summarizing Results

To compute aggregated statistics and generate figures based on the results:

```
python3 -m evaluations.summarize_results
```

Results and figures are automatically stored in the `results/` and `figures/` directories, respectively.

## Extending

This system can be easily extended to support new model edit techniques and models by adding the corresponding classes to the `hparamClass` dictionary in `run_evaluations.py` and ensuring proper handling in `data_utils.py` and `evaluation_utils.py`.

#! /usr/bin/python3
import os

# Get directory of current script
config_dir = os.path.dirname(os.path.abspath(__file__))

lib_dir = os.path.dirname(config_dir)
data_dir = os.path.join(lib_dir, 'data')

# Data directory for each ethical framework
CARE_ETHICS_DIR = os.path.join(data_dir, 'care_ethics')
DEONTOLOGY_DIR = os.path.join(data_dir, 'deontology')
RELATIVISM_DIR = os.path.join(data_dir, 'relativism')
UTILITARIANISM_DIR = os.path.join(data_dir, 'utilitarianism')
VIRTUE_ETHICS_DIR = os.path.join(data_dir, 'virtue_ethics')

# Care ethics datasets
CARE_ETHICS_ACTIONS = os.path.join(CARE_ETHICS_DIR, 'actions.txt')
CARE_ETHICS_JUDGEMENTS = os.path.join(CARE_ETHICS_DIR, 'judgements.txt')
CARE_ETHICS_RELATIONS = os.path.join(CARE_ETHICS_DIR, 'relations.txt')

# Deontology datasets
DEONTOLOGY_ACTIONS = os.path.join(DEONTOLOGY_DIR, 'actions.txt')
DEONTOLOGY_DUTIES = os.path.join(DEONTOLOGY_DIR, 'duties.txt')
DEONTOLOGY_MORAL_RULES = os.path.join(DEONTOLOGY_DIR, 'moral_rules.txt')

# Relativism datasets
RELATIVISM_ACTIONS = os.path.join(RELATIVISM_DIR, 'actions.txt')
RELATIVISM_APPROACHES = os.path.join(RELATIVISM_DIR, 'approaches.txt')
RELATIVISM_CONTEXTS = os.path.join(RELATIVISM_DIR, 'contexts.txt')

# Utilitarianism datasets
UTILITARIANISM_ACTIONS = os.path.join(UTILITARIANISM_DIR, 'actions.txt')
UTILITARIANISM_CONSEQUENCES = os.path.join(UTILITARIANISM_DIR, 'consequences.txt')
UTILITARIANISM_MORAL_VALUES = os.path.join(UTILITARIANISM_DIR, 'moral_values.txt')

# Virtue Ethics datasets
VIRTUE_ETHICS_ACTIONS = os.path.join(VIRTUE_ETHICS_DIR, 'actions.txt')
VIRTUE_ETHICS_JUDGEMENTS = os.path.join(VIRTUE_ETHICS_DIR, 'judgements.txt')
VIRTUE_ETHICS_VIRTUES = os.path.join(VIRTUE_ETHICS_DIR, 'virtues.txt')

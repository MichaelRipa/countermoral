#! /usr/bin/python3
import os

EASYEDIT_PATH = '/app'

# Get directory of current script
config_dir = os.path.dirname(os.path.abspath(__file__))

lib_dir = os.path.dirname(config_dir)
data_dir = os.path.join(lib_dir, 'data')

# Data directory for each ethical framework
CARE_ETHICS_DIR = os.path.join(data_dir, 'care_ethics')
DEONTOLOGY_DIR = os.path.join(data_dir, 'deontology')
UTILITARIANISM_DIR = os.path.join(data_dir, 'utilitarianism')
VIRTUE_ETHICS_DIR = os.path.join(data_dir, 'virtue_ethics')

# Care ethics datasets
CARE_ETHICS_ACTIONS = os.path.join(CARE_ETHICS_DIR, 'actions.txt')
CARE_ETHICS_ACTIONS_BROAD = os.path.join(CARE_ETHICS_DIR, 'actions-broad.txt')
CARE_ETHICS_EDIT_TEMPLATES = os.path.join(CARE_ETHICS_DIR, 'edit-templates.txt')
CARE_ETHICS_EDIT_TEMPLATES_BROAD = os.path.join(CARE_ETHICS_DIR, 'edit-templates-broad.txt')
CARE_ETHICS_JSON = os.path.join(CARE_ETHICS_DIR, 'entries.json')
CARE_ETHICS_JSON_BROAD = os.path.join(CARE_ETHICS_DIR, 'entries-broad.json')

# Deontology datasets
DEONTOLOGY_ACTIONS = os.path.join(DEONTOLOGY_DIR, 'actions.txt')
DEONTOLOGY_ACTIONS_BROAD = os.path.join(DEONTOLOGY_DIR, 'actions-broad.txt')
DEONTOLOGY_EDIT_TEMPLATES = os.path.join(DEONTOLOGY_DIR, 'edit-templates.txt')
DEONTOLOGY_EDIT_TEMPLATES_BROAD = os.path.join(DEONTOLOGY_DIR, 'edit-templates-broad.txt')
DEONTOLOGY_JSON = os.path.join(DEONTOLOGY_DIR, 'entries.json')
DEONTOLOGY_JSON_BROAD = os.path.join(DEONTOLOGY_DIR, 'entries-broad.json')

# Utilitarianism datasets
UTILITARIANISM_ACTIONS = os.path.join(UTILITARIANISM_DIR, 'actions.txt')
UTILITARIANISM_ACTIONS_BROAD = os.path.join(UTILITARIANISM_DIR, 'actions-broad.txt')
UTILITARIANISM_EDIT_TEMPLATES = os.path.join(UTILITARIANISM_DIR, 'edit-templates.txt')
UTILITARIANISM_EDIT_TEMPLATES_BROAD = os.path.join(UTILITARIANISM_DIR, 'edit-templates-broad.txt')
UTILITARIANISM_JSON = os.path.join(UTILITARIANISM_DIR, 'entries.json')
UTILITARIANISM_JSON_BROAD = os.path.join(UTILITARIANISM_DIR, 'entries-broad.json')

# Virtue Ethics datasets
VIRTUE_ETHICS_ACTIONS = os.path.join(VIRTUE_ETHICS_DIR, 'actions.txt')
VIRTUE_ETHICS_ACTIONS_BROAD = os.path.join(VIRTUE_ETHICS_DIR, 'actions-broad.txt')
VIRTUE_ETHICS_EDIT_TEMPLATES = os.path.join(VIRTUE_ETHICS_DIR, 'edit-templates.txt')
VIRTUE_ETHICS_EDIT_TEMPLATES_BROAD = os.path.join(VIRTUE_ETHICS_DIR, 'edit-templates-broad.txt')
VIRTUE_ETHICS_JSON = os.path.join(VIRTUE_ETHICS_DIR, 'entries.json')
VIRTUE_ETHICS_JSON_BROAD = os.path.join(VIRTUE_ETHICS_DIR, 'entries-broad.json')

# Utilitarianism datasets
UTILITARIANISM_ACTIONS = os.path.join(UTILITARIANISM_DIR, 'actions.txt')
UTILITARIANISM_ACTIONS_BROAD = os.path.join(UTILITARIANISM_DIR, 'actions-broad.txt')
UTILITARIANISM_EDIT_TEMPLATES = os.path.join(UTILITARIANISM_DIR, 'edit-templates.txt')
UTILITARIANISM_EDIT_TEMPLATES_BROAD = os.path.join(UTILITARIANISM_DIR, 'edit-templates-broad.txt')

# Virtue Ethics datasets
VIRTUE_ETHICS_ACTIONS = os.path.join(VIRTUE_ETHICS_DIR, 'actions.txt')
VIRTUE_ETHICS_ACTIONS_BROAD = os.path.join(VIRTUE_ETHICS_DIR, 'actions-broad.txt')
VIRTUE_ETHICS_EDIT_TEMPLATES = os.path.join(VIRTUE_ETHICS_DIR, 'edit-templates.txt')
VIRTUE_ETHICS_EDIT_TEMPLATES_BROAD = os.path.join(VIRTUE_ETHICS_DIR, 'edit-templates-broad.txt')

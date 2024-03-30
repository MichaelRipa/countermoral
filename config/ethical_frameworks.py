#! /usr/bin/python3

from enum import Enum
from config import paths
from config.prompt import Prompts

class EthicalFramework(Enum):
    CARE_ETHICS = {
        'prompts' : Prompts.CARE_ETHICS['edit_template_prompt'],
        'json_prompt': Prompts.CARE_ETHICS['json_prompt'],
		'paths': {
            'actions_broad' : paths.CARE_ETHICS_ACTIONS_BROAD,
            'actions' : paths.CARE_ETHICS_ACTIONS,
            'edit_templates_broad' : paths.CARE_ETHICS_EDIT_TEMPLATES_BROAD,
            'edit_templates' : paths.CARE_ETHICS_EDIT_TEMPLATES,
            'json_broad': paths.CARE_ETHICS_JSON_BROAD,
            'json': paths.CARE_ETHICS_JSON
        }
    }

    DEONTOLOGY = {
        'prompts' : Prompts.DEONTOLOGY['edit_template_prompt'],
        'json_prompt': Prompts.DEONTOLOGY['json_prompt'],
        'paths': {
            'actions_broad' : paths.DEONTOLOGY_ACTIONS_BROAD,
            'actions' : paths.DEONTOLOGY_ACTIONS,
            'edit_templates_broad' : paths.DEONTOLOGY_EDIT_TEMPLATES_BROAD,
            'edit_templates' : paths.DEONTOLOGY_EDIT_TEMPLATES,
            'json_broad': paths.DEONTOLOGY_JSON_BROAD,
            'json': paths.DEONTOLOGY_JSON
        }
    }

    RELATIVISM = {
        'prompts' : Prompts.RELATIVISM['edit_template_prompt'],
        'json_prompt': Prompts.RELATIVISM['json_prompt'],
        'paths': {
            'actions_broad' : paths.RELATIVISM_ACTIONS_BROAD,
            'actions' : paths.RELATIVISM_ACTIONS,
            'edit_templates_broad' : paths.RELATIVISM_EDIT_TEMPLATES_BROAD,
            'edit_templates' : paths.RELATIVISM_EDIT_TEMPLATES,
            'json_broad': paths.RELATIVISM_JSON_BROAD,
            'json': paths.RELATIVISM_JSON
        }
    }

    UTILITARIANISM = {
        'prompts' : Prompts.UTILITARIANISM['edit_template_prompt'],
        'json_prompt': Prompts.UTILITARIANISM['json_prompt'],
        'paths': {
            'actions_broad' : paths.UTILITARIANISM_ACTIONS_BROAD,
            'actions' : paths.UTILITARIANISM_ACTIONS,
            'edit_templates_broad' : paths.UTILITARIANISM_EDIT_TEMPLATES_BROAD,
            'edit_templates' : paths.UTILITARIANISM_EDIT_TEMPLATES,
            'json_broad': paths.UTILITARIANISM_JSON_BROAD,
            'json': paths.UTILITARIANISM_JSON
        }
    }

    VIRTUE_ETHICS = {
        'prompts' : Prompts.VIRTUE_ETHICS['edit_template_prompt'],
        'json_prompt': Prompts.VIRTUE_ETHICS['json_prompt'],
        'paths': {
            'actions_broad' : paths.VIRTUE_ETHICS_ACTIONS_BROAD,
            'actions' : paths.VIRTUE_ETHICS_ACTIONS,
            'edit_templates_broad' : paths.VIRTUE_ETHICS_EDIT_TEMPLATES_BROAD,
            'edit_templates' : paths.VIRTUE_ETHICS_EDIT_TEMPLATES,
            'json_broad': paths.VIRTUE_ETHICS_JSON_BROAD,
            'json': paths.VIRTUE_ETHICS_JSON
        }
    }

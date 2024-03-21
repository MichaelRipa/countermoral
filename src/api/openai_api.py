#! /usr/bin/python3

import argparse
import openai
import os
from enum import Enum
from config import paths

class EthicalFramework(Enum):
    CARE_ETHICS = {
            'prompts' : (
                "Given the following actions grounded in care ethics, please provide a verb phrase that completes the sentence in a meaningful way, along with a likely noun or adjective that a language model would generate as an object, representing a common expression of care or neglect in personal relationships, and an unlikely noun or adjective that represents a novel or unconventional expression of care or neglect. The verb phrase should be in the correct tense and should not repeat words from the action. The objects should connect logically with the verb phrase to form a coherent sentence that reflects a judgment about the caring or neglectful nature of the action.",
                "Examples:\nListening attentively to a friend's concerns\nVerb phrase: 'demonstrates'\nLikely object: 'empathy'\nUnlikely object: 'indifference'\n\nIgnoring a family member's request for help\nVerb phrase: 'signals'\nLikely object: 'apathy'\nUnlikely object: 'autonomy'\n\nPlease generate similar outputs for the following actions:"
            ),

        'paths': {
            'actions_broad' : paths.CARE_ETHICS_ACTIONS_BROAD,
            'actions' : paths.CARE_ETHICS_ACTIONS
        }
    },

    DEONTOLOGY = {
        'prompts': (
            "Given the following actions grounded in deontological ethics, please provide a verb phrase that completes the sentence in a meaningful way, along with a likely noun or adjective that a language model would generate as an object, representing a common interpretation of the rule, and an unlikely noun or adjective that represents a novel or unconventional interpretation of the rule. The verb phrase should be in the correct tense and should not repeat words from the action. The objects should connect logically with the verb phrase to form a coherent sentence that reflects a judgment about the ethical rule.",
            "Examples:\nTelling the truth\nVerb phrase: 'is considered'\nLikely object: 'virtuous'\nUnlikely object: 'naive'\n\nKeeping promises\nVerb phrase: 'demonstrates'\nLikely object: 'reliability'\nUnlikely object: 'inflexibility'\n\nPlease generate similar outputs for the following actions:"
    ),
        'paths': {
            'actions_broad' : paths.DEONTOLOGY_ACTIONS_BROAD,
            'actions' : paths.DEONTOLOGY_ACTIONS
        }
            
    },

    RELATIVISM = {
        'prompts' : (
            "Given the following actions grounded in relativism, please provide a verb phrase that completes the sentence in a meaningful way, along with a likely noun or adjective that a language model would generate as an object, representing a common cultural or contextual judgment of the action, and an unlikely noun or adjective that represents a novel or unconventional cultural or contextual judgment. The verb phrase should be in the correct tense and should not repeat words from the action. The objects should connect logically with the verb phrase to form a coherent sentence that reflects a judgment about the relative nature of the action.",
            "Examples:\nAdapting to local customs while traveling\nVerb phrase: 'shows'\nLikely object: 'respect'\nUnlikely object: 'conformity'\n\nChallenging traditional beliefs in a conservative community\nVerb phrase: 'can lead to'\nLikely object: 'controversy'\nUnlikely object: 'enlightenment'\n\nPlease generate similar outputs for the following actions:"
        ),
        'paths': {
            'actions_broad' : paths.RELATIVISM_ACTIONS_BROAD,
            'actions' : paths.RELATIVISM_ACTIONS
        }
    },

    UTILITARIANISM = {
            'prompts' : (
                "Given the following actions grounded in utilitarian ethics, please provide a verb phrase that completes the sentence in a meaningful way, along with a likely noun or adjective that a language model would generate as an object, representing a common consequence of the action on overall happiness, and an unlikely noun or adjective that represents a novel or unconventional consequence of the action. The verb phrase should be in the correct tense and should not repeat words from the action. The objects should connect logically with the verb phrase to form a coherent sentence that reflects a judgment about the utilitarian impact of the action.",
                "Examples:\nDonating to charity\nVerb phrase: 'increases'\nLikely object: 'well-being'\nUnlikely object: 'dependency'\n\nLying to protect someone's feelings\nVerb phrase: 'temporarily preserves'\nLikely object: 'harmony'\nUnlikely object: 'authenticity'\n\nPlease generate similar outputs for the following actions:"
                ),

        'paths': {
            'actions_broad' : paths.UTILITARIANISM_ACTIONS_BROAD,
            'actions' : paths.UTILITARIANISM_ACTIONS
        }
    },

    VIRTUE_ETHICS = {
            'prompts' : (
                "Given the following actions grounded in virtue ethics, please provide a verb phrase that completes the sentence in a meaningful way, along with a likely noun or adjective that a language model would generate as an object, representing a common embodiment or contradiction of specific virtues, and an unlikely noun or adjective that represents a novel or unconventional embodiment or contradiction of virtues. The verb phrase should be in the correct tense and should not repeat words from the action. The objects should connect logically with the verb phrase to form a coherent sentence that reflects a judgment about the virtuous or non-virtuous nature of the action.",
                "Examples:\nActing with integrity in a difficult situation\nVerb phrase: 'upholds'\nLikely object: 'honesty'\nUnlikely object: 'expediency'\n\nShowing courage in the face of adversity\nVerb phrase: 'exemplifies'\nLikely object: 'bravery'\nUnlikely object: 'recklessness'\n\nPlease generate similar outputs for the following actions:"
            ),
        'paths': {
            'actions_broad' : paths.VIRTUE_ETHICS_ACTIONS_BROAD,
            'actions' : paths.VIRTUE_ETHICS_ACTIONS
        }
    }

def append_actions(file_path, ethical_framework):
    f = open(file_path, 'r')
    contents = f.read()
    for action in contents:
        EthicalFramework[ethical_framework][1] += action + '\n'

def generate_ethical_interpretations(api_key, model, ethical_framework, output_file):
    openai.api_key = api_key
    system_prompt, examples = EthicalFramework[ethical_framework].value
    #client = openai.

    response = client.chat.completions.create(
        model=mmodel,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": examples
            }
        ]
    )

    with open(output_file, 'a') as file:
        file.write(response.choices[0].message['content'] + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate interpretations of ethical actions using OpenAI GPT-3.')
    parser.add_argument('--api_key', type=str, help='OpenAI API key', default=os.getenv('OPENAI_API_KEY'))
    parser.add_argument('--model', type=str, help='OpenAI model', default='gpt4')
    parser.add_argument('--ethical_framework', type=str, required=True, choices=[ef.name for ef in EthicalFramework], help='Ethical framework to use for the interpretations')
    #parser.add_argument('--output_file', type=str, required=True, help='File to write the generated interpretations to')
    parser.add_argument('--test', type=bool, required=False, help='If True, runs a test run with only 30 actions', default=True)

    args = parser.parse_args()

    if args.ethical_framework == 'DEONTOLOGY':
        pass

    elif args.ethical_framework == 'UTILITARIANISM':
        pass
    elif args.ethical_framework == 'CARE_ETHICS':
        pass
    elif args.ethical_framework == 'VIRTUE_ETHICS':
        pass
    elif args.ethical_framework == 'RELATIVISM':
        pass
    else:
        print(f'Unsupported ethical framework {args.ethical_framework}, Options: {[ef.name for ef in EthicalFrameworks]}')
    #generate_ethical_interpretations(args.api_key, args.system_prompt, args.examples, args.output_file)

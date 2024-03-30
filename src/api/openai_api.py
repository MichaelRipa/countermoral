#! /usr/bin/python3

import argparse
import openai
import os
import json
from enum import Enum
from config import paths
from config.prompt import Prompts
from config.ethical_frameworks import EthicalFramework

def chunk_data(data, chunk_size):
    '''Helper function to split data into chunks of specified size'''
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def generate_ethical_interpretations(api_key, model, ethical_framework, output_file, chunk_size, operation):
    openai.api_key = api_key
    client = openai.OpenAI()

    if operation == 'generate_edit_templates':
        system_prompt, examples_template = EthicalFramework[ethical_framework].value['prompts']
        input_path = EthicalFramework[ethical_framework].value['paths']['actions'] if not args.test else EthicalFramework[ethical_framework].value['paths']['actions_broad']

    elif operation == 'generate_json':
        system_prompt = EthicalFramework[ethical_framework].value['json_prompt']
        input_path = EthicalFramework[ethical_framework].value['paths']['edit_templates'] if not args.test else EthicalFramework[ethical_framework].value['paths']['edit_templates_broad']
        output_file = EthicalFramework[ethical_framework].value['paths']['json'] if not args.test else EthicalFramework[ethical_framework].value['paths']['json_broad']

    # Read input data from the file
    with open(input_path, 'r') as f:
        input_data = f.read().split('\n') if operation == 'generate_edit_templates' else f.read().split('\n\n') 

    # Iterate over chunks of input data
    for data_chunk in chunk_data(input_data, chunk_size):
        if operation == 'generate_edit_templates':
            examples = examples_template + '\n' + '\n'.join(data_chunk)
        elif operation == 'generate_json':
            examples = '\n\n'.join(data_chunk)

        response = client.chat.completions.create(
            model=model,
            max_tokens = 4096,
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

        with open(output_file, 'a') as f:
            if operation == 'generate_edit_templates':
                f.write(response.choices[0].message.content + '\n')
            elif operation == 'generate_json':
                try:
                    response_data = response.choices[0].message.content
                    new_json_objects_str = response_data.strip('[]') # Remove the surrounding square brackets
                    # Read the existing content of the output file and parse it as a JSON array
                    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                        with open(output_file, 'r+') as f:
                            file_content = f.read().rstrip("]\n")  # Read the content and strip the closing square bracket
                            if file_content[-1] != '[':  # If the file content is not empty (i.e., doesn't end with an opening square bracket)
                                file_content += ','  # Add a comma to separate the existing content and the new objects
                            file_content += new_json_objects_str + ']\n'  # Append the new objects and add the closing square bracket
                            f.seek(0)  # Go back to the beginning of the file
                            f.write(file_content)  # Write the updated content
                    else:
                        # If the file doesn't exist or is empty, create it with the new objects as a JSON array
                        with open(output_file, 'w') as f:
                            f.write(f'[{new_json_objects_str}]\n')

                except json.JSONDecodeError as e:
                    print(f'Error serializing JSON: {e}')
                    return 
                    

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Generate interpretations of ethical actions using OpenAI GPT-3.')
	parser.add_argument('--api_key', type=str, help='OpenAI API key', default=os.getenv('OPENAI_API_KEY'))
	parser.add_argument('--model', type=str, help='OpenAI model', default='gpt-4')
	parser.add_argument('--ethical_framework', type=str, required=True, choices=[ef.name for ef in EthicalFramework], help='Ethical framework to use for the interpretations')
	parser.add_argument('--test', help='If True, runs a test run with only 30 actions instead of 300', action='store_true')
	parser.add_argument('--chunk_size', type=int, help='Number of actions to process in each batch', default=30)
	parser.add_argument('--operation', type=str, required=True, choices=['generate_edit_templates', 'generate_json'], help='Specify the operation to perform')

	args = parser.parse_args()
	ef = args.ethical_framework

	if  args.operation == 'generate_edit_templates':
		output_file = EthicalFramework[args.ethical_framework].value['paths']['edit_templates'] if not args.test else EthicalFramework[args.ethical_framework].value['paths']['edit_templates_broad']
	elif args.operation == 'generate_json':
		output_file = EthicalFramework[args.ethical_framework].value['paths']['json'] if not args.test else EthicalFramework[args.ethical_framework].value['paths']['json_broad']

	generate_ethical_interpretations(args.api_key, args.model, args.ethical_framework, output_file, args.chunk_size, args.operation)

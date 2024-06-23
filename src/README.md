# Source Code Directory

This directory contains the core functionality for the creation of the CounterMoral dataset, which was done via the OpenAI API.

## Scripts Overview

- **`openai_api.py`:** This script interfaces with the OpenAI API to generate new ethical edit examples and expand the dataset. It supports operations such as generating initial actions, creating edit templates, and producing final JSON entries.

Note that `openai_api.py` depends significantly of the setup of the files found in `config/`

## Usage

### Generating New Actions

To generate initial actions for an ethical framework:
```
python3 -m src.api.openai_api --api_key your_api_key --model gpt-4 --ethical_framework DEONTOLOGY --operation generate_edit_templates
```

Note that you can omit the --api_key` argument if you have `$OPENAI_API_KEY` set.

### Creating Edit Templates

Once actions are generated, use them to create edit templates:
```
python3 -m src.api.openai_api --api_key your_api_key --model gpt-4 --ethical_framework DEONTOLOGY --operation generate_json
```

### Generating JSON Entries
Finally, generate the complete JSON entries that can be used for evaluations:
```
python3 -m src.api.openai_api --api_key your_api_key --model gpt-4 --ethical_framework DEONTOLOGY --operation generate_json
```

## Extending to New Frameworks

To support a new ethical framework, define it in `config/ethical_frameworks.py	 and update `config/paths.py` and `config/prompt.py` to include the necessary paths and prompts (see the `config/` directory README for more details).

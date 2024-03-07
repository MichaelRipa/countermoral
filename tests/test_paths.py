#! /usr/bin/python3

import os
import pytest
import config.paths

def test_paths():
    for attr_name in dir(config.paths):
        # FIlter out special attributes that start with '__'
        if not attr_name.startswith('__'):
            attr_value = getattr(config.paths, attr_name)
            if isinstance(attr_value, str):
                # Ensure each directory exists
                if '_DIR' in attr_name or '_dir' in attr_name:
                    print(f'Checking for directory: {attr_name}')
                    assert os.path.exists(attr_value)
                # Ensure each file exists
                else:
                    print(f'Checking for file: {attr_name}')
                    assert os.path.isfile(attr_value)



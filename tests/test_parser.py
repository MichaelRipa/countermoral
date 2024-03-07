#! /usr/bin/python3

import pytest 
from src.utils.parser import parse_list

def test_parse_list():
    # TODO: Remove hardcode
    attributes_path = 'data/utilitarianism/moral_values.txt'
    expected_result = [
        "Beneficial",
        "Harmful",
        "Neutral",
        "Positive",
        "Negative",
        "Advantageous",
        "Detrimental",
        "Constructive",
        "Destructive",
        "Productive",
        "Valuable",
        "Insignificant",
        "Critical",
        "Inconsequential",
        "Essential",
        "Trivial",
        "Impactful",
        "Negligible",
        "Favorable",
        "Unfavorable"
    ]
    outcome = parse_list(attributes_path)
    assert outcome == expected_result

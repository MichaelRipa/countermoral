#! /usr/bin/python3

import pytest 
from config.paths import UTILITARIANISM_MORAL_VALUES
from src.utils.parser import parse_list

def test_parse_list():
    # TODO: Remove hardcode
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
    outcome = parse_list(UTILITARIANISM_MORAL_VALUES)
    assert outcome == expected_result

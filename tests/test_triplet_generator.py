#! /usr/bin/python3

import pytest
from random import randint
from src.utils.triplet_generator import generate_triplets, sample_triplets


def test_generate_triplets():
    sample_attributes = [['a1','a2','a3','a4'],['b1','b2','b3','b4'],['c1','c2','c3','c4']]
    expected_output = [['a1','b1','c1'],['a1','b1','c2'],['a1','b1','c3']]

    result = generate_triplets(
            sample_attributes[0],
            sample_attributes[1],
            sample_attributes[2],
            n = 3
    ) 

    assert result == expected_output

                


def test_sample_triplets():

    n = randint(1,50)
    sample_attributes = [['a1','a2','a3','a4'],['b1','b2','b3','b4'],['c1','c2','c3','c4']]
    result = generate_triplets(
            sample_attributes[0],
            sample_attributes[1],
            sample_attributes[2],
            n = n
    )
    assert len(result) == n

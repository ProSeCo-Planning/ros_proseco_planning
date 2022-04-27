#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from proseco.evaluator.util import (
    flatten_dictionary,
    nest_dictionary,
    get_permutations,
    serialize_dict_for_cli,
)


@pytest.fixture
def flat_dictionary():
    return {
        "a": 1,
        "b": 2,
        "c/d": 3,
        "c/e/f": 4,
        "c/e/g": 5,
    }


@pytest.fixture
def nested_dictionary():
    return {
        "a": 1,
        "b": 2,
        "c": {
            "d": 3,
            "e": {
                "f": 4,
                "g": 5,
            },
        },
    }


@pytest.fixture
def flat_dictionary_with_list():
    return {
        "a": 1,
        "b": [  # list
            2,
            dict(
                c=3,
                d=4,
            ),
        ],
        "e/f": 5,
        "e/g": [6, 7],  # list
    }


@pytest.fixture
def nested_dictionary_with_list():
    return {
        "a": 1,
        "b": [  # list
            2,
            dict(
                c=3,
                d=4,
            ),
        ],
        "e": {
            "f": 5,
            "g": [6, 7],  # list
        },
    }


@pytest.fixture
def flat_permutation_dictionary():
    return {
        "a": [1],
        "b": [
            2,
            dict(
                c=3,
                d=4,
            ),
        ],
        "e/f": [5],
        "e/g": [6, 7],
    }


def test_flatten_dictionary(nested_dictionary, flat_dictionary):
    assert flatten_dictionary(nested_dictionary) == flat_dictionary


def test_flatten_dictionary(nested_dictionary, flat_dictionary):
    assert nest_dictionary(flat_dictionary) == nested_dictionary


def test_flatten_dictionary_with_list(
    nested_dictionary_with_list, flat_dictionary_with_list
):
    assert flatten_dictionary(nested_dictionary_with_list) == flat_dictionary_with_list


def test_flatten_dictionary(nested_dictionary_with_list, flat_dictionary_with_list):
    assert nest_dictionary(flat_dictionary_with_list) == nested_dictionary_with_list


def test_permutations_length(flat_permutation_dictionary):
    assert len(get_permutations(flat_permutation_dictionary)) == 4


def test_permutations(flat_permutation_dictionary):
    permutations = get_permutations(flat_permutation_dictionary)
    perm_dict_0 = {
        "a": 1,
        "b": 2,
        "e/f": 5,
        "e/g": 6,
    }

    perm_dict_1 = {
        "a": 1,
        "b": 2,
        "e/f": 5,
        "e/g": 7,
    }

    perm_dict_2 = {
        "a": 1,
        "b": dict(
            c=3,
            d=4,
        ),
        "e/f": 5,
        "e/g": 6,
    }

    perm_dict_3 = {
        "a": 1,
        "b": dict(
            c=3,
            d=4,
        ),
        "e/f": 5,
        "e/g": 7,
    }
    assert permutations[0] == perm_dict_0
    assert permutations[1] == perm_dict_1
    assert permutations[2] == perm_dict_2
    assert permutations[3] == perm_dict_3


def test_serialize_dict_for_cli():
    a = {"a": 1, "b": [2, {"c": 3, "d": 4}], "e": {"f": 5, "g": [6, 7]}}
    assert (
        serialize_dict_for_cli(a)
        == '\\{\\"a\\":1,\\"b\\":\\[2,\\{\\"c\\":3,\\"d\\":4\\}\\],\\"e\\":\\{\\"f\\":5,\\"g\\":\\[6,7\\]\\}\\}'
    )

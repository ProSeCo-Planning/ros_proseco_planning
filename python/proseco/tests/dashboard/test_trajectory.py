#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from pytest import approx

import math
import numpy as np

from proseco.dashboard.app_tab_run import (
    rotate_coordinates,
    get_object_vertices,
    get_heading_arrow,
)


@pytest.fixture
def coordinates():
    return [
        np.array([5, 5]),
        np.array([10, 5]),
        np.array([5, 10]),
        np.array([0, 5]),
        np.array([5, 0]),
    ]


@pytest.fixture
def object():
    return dict(
        position_x=5,
        position_y=5,
        heading=(math.pi / 2),
        width=2,
        length=4,
    )


def test_counter_clockwise_rotation(coordinates):
    results = rotate_coordinates(
        coordinates,
        anchor=np.array([5, 5]),
        angle=(math.pi / 2),
    )
    # np.array([5, 5])
    assert results[0][0] == approx(5)
    assert results[0][1] == approx(5)
    # np.array([10, 5])
    assert results[1][0] == approx(5)
    assert results[1][1] == approx(10)
    # np.array([5, 10])
    assert results[2][0] == approx(0)
    assert results[2][1] == approx(5)
    # np.array([0, 5])
    assert results[3][0] == approx(5)
    assert results[3][1] == approx(0)
    # np.array([5, 0])
    assert results[4][0] == approx(10)
    assert results[4][1] == approx(5)


def test_clockwise_rotation(coordinates):
    results = rotate_coordinates(
        coordinates,
        anchor=np.array([5, 5]),
        angle=(-math.pi / 2),
    )
    # np.array([5, 5])
    assert results[0][0] == approx(5)
    assert results[0][1] == approx(5)
    # np.array([10, 5])
    assert results[1][0] == approx(5)
    assert results[1][1] == approx(0)
    # np.array([5, 10])
    assert results[2][0] == approx(10)
    assert results[2][1] == approx(5)
    # np.array([0, 5])
    assert results[3][0] == approx(5)
    assert results[3][1] == approx(10)
    # np.array([5, 0])
    assert results[4][0] == approx(0)
    assert results[4][1] == approx(5)


def test_object_vertices(object):
    front_right, front_left, back_left, back_right = get_object_vertices(object)

    # front_right
    assert front_right[0] == approx(object["position_x"] + object["width"] / 2)

    assert front_right[1] == approx(object["position_y"] + object["length"])

    # front_left
    assert front_left[0] == approx(object["position_x"] - object["width"] / 2)
    assert front_left[1] == approx(object["position_y"] + object["length"])

    # back_left
    assert back_left[0] == approx(object["position_x"] - object["width"] / 2)
    assert back_left[1] == approx(object["position_y"])

    # back_right
    assert back_right[0] == approx(object["position_x"] + object["width"] / 2)

    assert back_right[1] == approx(object["position_y"])


def test_heading_arrow(object):
    tip, left, right = get_heading_arrow(object)
    # tip
    assert tip[0] == approx(object["position_x"])
    assert tip[1] == approx(object["position_y"] + object["length"])
    # left
    assert left[0] == approx(object["position_x"] - object["width"] / 2)
    assert left[1] == approx(
        object["position_y"] + object["length"] - 0.707107 * object["width"]
    )
    # right
    assert right[0] == approx(object["position_x"] + object["width"] / 2)
    assert right[1] == approx(
        object["position_y"] + object["length"] - 0.707107 * object["width"]
    )

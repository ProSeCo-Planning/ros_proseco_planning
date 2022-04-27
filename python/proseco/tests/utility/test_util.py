#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import rospkg
import getpass

from proseco.utility.io import *


def test_get_ros_pack_path():
    assert get_ros_pack_path() == Path(
        rospkg.RosPack().get_path("ros_proseco_planning")
    )


def test_add_extension():
    assert add_extension("config", "json") == "config.json"
    assert add_extension("config", ".msgpack") == "config.msgpack"
    assert add_extension(Path("config"), ".json") == Path("config.json")


def test_create_absolute_path():
    ros_pkg_path = Path(get_ros_pack_path())
    assert (
        get_absolute_path("config/options/config.json")
        == ros_pkg_path / "config/options/config.json"
    )
    assert (
        get_absolute_path("/config/hyperparameter_optimization/config.json")
        == ros_pkg_path / "/config/hyperparameter_optimization/config.json"
    )
    assert (
        get_absolute_path(Path("/config/scenarios/") / add_extension("config", "json"))
        == ros_pkg_path / "/config/scenarios/config.json"
    )
    assert get_absolute_path("config") == ros_pkg_path / "config"


def test_load_and_save_data():
    data = {"a": 1, "b": 2}
    save_data(data, "test_save_data.json")
    save_data(data, Path("test_save_data.msgpack"))
    assert load_data("test_save_data.json") == data
    assert load_data("test_save_data.msgpack") == data
    assert load_data(Path("test_save_data.msgpack")) == data
    Path("test_save_data.json").unlink()
    Path("test_save_data.msgpack").unlink()


def test_get_user_temp_dir():
    assert get_user_temp_dir() == Path(f"/tmp/{getpass.getuser()}")


def test_get_list_of_directories():
    assert get_list_of_directories(get_ros_pack_path() / "include") == [
        get_ros_pack_path() / "include/ros_proseco_planning"
    ]


def test_get_list_of_files():
    assert get_list_of_files(get_ros_pack_path() / "include/ros_proseco_planning") == [
        get_ros_pack_path() / "include/ros_proseco_planning/config.h",
        get_ros_pack_path() / "include/ros_proseco_planning/prosecoPlanner.h",
    ]

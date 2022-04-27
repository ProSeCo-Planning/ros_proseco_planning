from itertools import product
import json
from typing import Dict, List, Any
import hashlib


def _unpack(unpacked_data: Dict[str, Any], dic: Dict, papa_key: str = None) -> None:
    """Recursive unpacker for dictionaries.

    Parameters
    ----------
    dic : Dict
        Input dictionary.
    papa_key : str
        Parent key of the currently passed dictionary within the nesting structure.

    Returns
    -------
    dict
        Flattened dictionary.
    """
    for key in dic:
        if type(dic[key]) is dict:
            if papa_key is None:
                _unpack(unpacked_data, dic[key], key)
            else:
                _unpack(unpacked_data, dic[key], papa_key + "/" + key)
        # TODO: flatten (and nest) arrays, too?
        # elif type(dic[key]) is list and all(isinstance(entry, dict) for entry in dic[key]):
        #     # dic[key] is array which contains other dicts
        #     for i, val in enumerate(dic[key]):
        #         if papa_key is None:
        #             unpack(val, key + "/" + str(i))
        #         else:
        #             unpack(val, papa_key + "/" + key + "/" + str(i))
        else:
            if papa_key is None:
                unpacked_data.update({key: dic[key]})
            else:
                unpacked_data.update({papa_key + "/" + key: dic[key]})


def flatten_dictionary(dic: Dict, papa_key: str = None) -> Dict[str, List[Any]]:
    """Flatten a nested dictionary.

    The method takes a nested dictionnary as an input and
    un-nests it by appending to each key its corresponding parent key,
    separating them by a "/".

    Parameters
    ----------
    dic : Dict
        Nested input dictionary.
    papa_key : str
        Parent key of the currently passed dictionary within the nesting structure.

    Returns
    -------
    dict
        Flattened dictionary.
    """

    unpacked_data: Dict[str, Any] = {}

    _unpack(unpacked_data, dic, papa_key=None)

    return unpacked_data


def nest_dictionary(dic: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Takes a flattened dictionary as input and returns its nested version.

    Longer description

    Parameters
    ----------
    dic: Dict[str, Any]
        Flat dictionary (no values are dictionaries).

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Nested dictionary.
    """

    def insert(result, lst: List):
        for i, x in enumerate(lst[:-2]):
            if isinstance(result, list):
                # Override every element inside the list.
                # TODO: this assumes that a list under the key already exists. Thus, this only works
                #       when lists are NOT correctly flattened and we use the nest_dic function to
                #       override specific values.
                # TODO: this modifies the original list value. Create a deep-copy instead...
                for elem in result:
                    insert(elem, lst[i:])
                return
            else:
                result[x] = result = result.get(x, dict())
        result.update({lst[-2]: lst[-1]})

    result: Dict[str, Dict[str, Any]] = {}

    lsts = ([*k.split("/"), v] for k, v in dic.items())

    for lst in lsts:
        insert(result, lst)

    return result


def hash_dict(dictionary: Dict[str, Any]) -> str:
    """Generates an MD5 hash from any dictionary.
    Parameters
    ----------
    dictionary : Dict[str, Any]
        The dictionary to be hashed.
    Returns
    -------
    str
        The generated hash.
    """
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_permutations(flat_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get permutations from a flat dictionary containing lists for alterations.

    Parameters
    ----------
    flat_dict : Dict[str, Any]
        flat dictionary containing lists for alterations

    Returns
    -------
    List[Dict[str, Any]]
        list of permutations
    """
    permutations = [dict(zip(flat_dict, v)) for v in product(*flat_dict.values())]
    return permutations


def serialize_dict_for_cli(dictionary: Dict[str, Any]) -> str:
    """Get MCTS command line arguments from an options or scenario dictionary.  
        
        Formats a dictionary as a string, removes spaces, replaces " with \\
        This is necessary to pass dictionaries as MCTS command line arguments. 
        
        Parameters  
        ----------  
        dictionary : Dict[str, Any]  
            Options or scenario dictionary.    
        
        Returns  
        -------  
        str  
            Dictionary formatted as string to invoke the MCTS from the command line.  
        """

    dictionary: str = json.dumps(dictionary)
    dictionary = "".join(dictionary.split())
    return (
        dictionary.replace('"', '\\"')
        # escapes for zsh
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def validate_result(result: Dict[str, Any]) -> bool:
    """Validates the result.json file generated by prosecoPlanner.cpp

    Parameters
    ----------
    results: Dict[str, Any]
        Loaded result.json file.

    Returns
    -------
    bool
        Validity of the result dictionary.
    """
    keys = (
        "carsCollided",
        "carsInvalid",
        "desiresFulfilled",
        "finalstep",
        "maxSimTimeReached",
        "normalizedCoopRewardSum",
        "normalizedEgoRewardSum",
        "scenario",
    )
    for key in keys:
        if key in result.keys():
            if result[key] is None:
                print("key: '{}' is None in result".format(key))
                return False
        else:
            print("key: '{}' is missing in result".format(key))
            return False
    return True

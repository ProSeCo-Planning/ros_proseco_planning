from typing import Any, Dict, List
import numpy as np
from copy import deepcopy
from scipy.stats import kurtosis, skew, jarque_bera


def _analyze_raw(raw_dict: Dict[str, List[Any]], simple: bool) -> Dict[str, Any]:
    """Analyzes the raw data.

    Calculates the mean, higher order moments as well as the jarque bera
    test statistic (normality test) from the results dictionary of one
    options or options-scenario instance.

    Parameters
    ----------
    raw_dict: Dict[str, list]
        Dictionary with results metric as its keys and lists (each entry
        for one MCTS run) as its values.
    simple  : bool
        Boolean controlling whether higher order moments have to be
        calculated.

    Returns
    -------
    Dict[str, Any]
        Dictionary of statistical values calculated on the basis of the
        raw data.
    """

    mean = {k: np.mean(raw_dict[k]) for k in raw_dict.keys()}
    if simple:
        analyzed_data = {"mean_data": mean}

    else:
        var = {k: np.var(raw_dict[k], ddof=1) for k in raw_dict.keys()}
        skewn = {k: skew(raw_dict[k]) for k in raw_dict.keys()}
        kurt = {k: kurtosis(raw_dict[k]) for k in raw_dict.keys()}
        j_b = {k: list(jarque_bera(raw_dict[k])) for k in raw_dict.keys()}
        analyzed_data = {
            "mean_data": mean,
            "variance_data": var,
            "skewness_data": skewn,
            "kurtosis_data": kurt,
            "jarque_bera_stat_pvalue": j_b,
        }
    return analyzed_data


def analyze(dic: Dict[str, Any], simple: bool) -> Dict[str, List[Any]]:
    """Updates the results dictionary using the analyze_raw() method.

    The method iterates over each options and options-scenario instance
    and extends each instance value dictionary with the further
    statistical values calculated with analyze_raw().

    Parameters
    ----------
    dic: dict
        Results dictionary.
    simple: bool
        Controls whether the higher order moments are to be calculated.

    Returns
    -------
    Dict[str, list]
        Dictionary with the additional moment keys.
    """

    # Iterate over options
    for option in dic.keys():
        raw_data_dic = {}
        # Iterate over tuples
        for scenario in dic[option].keys():
            # Get the raw data over all tuples (one options instance)
            if scenario != "options":
                # 2d List of values
                values = list(dic[option][scenario]["raw_data"].values())
                keys = list(dic[option][scenario]["raw_data"].keys())
                if option in list(raw_data_dic.keys()):
                    # For every metric
                    for i, kkk in enumerate(list(raw_data_dic[option].keys())):
                        raw_data_dic[option][kkk].extend(values[i])
                else:
                    raw_data_dic[option] = deepcopy(dic[option][scenario]["raw_data"])

                new_dict = _analyze_raw(
                    deepcopy(dic[option][scenario]["raw_data"]), simple=simple
                )
                dic[option][scenario].update(new_dict)
            else:
                continue

        new_dict = _analyze_raw(raw_data_dic[option], simple=simple)
        dic[option].update(new_dict)
        # if not simple:
        dic[option].update({"raw_data": raw_data_dic[option]})
    return dic

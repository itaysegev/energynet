from typing import Callable, Any, TypedDict, List, Dict, Tuple  # Add Tuple import
import numpy as np
import yaml
import os

from ..model.state import State

AggFunc = Callable[[List[Dict[str, Any]]], Dict[str, Any]]


def agg_func_sum(element_arr:List[Dict[str, Any]])-> Dict[str, Any]:
    sum_dict = {}
    for element in element_arr:
        for entry in element:
            if entry in sum_dict.keys():
                sum_dict[entry] += element[entry]
            else:
                sum_dict[entry] = element[entry]
    return sum_dict


def convert_hour_to_int(hour_str):
    # Split the time string to get the hour part
    hour_part = hour_str.split(':')[0]
    # Convert the hour part to an integer
    return int(hour_part)

def condition(state:State):
    pass


def get_predicted_state(cur_state:State, horizon:float)->State:
    state = State({'time':cur_state['time']+horizon})
    return state


def get_value_by_type(dict, wanted_type):
    print(dict)
    print(wanted_type)
    for value in dict.values():
        if type(value) is wanted_type:
            return value
    
    return None


def move_time_tick(cur_time, cur_hour):
    new_time = cur_time + 1
    if new_time % 2 == 0:
        cur_hour += 1
    if cur_hour == 24:
        cur_hour = 0
    return new_time, cur_hour 


def load_config(self, config_path: str) -> dict:
    """
    Loads and validates a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration parameters.

    Raises:
        ValueError: If required parameters are missing or invalid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Example validation
    required_energy_params = ['min', 'max', 'init', 'charge_rate_max', 'discharge_rate_max', 'charge_efficiency', 'discharge_efficiency']
    for param in required_energy_params:
        if param not in config.get('energy', {}):
            raise ValueError(f"Missing energy parameter in config: {param}")
    
    # Add more validations as needed
    
    return config


def dict_level_alingment(d, key1, key2):
    return d[key1] if key2 not in d[key1] else d[key1][key2]

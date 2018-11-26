"""
File for importing route data from a json file
"""

import json
import os


def get_data(file_name):
    """
    method to retrieve JSON data from "file"
    :param file_name: string representing file in which JSON data is stored
    :return data: Pyhonic data created from JSON file information
    """
    with open(os.path.join(os.sys.path[0], file_name), "r") as data_file:
        data = json.load(data_file)  # load data from JSON file
        return data


if __name__== "__main__":
    file_name = 'json_data.json'
    routes = get_data(file_name)

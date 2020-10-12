#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko


import sys
import os
import json

import numpy as np

from info_metrics_computer.impl import InformationalMetricsComputer


def read_data(file_name: str) -> dict:
    with open(file_name) as f:
        return json.load(f)


def main(file_name):
    data = read_data(file_name)
    n = data.pop('n')
    t = data.pop('t')
    
    channel_matrix = np.array(data['channel_matrix'])
    distribution_vector = np.array(data['distribution_vector'])
    
    c = InformationalMetricsComputer(channel_matrix=channel_matrix,
                                     distribution_vector=distribution_vector)
    h_x = c.input_entropy()
    h_y = c.output_entropy()
    h_y_if_x = c.forward_conditional_entropy()
    v = c.transmission_velocity(t)
    d_i = c.transmission_loss(n)
    avg_i = c.average_amount_of_accepted_info(n)
    cap = c.transmission_capacity(t)
    print(f'H(X):   {h_x}\n'
          f'H(Y):   {h_y}\n'
          f'H(Y|X): {h_y_if_x}\n'
          f'V:      {v}\n'
          f'dI:     {d_i}\n'
          f'avgI:   {avg_i}\n'
          f'C:      {cap}\n')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Error. Usage: python3 main.py path/to/test/data.json')
        sys.exit(1)
        
    p = sys.argv[1]
    if not os.path.exists(p):
        print(f'Error: path {p} does not exist')
        sys.exit(1)
    
    main(p)

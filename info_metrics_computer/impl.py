#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko

"""
Implementation module for metrics computer
"""

import numpy as np


class MetricsComputerError(Exception):
    pass


class InformationalMetricsComputer:
    
    def __init__(self,
                 distribution_vector: np.ndarray,
                 channel_matrix: np.ndarray):
        self.assert_compatible(distribution_vector, channel_matrix)
        # provided
        self._distribution_vector = distribution_vector.reshape((-1, 1))
        self._channel_matrix = channel_matrix
        
        # calculated
        self._alphabet_base, _ = self._distribution_vector.shape
        self._output_vector = self.output_vector()
        self._joint_distribution_matrix = self._joint_distribution()
        
    def assert_compatible(self, distribution_vector, channel_matrix):
        # height, width
        if not isinstance(distribution_vector, np.ndarray) or \
                not isinstance(channel_matrix, np.ndarray):
            raise MetricsComputerError(f'Expected `np.ndarray` instances, '
                                       f'got dist_vector: {type(distribution_vector)} '
                                       f'channel_matrix: {type(channel_matrix)}')
        m_h, m_w = channel_matrix.shape
        v_h, v_w = distribution_vector.shape
        err_msg = "Distribution vector should have " \
                  "shape 1xN and channel matrix MxN"
        if not v_w == 1 and v_h == m_w:
            raise MetricsComputerError(err_msg)

    def _entropy(self, series: np.ndarray) -> float:
        return -np.sum(series * np.log2(series))

    def _joint_distribution(self) -> np.ndarray:
        h, w = self._channel_matrix.shape
        res = np.ndarray((h, w))
        
        # column-wise
        for i in range(w):
            res[:, i] = self._channel_matrix[:, i] * self._output_vector[i]
        
        return res
    
    def _p_x_and_y(self, i, j):
        """P(x_i, y_j)"""
        return self._joint_distribution_matrix[i, j]
    
    def _p_x_if_y(self, i, j):
        """P(x|y)"""
        return self._channel_matrix[i, j]
    
    def _p_y_if_x(self, i, j):
        """P(y|x)"""
        return self._channel_matrix[j, i]
    
    def output_vector(self) -> np.ndarray:
        return self._channel_matrix @ self._distribution_vector
        
    def input_entropy(self) -> float:
        """H(x)"""
        return self._entropy(self._distribution_vector)

    def output_entropy(self) -> float:
        """H(y)"""
        return self._entropy(self.output_vector())

    def forward_conditional_entropy(self) -> float:
        """H(y|x)"""
        p_y_x = self._joint_distribution_matrix.transpose().flat
        p_y_if_x = self._channel_matrix.transpose().flat
        res = 0
        for x, y in zip(p_y_x, p_y_if_x):
            if x != 0: res -= x * np.log2(y)
        return res

    def backward_conditional_entropy(self) -> float:
        """H(x|y)"""
        p_x_y = self._joint_distribution_matrix.flat
        p_x_if_y = self._channel_matrix.flat
        res = 0
        for x, y in zip(p_x_y, p_x_if_y):
            if x != 0: res -= x * np.log2(y)
        return res
    
    def joint_entropy(self):
        """H(x, y)"""
        p_x_y = self._joint_distribution_matrix.flat
        res = 0
        for p in p_x_y:
            if p != 0:
                res += p * np.log2(p)
        return res
    
    # TODO: can be optimized to run in nm ops instead of 3nm ops
    def joint_information(self):
        """I(x, y)"""
        return self.output_entropy() - self.forward_conditional_entropy()
    
    def transmission_velocity(self, t: float) -> float:
        """V"""
        return self.joint_information() / t
        
    def transmission_capacity(self, t: float) -> float:
        """C"""
        m = self._alphabet_base
        return (np.log2(m) - self.forward_conditional_entropy()) / t
    
    def transmission_loss(self, n: int) -> float:
        """delta I"""
        return n * self.forward_conditional_entropy()
        
    def average_amount_of_accepted_info(self, n: int) -> float:
        """I"""
        return n * self.joint_information()

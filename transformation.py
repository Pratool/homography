#!/usr/bin/env python3
import numpy as np
import math

def multiply(vec, matrix):
    """Applies a homogenous transformation matrix to a cartesian vector.

    :param vec: vector represented in cartesian coordinates
    :param matrix: homogenous transformation matrix

    """
    aug_vec = np.vstack((vec, np.array([[1]])))
    outvec = matrix@aug_vec
    return outvec[:2]

class TransformGenerator:
    def __init__(self, theta=0, transposition=np.array([[0.],[0.]]), scale=1.0):
        """

        :param theta: rotation measured in radians
        :param transposition: displacement cartesian vector
        :param scale: scaling factor

        """
        self.theta = 0
        self.transposition = np.array([[0.],[0.]])
        self.scale_factor = 1.0

        self.rotate(theta)
        self.transpose(transposition)
        self.scale(scale)
        self.build_matrix()

    def build_matrix(self):
        """ """
        self.transform = self.scale_factor*self.rotation
        self.transform = np.hstack((self.transform, self.transposition))
        self.transform = np.vstack((self.transform, np.array([[0, 0, 1]])))
        return self.transform

    def rotate(self, theta):
        """

        :param theta: rotation mesaured in radians

        """
        self.theta += theta
        self.rotation = np.array([
            [ math.cos(self.theta), -1*math.sin(self.theta) ],
            [ math.sin(self.theta), math.cos(self.theta) ]
        ])

    def transpose(self, transposition):
        """

        :param transposition: displacement cartesian vector

        """
        self.transposition += transposition

    def scale(self, scale):
        """

        :param scale: scaling factor

        """
        self.scale_factor *= scale

    def get_matrix(self):
        """ """
        return self.transform

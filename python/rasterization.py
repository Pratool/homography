#!/usr/bin/env python3
from transformation import multiply
from transformation import TransformGenerator
import math
import numpy as np
import sys

class Rasterizer:
    def __init__(self, image_matrix, transformation=None,
                 transformation_matrix=None, canvas_dimensions=None,
                 canvas_location=(0,0), background=None):
        if transformation is not None and transformation_matrix is None:
            self.inverse_transformation_matrix = \
                    np.linalg.inv(transformation.build_matrix())

        elif transformation is None and transformation_matrix is not None:
            self.inverse_transformation_matrix = np.linalg.inv(
                                                    transformation_matrix
                                                )

        else:
            raise Exception("Need to provide one of transformation input class "
                            "or a transformation matrix.")

        self.output = background
        if canvas_dimensions is None and self.output is None:
            self.canvas_dimensions = image_matrix[:,:,0].shape
            self.output = np.zeros(shape = image_matrix.shape)
        elif canvas_dimensions is None and self.output is not None:
            max_y, max_x, max_layer = self.output.shape
            self.canvas_dimensions = (max_x, max_y, max_layer)
        elif canvas_dimensions is not None and self.output is None:
            self.canvas_dimensions = canvas_dimensions
            self.output = np.zeros(
                shape=(
                    self.canvas_dimensions[0],
                    self.canvas_dimensions[1],
                    3
                )
            )
        elif canvas_dimensions is not None and self.output is not None:
            raise RuntimeError("Scaling background image by linearly "
                               "interpolating to specified canvas is "
                               "unsupported.")
        else:
            raise Exception("Unknown combination of canvas_dimensions and "
                            "output parameters")

        self.canvas_location = canvas_location
        self.image_matrix = image_matrix
        self.export = None

    def rasterize(self):
        im = self.image_matrix
        original_max_y, original_max_x = self.image_matrix[:,:,0].shape

        transform_matrix = self.inverse_transformation_matrix

        for i in range(self.canvas_location[0], self.canvas_dimensions[0]):
            for j in range(self.canvas_location[1], self.canvas_dimensions[1]):
                vec = np.array([j,i]).reshape(2,1)
                vec_in_original = multiply(vec, transform_matrix)

                if vec_in_original[0]+1 > original_max_y or vec_in_original[1]+1 > original_max_x:
                    continue
                if vec_in_original[0]-1 < 0 or vec_in_original[1]-1 < 0:
                    continue

                h = math.floor(vec_in_original[1])
                k = math.floor(vec_in_original[0])
                delta_x = vec_in_original[1] - h
                delta_y = vec_in_original[0] - k

                interpolate = lambda l: _linear_interpolate(delta_x,
                                                        delta_y,
                                                        im[k][h][l],
                                                        im[k][h+1][l],
                                                        im[k+1][h][l],
                                                        im[k+1][h+1][l])
                self.output[j][i][0] = interpolate(0)
                self.output[j][i][1] = interpolate(1)
                self.output[j][i][2] = interpolate(2)

        self.export = self.output
        return self.export

def _linear_interpolate(delta_x, delta_y, top_left, top_right, bottom_left,
                       bottom_right):
    return delta_x*delta_y * (
                top_left + bottom_right - top_right - bottom_left
            ) + (
                delta_y * (bottom_left - top_left)
            ) + (
                delta_x * (top_right - top_left)
            ) + top_left

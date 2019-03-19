#!/usr/bin/env python3
from transformation import multiply
from transformation import TransformGenerator
import math
import numpy as np
import sys

class Rasterizer:
    def __init__(self, image_matrix, transformation=None,
                 transformation_matrix=None):
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

        self.image_matrix = image_matrix
        self.export = None

    def rasterize(self):
        im = self.image_matrix
        original_max_x, original_max_y = self.image_matrix[:,:,0].shape

        transform_matrix = self.inverse_transformation_matrix

        output_image = np.zeros(shape=(original_max_x, original_max_y, 3))

        for i in range(0, output_image[:,:,0].shape[0]):
            for j in range(0, output_image[:,:,0].shape[1]):
                vec = np.array([i,j]).reshape(2,1)
                vec_in_original = multiply(vec, transform_matrix)

                if vec_in_original[0]+1 > original_max_x or vec_in_original[1]+1 > original_max_y:
                    continue
                if vec_in_original[0]-1 < 0 or vec_in_original[1]-1 < 0:
                    continue

                h = math.floor(vec_in_original[0])
                k = math.floor(vec_in_original[1])
                delta_x = vec_in_original[0] - h
                delta_y = vec_in_original[1] - k

                interpolate = lambda l: _linear_interpolate(delta_x,
                                                        delta_y,
                                                        im[h][k][l],
                                                        im[h+1][k][l],
                                                        im[h][k+1][l],
                                                        im[h+1][k+1][l])
                output_image[i][j][0] = interpolate(0)
                output_image[i][j][1] = interpolate(1)
                output_image[i][j][2] = interpolate(2)

        self.export = output_image
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

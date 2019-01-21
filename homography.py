#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import sys

sns.set(
    context="paper",
    font_scale=1.3,
    font="serif",
    palette="bright",
    style="whitegrid"
)

def main():
    """ """
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
    else:
        image_path = "./media/t2.png"

    im = mpimg.imread(image_path)
    im0 = im[:,:,0]
    #im_shape = im0.shape
    #im0 = im0.ravel()

    transform = TransformGenerator(
        theta = math.pi/3.0,
        transposition = np.asarray(
                            [math.sqrt(68)/2, math.sqrt(68)/2]
                        ).reshape(2,1),
        scale = 2.0
    )

    transform_matrix = transform.build_matrix()
    tmp_im = np.array([[0, 0]])

    # NOTE: loop is extremely slow because adding to canvas on each iteration
    for j in range(111, 211):
        for i in range(420, 450):
            vec = np.array([i,j]).reshape(2,1)
            transformed_vec = multiply(vec, transform_matrix)
            tmp_im = np.vstack((tmp_im, transformed_vec.reshape(1,2)))

            # for each transformed point, colorize marker according to
            # original image
            plt.scatter(
                transformed_vec[0], transformed_vec[1],
                marker='o', s=5**2,
                color=(im[i][j][0], im[i][j][1], im[i][j][2])
            )

    # quick way to determine the shape of transformed image
    #plt.scatter(tmp_im[1:,0], tmp_im[1:,1], marker='o', s=5**2, color='black')

    # show all color layers
    plt.imshow(im[:,:,0])
    plt.imshow(im[:,:,1])
    plt.imshow(im[:,:,2])
    #plt.imshow(im0.reshape(im_shape))

    plt.show()

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
            [ math.cos(theta), -1*math.sin(theta) ],
            [ math.sin(theta), math.cos(theta) ]
        ])
        return self.build_matrix()

    def transpose(self, transposition):
        """

        :param transposition: displacement cartesian vector

        """
        self.transposition += transposition
        return self.build_matrix()

    def scale(self, scale):
        """

        :param scale: scaling factor

        """
        self.scale_factor *= scale
        return self.build_matrix()

    def get_matrix(self):
        """ """
        return self.transform

def plot2dvec(vec):
    """

    :param vec: two-element, column-vector

    """
    plt.scatter(vec[:,0], vec[:,1], marker='o', s=10**2)

def transform2d_test():
    """ """
    vec = np.asarray([3,5]).reshape(2,1)
    plot2dvec(vec.reshape(1,2))

    transform = TransformGenerator(
        transposition = np.asarray(
                            [math.sqrt(68)/2,math.sqrt(68)/2]
                        ).reshape(2,1),
        scale = 0.25
    )

    for i in range(10,360,10):
        transform.theta = i*180.0/math.pi
        transform_matrix = transform.build_matrix()
        rvec = multiply(
            vec,
            TransformGenerator(theta = transform.theta).get_matrix()
        )
        plot2dvec(rvec.reshape(1,2))
        tvec = multiply(vec, transform_matrix)
        plot2dvec(tvec.reshape(1,2))

    plt.show()

if __name__ == "__main__":
    main()

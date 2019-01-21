#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
from transformation import multiply
from transformation import TransformGenerator
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

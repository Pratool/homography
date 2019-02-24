#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
from rasterization import Rasterizer
from transformation import multiply
from transformation import TransformGenerator
import math
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

def main():
    """ """
    if len(sys.argv) == 4:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        scale = float(sys.argv[3])
    else:
        input_image_path = "./media/t2.png"
        output_image_path = "output.png"
        scale = 300;
    # 825 by 562
    homography_matrix(
        input_image_path,
        output_image_path,
        [
            (
                (198, 52),
                (0, 0)
            ),
            (
                (799, 87),
                (825, 0)
            ),
            (
                (739, 523),
                (825, 562)
            ),
            (
                (34, 448),
                (0, 562)
            )
        ],
        scale
    )

def homography_matrix(input_image_path, output_image_path,
                      corresponding_points, scale):
    equation_matrix = []
    for point_pair in corresponding_points:
        original_point, transformed_point = point_pair
        equation_matrix.append([
            -1.*original_point[0],
            -1.*original_point[1],
            -1.,
            0.,
            0.,
            0.,
            original_point[0]*transformed_point[0],
            original_point[1]*transformed_point[0],
            transformed_point[0]
        ])
        equation_matrix.append([
            0.,
            0.,
            0.,
            -1.*original_point[0],
            -1.*original_point[1],
            -1.,
            original_point[0]*transformed_point[1],
            original_point[1]*transformed_point[1],
            transformed_point[1]
        ])
    equation_matrix = np.asarray(equation_matrix)
    u, s, vh = np.linalg.svd(equation_matrix)
    transform_matrix = vh[8][:].reshape((3,3))
    transform_matrix *= scale;

    im = mpimg.imread(input_image_path)
    image_rasterization = Rasterizer(
        im,
        transformation_matrix = transform_matrix
    )
    matplotlib.image.imsave(
        output_image_path,
        image_rasterization.rasterize()
    )


def reverse_transformation(input_image_path, output_image_path, rotation):
    im = mpimg.imread(input_image_path)

    transform = TransformGenerator(
        scale = 1.0,
        theta = rotation*math.pi/180.0,
    )

    image_rasterization = Rasterizer(im, transformation = transform)

    matplotlib.image.imsave(output_image_path, image_rasterization.rasterize())

def forward_transformation(input_image_path, output_image_path):
    im = mpimg.imread(input_image_path)
    im0 = im[:,:,0]

    transform = TransformGenerator(
        theta = math.pi/3.0,
        transposition = np.asarray(
                            [math.sqrt(68)/2, math.sqrt(68)/2]
                        ).reshape(2,1),
        scale = 1.0
    )

    transform_matrix = transform.build_matrix()

    output_image = np.zeros(shape=(1000,1000,3))
    for i in range(0, im0.shape[0]):
        for j in range(0, im0.shape[1]):
            vec = np.array([i,j]).reshape(2,1)
            transformed_vec = multiply(vec, transform_matrix)
            x = math.floor(transformed_vec[0]+0.5)
            y = math.floor(transformed_vec[1]+0.5)
            output_image[x][y][0] = im[i][j][0]
            output_image[x][y][1] = im[i][j][1]
            output_image[x][y][2] = im[i][j][2]

    matplotlib.image.imsave(output_image_path, output_image)

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

#!/usr/bin/env python3
"""
projective-plot.py

Meant to demonstrate 3d vector plotting and manipulation. Primary purpose is to
improve projective geometry understanding.
"""
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

sns.set(
    context="paper",
    style="whitegrid",
    palette="bright",
    font_scale=1.3,
    font="serif"
)

def main():
    """ """
    im = mpimg.imread("t2.png")
    im0 = im[:,:,0]
    mat = np.identity(im0.shape[0])
    im0 = im0.unravel()
    mat[0]
    print(im.shape)
    imgplot = plt.imshow(im[:,:,0])
    plt.show()

def rotate2d(vector, theta):
    """

    :param vector: two-element, column-vector
    :param theta: theta in radians

    """
    matrix = np.asarray([
        [ math.cos(theta), -1*math.sin(theta)],
        [ math.sin(theta), math.cos(theta)]
    ])
    return matrix@vector

def plot2dvec(vector):
    """

    :param vector: two-element, column-vector

    """
    plot_vecs = np.vstack((np.zeros((1,2)), vector.reshape(1,2)))
    plt.plot(plot_vecs[:,0], plot_vecs[:,1])

def plot3dvecs(vectors):
    """

    :param vectors: three-element, column-vector

    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    for vector in vectors:
        plot_vecs = np.vstack((np.zeros((1,3)), vector.reshape(1,3)))
        ax.plot(plot_vecs[:,0], plot_vecs[:,1], plot_vecs[:,2])
    ax.view_init(elev=20., azim=-35)

def rotate2d_test():
    """ """
    vector = np.asarray([3,5]).reshape(2,1)
    plot2dvec(vector)
    for i in range(10):
        theta = random.random()*math.pi*2.0
        rvec = rotate2d(vector, theta)

        print("{deg:.2f} degree rotation on {vec} = {rvec}".format(
            deg = 180.0/math.pi*theta,
            vec = vector.reshape(1,2),
            rvec = rvec.reshape(1,2)
        ))
        plot2dvec(rvec)
    plt.show()

def dot_test():
    """ """
    vec1 = np.asarray([3,5]).reshape(1,2)
    vec2 = np.asarray([2,1]).reshape(2,1)
    dot = vec1@vec2
    print("{v1} dot {v2} = {v3}".format(
        v1 = vec1,
        v2 = vec2.reshape(1,2),
        v3 = dot
    ))

    plot2dvec(vec1.reshape(2,1))
    plot2dvec(vec2.reshape(2,1))
    plt.show()

def cross_test():
    """ """
    vec1 = np.asarray([3,5,0]).reshape(1,3)
    vec2 = np.asarray([2,1,0]).reshape(1,3)
    cross = np.cross(vec1, vec2)
    print("{v1} x {v2} = {v3}".format(
        v1 = vec1,
        v2 = vec2,
        v3 = cross
    ))

    plot3dvecs([
        vec1.reshape(3,1),
        vec2.reshape(3,1),
        cross.reshape(3,1)
    ])
    plt.show()

if __name__ == "__main__":
    cross_test()

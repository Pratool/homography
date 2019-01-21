from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

sns.set(
    context="paper",
    font_scale=1.3,
    font="serif",
    palette="bright",
    style="whitegrid"
)

def main():
    """ """
    im = mpimg.imread("./media/t2.png")
    im0 = im[:,:,0]
    #im_shape = im0.shape
    #im0 = im0.ravel()

    trans = np.asarray([math.sqrt(68)/2,math.sqrt(68)/2]).reshape(2,1)
    scale = 2
    theta = math.pi/3.0
    theta = 0
    tmp_im = np.array([[0, 0]])

    # NOTE: loop is extremely slow because adding to canvas on each iteration
    for j in range(111, 211):
        for i in range(420, 450):
            vec = np.array([i,j]).reshape(2,1)
            tvec = similarity(vec, theta, trans, scale)
            tmp_im = np.vstack((tmp_im, tvec.reshape(1,2)))

            # for each transformed point, colorize marker according to
            # original image
            plt.scatter(
                tvec[0], tvec[1],
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

def rotate2d(vec, theta):
    """

    :param vec: two-element, column-vector
    :param theta: theta in radians

    """
    matrix = np.asarray([
        [ math.cos(theta), -1*math.sin(theta)],
        [ math.sin(theta), math.cos(theta)]
    ])
    return matrix@vec

def euclidean(vec, theta, trans):
    """

    :param vec: two-element, column-vector
    :param theta: theta in radians
    :param trans: two-element, column-vector

    """
    matrix = np.asarray([
        [ math.cos(theta), -1*math.sin(theta), trans[0][0] ],
        [ math.sin(theta), math.cos(theta), trans[1][0] ],
        [ 0.0, 0.0, 1.0 ]
    ])
    aug_vec = np.asarray([
        vec[0],
        vec[1],
        np.asarray([[1]])
    ])
    outvec = matrix@aug_vec
    return np.asarray([outvec[0],outvec[1]])

def similarity(vec, theta, trans, scale):
    """

    :param vec: two-element, column-vector
    :param theta: theta in radians
    :param trans: two-element, column-vector
    :param scale: 

    """
    matrix = np.array([
        [ math.cos(theta), -1*math.sin(theta) ],
        [ math.sin(theta), math.cos(theta) ]
    ])
    matrix = scale*matrix
    matrix = np.hstack((matrix, trans))
    matrix = np.vstack((matrix, np.array([[0, 0, 1]])))
    aug_vec = np.vstack((vec, np.array([[1]])))
    outvec = matrix@aug_vec
    return outvec[:2]

def plot2dvec(vec):
    """

    :param vec: two-element, column-vector

    """
    plt.scatter(vec[:,0], vec[:,1], marker='o', s=10**2)

def transform2d_test():
    """ """
    vec = np.asarray([3,5]).reshape(2,1)
    plot2dvec(vec.reshape(1,2))
    trans = np.asarray([math.sqrt(68)/2,math.sqrt(68)/2]).reshape(2,1)
    scale = 0.25

    for i in range(10,360,10):
        theta = i*180.0/math.pi
        rvec = rotate2d(vec, theta)
        plot2dvec(rvec.reshape(1,2))
        tvec = similarity(vec, theta, trans, scale)
        plot2dvec(tvec.reshape(1,2))

    plt.show()

if __name__ == "__main__":
    main()

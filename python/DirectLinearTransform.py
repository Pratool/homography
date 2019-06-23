#!/usr/bin/env python3
import logging
import numpy as np

def main():
    logging.basicConfig(
        level = logging.DEBUG
    )
    testComputeTransform()

def computeTransform(correspondingPoints):
    """ Solves series of equations that relates inputs to outputs by a single
    3x3 matrix.

    :param correspondingPoints: list of four pairs of corresponding 2d points
    :returns transfomMatrix: 3x3 homography matrix that transforms corresponding
                             points

    """
    P = []
    for (x,y),(u,v) in correspondingPoints:
        logging.debug("correspondence: ({x},{y}) -> ({u},{v})".format(
            x = x,
            y = y,
            u = u,
            v = v
        ))
        P.extend([
            [-x, -y, -1] + [0]*3 + [x*u, y*u, u],
            [0]*3 + [-x, -y, -1] + [x*v, y*v, v]
        ])
    u, s, vh = np.linalg.svd(P)
    transformMatrix = vh[8][:].reshape((3,3))
    return transformMatrix

def testComputeTransform():
    """ Tests computeTransform with a series of points with known inputs and
    outputs.
    """
    tm = computeTransform(
        [ ((198, 52), (0, 0)),
          ((799, 87), (825, 0)),
          ((739, 523), (825, 562)),
          ((34, 448), (0, 562)) ]
    )

    test_vecs = []
    test_vecs.append(np.asarray([[198], [52], [1]]))
    test_vecs.append(np.asarray([[799], [87], [1]]))
    test_vecs.append(np.asarray([[0], [0], [1]]))
    test_vecs.append(np.asarray([[825.], [0], [1]]))
    test_vecs.append(np.asarray([[825.], [562], [1]]))

    for v in test_vecs:
        print("input vector")
        print(v)
        print()

        u = tm@v
        print("transformed vector")
        print(u)
        print()

        w = u[2][0]
        u = u / w
        print("normalized vector")
        print(u)
        print()
        print("-----------------")

if __name__ == "__main__":
    main()

import numpy as np

def main():
    test_get_p()

def get_p(inputs, outputs):
    P = []
    print([ i for i in zip(inputs,outputs)])
    for (x,y),(u,v) in zip(inputs,outputs):
        P.extend([
            [-x, -y, -1] + [0]*3 + [x*u, y*u, u],
            [0]*3 + [-x, -y, -1] + [x*v, y*v, v]
        ])
    u, s, vh = np.linalg.svd(P)
    transform_matrix = vh[8][:].reshape((3,3))
    return transform_matrix

def test_get_p():
    tm = get_p(
        [ (198, 52), (799, 87), (739, 523), (34, 448) ],
        [ (0, 0), (825, 0), (825, 562), (0, 562) ]
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
    test_get_p()

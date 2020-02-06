#!/usr/bin/env python3
# Poorly documented and unmaintained code to prototype convex polygon
# intersection algorithm.

import sys

from functools import reduce
from random import random

import numpy as np

from matplotlib import pyplot as plt

def main():
    quad0 = Polygon( [(5,14), (9,8), (5,3), (1,7)] )
    quad1 = Polygon( [(9,14), (13,9), (9,4), (6,9)] )

    poly0 = Polygon( [(18,76), (45, 91), (91,69), (43, 13), (21, 23)] )
    poly1 = Polygon( [(76,92), (88,74), (89, 57), (45, 9), (36, 80)] )

    run_and_plot(poly1, poly0)

def run_and_plot(poly0, poly1):
    ax = plt.gca()
    poly0.plot(ax)
    poly1.plot(ax)
    intersect_polys(poly0, poly1).plot(ax)
    plt.show()

def gen_random_poly(number_of_vertices):
    # Note that this result will not be convex.
    return Polygon([ (int(random()*100),int(random()*100)) for i in range(number_of_vertices) ])

class Vertex:
    def __init__(self, x, y, prev=None, next=None):
        self.x = x
        self.y = y
        self.coord = (x,y)
        self.prev = prev
        self.next = next

    def __init__(self, coord, prev=None, next=None):
        self.x = coord[0]
        self.y = coord[1]
        self.coord = coord
        self.prev = prev
        self.next = next

    def __repr__(self):
        return '{}->{}->{}'.format(
            str(self.prev), str(self), str(self.next))

    def __str__(self):
        return str(self.coord)

class Polygon:
    def __init__(self, vertices):
        self.vertices = []

        for v in vertices:
            self.vertices.append(Vertex(v))

        for i in range(len(self.vertices)):
            self.vertices[i-1].next = self.vertices[i]
            self.vertices[i].prev = self.vertices[i-1]

    def __repr__(self):
        return '->'.join([str(v) for v in self.vertices])

    def plot(self, mplib_axes):
        xs = [v.x for v in self.vertices] + [self.vertices[0].x]
        ys = [v.y for v in self.vertices] + [self.vertices[0].y]
        mplib_axes.plot(xs, ys, 'o-')
        return mplib_axes

def line_intersect(p0, p1, p2, p3):
    lhs = np.array([
        [p0.y-p1.y, p1.x-p0.x],
        [p2.y-p3.y, p3.x-p2.x]
    ])
    rhs = np.array([
        p0.y * (p1.x-p0.x) - p0.x * (p1.y-p0.y),
        p2.y * (p3.x-p2.x) - p2.x * (p3.y-p2.y)
    ])

    try:
        res = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None

    if not np.allclose(np.dot(lhs, res), rhs):
        return None

    x = res[0]
    y = res[1]

    err = 1e-9

    if x >= min(p0.x, p1.x)-err and \
            x <= max(p0.x, p1.x)+err and \
            y >= min(p0.y, p1.y)-err and \
            y <= max(p0.y, p1.y)+err and \
            x >= min(p2.x, p3.x)-err and \
            x <= max(p2.x, p3.x)+err and \
            y >= min(p2.y, p3.y)-err and \
            y <= max(p2.y, p3.y)+err:
        return (x,y)

    return None


pt_eq = lambda pt0, pt1: (abs(pt0.x-pt1.x) < 1e-9) and (abs(pt0.y-pt1.y) < 1e-9)
or_op = lambda x, y: x or y
and_op = lambda x, y: x and y
pt_vertices = lambda pt, vertices: reduce(or_op, [pt_eq(pt, Vertex(v)) for v in vertices]) if len(vertices) > 0 else False
is_left = lambda pt, v0, v1: ((v0.x-pt.x)*(v1.y-pt.y) - (v0.y-pt.y)*(v1.x-pt.x)) < 0
is_inside = lambda pt, poly: reduce(and_op, [is_left(pt, poly.vertices[i-1], poly.vertices[i]) for i in range(len(poly.vertices))])

def intersect_polys(poly0, poly1):
    inter = []

    poly0v = poly0.vertices[0]
    poly1v = poly1.vertices[0]

    max_iters = 2*(len(poly0.vertices)+len(poly1.vertices))
    c = 0

    while True:
        if is_inside(poly0v, poly1):
            if pt_vertices(poly0v, inter):
                break
            inter.append(poly0v.coord)

        tmp_inters = []
        for p1v in poly1.vertices:
            tmp_inter = line_intersect(poly0v, poly0v.next, p1v, p1v.next)
            if tmp_inter is not None:
                tmp_inters.append((tmp_inter[0], tmp_inter[1], p1v.next))

        # sort by euclidean distance from poly0v
        tmp_inters = sorted(tmp_inters, key=lambda x: (poly0v.x-x[0])**2 + (poly0v.y-x[1])**2)

        broke = False
        for i in tmp_inters:
            if pt_vertices(Vertex((i[0], i[1])), inter):
                broke = True
                break
            inter.append((i[0], i[1]))
        if broke:
            break

        if len(tmp_inters) != 0:
            poly1v_after = tmp_inters[-1][2]
            if is_inside(poly1v_after, poly0):
                # swap
                poly0, poly1 = poly1, poly0
                poly0v, poly1v = poly1v_after, poly0v.next
            else:
                poly0v = poly0v.next

        if len(tmp_inters) == 0:
            poly0v = poly0v.next

        c+=1
        if c > max_iters:
            raise RuntimeError('You found a bug!')

    return Polygon(inter)

if __name__ == '__main__':
    main()

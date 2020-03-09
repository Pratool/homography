#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation(s) of Concave Hull algorithm(s).
"""

import json
import math
import unittest

from matplotlib import pyplot as plt

from Polygon import *

def main():
    with open('concaveSet00.json', 'r') as f:
        concave_set = json.load(f)
    concave_set = [tuple(point) for point in concave_set]

    concave_poly = create_concave_poly(concave_set, 5)

    axes = plt.gca()
    _plot_set(axes, concave_set)
    concave_poly.plot(axes)
    plt.show()


# pt is the point to test, v0, v1 are endpoints of the line segment.
signed_area = lambda pt, v0, v1: ((v0[0] - pt[0]) * (v1[1] - pt[1]) -
                                  (v0[1] - pt[1]) * (v1[0] - pt[0]))


def _plot_set(mplib_axes, point_set):
    xs = [p[0] for p in point_set]
    ys = [p[1] for p in point_set]
    mplib_axes.plot(xs, ys, 'o')
    return mplib_axes


class Subpoint:
    def __init__(self, point, distance):
        self.point = point
        self.distance = distance


    def __repr__(self):
        return (self.distance, self.point).__repr__()


def create_concave_poly(point_set, k):
    # Sort 2D points by y-value
    point_set.sort(key=lambda x: x[1])

    knn_graph = create_knn_graph(point_set, k)

    vertices = []
    current_point = point_set[0]
    prev_point = (current_point[0]+1, current_point[1])
    while len(vertices) <= 1 or vertices[0] != current_point:
        linked_points = knn_graph[current_point]
        rightmost_neighbor = None
        for neighbor in linked_points:
            if neighbor.point in vertices and neighbor.point != vertices[0]:
                continue

            if rightmost_neighbor is None:
                rightmost_neighbor = neighbor.point

            if signed_area(neighbor.point, current_point, rightmost_neighbor) < 0:
                rightmost_neighbor = neighbor.point

        vertices.append(current_point)
        if rightmost_neighbor is None:
            break
        prev_point = current_point
        current_point = rightmost_neighbor

    return Polygon(vertices)


def create_knn_graph(point_set, k):
    if k <= 0 or type(k) != int:
        raise ValueError('k-NN graph requires positive integer k.')
    if k >= len(point_set) - 1:
        raise ValueError('Max k in k-NN graph is one fewer than the total '
                         'number of nodes.')

    graph = {}
    for point in point_set:
        # Order all k linked points by ascending euclidean distance from point.
        graph[point] = []
        for linked_point in point_set:
            distance = euclidean_distance(point, linked_point)
            for i in range(k):
                if (len(graph[point]) < i + 1 or
                    distance < graph[point][i].distance):
                    graph[point].insert(i, Subpoint(linked_point, distance))
                    break
        graph[point] = graph[point][1:k+1]
    return graph


def euclidean_distance(point0, point1):
    if len(point0) != len(point1):
        raise ValueError('Points must be the same dimensionality.')
    accumulator = 0
    for i in range(len(point0)):
        accumulator += (point0[i]-point1[i])**2
    return math.pow(accumulator, float(1.0/len(point0)))


class TestEuclideanDistance(unittest.TestCase):
    def test_2d(self):
        self.assertAlmostEqual(
            euclidean_distance((3, 0), (0, 4)), float(5), delta=1e-9)

    def test_3d(self):
        self.assertAlmostEqual(
            euclidean_distance((1, 2, 3), (4, 5, 6)), float(3), delta=1e-9)


if __name__ == '__main__':
    main()

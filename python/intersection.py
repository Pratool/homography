#!/usr/bin/env python3
# Poorly documented and unmaintained code to prototype convex polygon
# intersection algorithm.

import math
import sys

from datetime import datetime as dt
from functools import reduce
from random import random

import numpy as np

from matplotlib import pyplot as plt

from Polygon import *


def main():
    quad0 = ConvexPolygon([(5, 14), (9, 8), (5, 3), (1, 7)])
    quad1 = ConvexPolygon([(9, 14), (13, 9), (9, 4), (6, 9)])

    poly0 = ConvexPolygon([(18, 76), (45, 91), (91, 69), (43, 13), (21, 23)])
    poly1 = ConvexPolygon([(76, 92), (88, 74), (89, 57), (45, 9), (36, 80)])

    run_and_plot(quad0, quad1)
    run_and_plot(poly0, poly1)

    for i in range(10):
        run_and_plot(gen_random_convex_poly(11), gen_random_convex_poly(10))

    run_and_plot(gen_random_convex_poly(111), gen_random_convex_poly(120))


def find_mismatches():
    quad0 = ConvexPolygon([(5, 14), (9, 8), (5, 3), (1, 7)])
    mismatches = []
    # Compare straight-forward O(n) implementation with O(lg(n)) implementation.
    grid_size = 10
    for x in range(0, 15 * grid_size, 1):
        for y in range(0, 15 * grid_size, 1):
            pt = Vertex((x / float(grid_size), y / float(grid_size)))
            res = is_inside(pt, quad0)
            res2 = is_inside_n2(pt, quad0)
            if res != res2:
                mismatches.append((pt, res, res2))
    print(len(mismatches))
    for m in mismatches[:10]:
        print(m)
        res = is_inside(m[0], quad0, log=True)
        axes = plt.gca()
        quad0.plot(axes)
        plt.plot(m[0].x(), m[0].y(), 'o')
        plt.show()


def run_and_plot(poly0, poly1):
    ax = plt.gca()
    poly0.plot(ax)
    poly1.plot(ax)
    intersect_polys(poly0, poly1, is_inside).plot(ax)
    plt.show()

    ax = plt.gca()
    poly0.plot(ax)
    poly1.plot(ax)
    intersect_polys(poly0, poly1).plot(ax)
    plt.show()


def gen_random_convex_poly(number_of_vertices,
                           radius=50.0,
                           x_min=0,
                           x_max=100,
                           y_min=0,
                           y_max=100):
    """Not truly random. Mostly an observation that yields convex polygons.
    """
    offsets = [random() * 2.0 * math.pi for i in range(number_of_vertices)]
    offsets.sort()
    offsets = offsets[::-1]
    x_offset = x_min + (x_max - x_min) * random()
    y_offset = y_min + (y_max - y_min) * random()

    return ConvexPolygon([(radius * math.cos(theta) + x_offset,
                           radius * math.sin(theta) + y_offset)
                          for theta in offsets])


# BEGINNING EXTREMELY TERSE FUNCTIONS
def line_intersect(p0, p1, p2, p3, err=0):
    lhs = np.array([[p0.y() - p1.y(), p1.x() - p0.x()],
                    [p2.y() - p3.y(), p3.x() - p2.x()]])
    rhs = np.array([
        p0.y() * (p1.x() - p0.x()) - p0.x() * (p1.y() - p0.y()),
        p2.y() * (p3.x() - p2.x()) - p2.x() * (p3.y() - p2.y())
    ])

    try:
        res = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None

    if not np.allclose(np.dot(lhs, res), rhs):
        return None

    x = res[0]
    y = res[1]

    if x >= min(p0.x(), p1.x())-err and \
            x <= max(p0.x(), p1.x())+err and \
            y >= min(p0.y(), p1.y())-err and \
            y <= max(p0.y(), p1.y())+err and \
            x >= min(p2.x(), p3.x())-err and \
            x <= max(p2.x(), p3.x())+err and \
            y >= min(p2.y(), p3.y())-err and \
            y <= max(p2.y(), p3.y())+err:
        return (x, y)

    return None


pt_eq = lambda pt0, pt1: (abs(pt0.x() - pt1.x()) < 1e-9) and (abs(pt0.y() - pt1.
                                                                  y()) < 1e-9)
or_op = lambda x, y: x or y
and_op = lambda x, y: x and y
pt_vertices = lambda pt, vertices: reduce(
    or_op, [pt_eq(pt, Vertex(v))
            for v in vertices]) if len(vertices) > 0 else False
# pt is the point to test, v0, v1 are endpoints of the line segment.
signed_area = lambda pt, v0, v1: ((v0.x() - pt.x()) * (v1.y() - pt.y()) -
                                  (v0.y() - pt.y()) * (v1.x() - pt.x()))
is_left = lambda pt, v0, v1: signed_area(pt, v0, v1) <= 0
is_inside_n2 = lambda pt, poly: reduce(and_op, [
    is_left(pt, poly.vertices[i - 1], poly.vertices[i])
    for i in range(len(poly.vertices))
])

# END EXTREMELY TERSE FUNCTIONS


def is_inside(point, polygon, log=False):
    """Determines if a point lies inside a convex polygon.

    Args:
        point: A Vertex object representing 2D point to query polygon for
            inclusion truth.
        polygon: A Polygon object which provides bounds to which insidedness is
            concerned.

    Returns:
        Boolean value where True indicates the point is inside the polygon.
    """
    top = polygon.get_top_vertex()
    bottom = polygon.get_bottom_vertex()

    # If the point's y-value is not within the y-value bounds of the polygon
    if point.y() > top.get_value().y() or point.y() < bottom.get_value().y():
        return False

    # Condition input before next stage.
    offset = -bottom.index
    start_bottom = bottom.index + offset
    if top.index + offset < 0:
        end_top = len(
            polygon.vertices) - (-(top.index + offset) % len(polygon.vertices))
    else:
        end_top = top.index + offset
    comparator = lambda v0, v1: v1.y() > v0.y()
    offset = -offset

    # Find polygon line segment that intersects a horizontal line that passes
    # through the point from the left side of the polygon.
    lhs_tail = get_segment_tail(point, polygon, comparator, start_bottom,
                                end_top, offset)

    # Condition input before next stage.
    offset = -top.index
    start_top = top.index + offset
    if bottom.index + offset < 0:
        end_bottom = len(polygon.vertices) - (-(bottom.index + offset) %
                                              len(polygon.vertices))
    else:
        end_bottom = bottom.index + offset
    comparator = lambda v0, v1: v1.y() < v0.y()
    offset = -offset

    # Find polygon line segment that intersects a horizontal line that passes
    # through the point from the right side of the polygon.
    rhs_tail = get_segment_tail(point, polygon, comparator, start_top,
                                end_bottom, offset)

    return (signed_area(point, lhs_tail.next_vertex(), lhs_tail) >= 0 and
            is_left(point, rhs_tail, rhs_tail.next_vertex()))


def get_segment_tail(point, polygon, comparator, start, end, offset):
    """Assumes that start indicates an index value of 0 at the beginning of
    recursion. At no point shall the value of end be less than the value of
    start. Indices refer to the index within the polygon vertex list. Search
    cases may not neatly have the start index as 0, in which case an offset is
    added to bring both start and end indices to the intended indices in the
    polygon vertex list.
    """
    # Permit access to the vertex data.
    start_iter = polygon.get_iterator_wrapped_in_bounds(start + offset)
    end_iter = polygon.get_iterator_wrapped_in_bounds(end + offset)

    if start == end:
        return start_iter.get_value()

    if end - start == 1:
        if comparator(start_iter.get_value(), end_iter.get_value()):
            return start_iter.get_value()
        return end_iter.get_value()

    mid = start + int((end - start) / 2.0)
    mid_point = polygon.get_iterator_wrapped_in_bounds(mid + offset).get_value()
    prev_point = mid_point.prev_vertex()
    next_point = mid_point.next_vertex()

    if comparator(prev_point, point) and comparator(point, mid_point):
        return prev_point
    if comparator(mid_point, point):
        return get_segment_tail(point, polygon, comparator, mid, end, offset)
    return get_segment_tail(point, polygon, comparator, start, mid - 1, offset)


# This function has many bugs. Use with caution: does not work as advertised!
def intersect_polys(poly0, poly1, is_inside=is_inside_n2):
    inter = []

    poly0v = poly0.vertices[0]
    poly1v = poly1.vertices[0]

    # Prevent faulty logic from causing an infinite loop.
    max_iters = 2 * (len(poly0.vertices) + len(poly1.vertices))
    c = 0

    while True:
        if is_inside(poly0v, poly1):
            if pt_vertices(poly0v, inter):
                break
            inter.append((poly0v.x(), poly0v.y()))

        tmp_inters = []
        for p1v in poly1.vertices:
            tmp_inter = line_intersect(poly0v, poly0v.next_vertex(), p1v,
                                       p1v.next_vertex())
            if tmp_inter is not None:
                tmp_inters.append(
                    (tmp_inter[0], tmp_inter[1], p1v.next_vertex()))

        # sort by euclidean distance from poly0v
        tmp_inters = sorted(tmp_inters,
                            key=lambda x: (poly0v.x() - x[0])**2 +
                            (poly0v.y() - x[1])**2)

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
                poly0v, poly1v = poly1v_after, poly0v.next_vertex()
            else:
                poly0v = poly0v.next_vertex()

        if len(tmp_inters) == 0:
            poly0v = poly0v.next_vertex()

        c += 1
        if c > max_iters:
            raise RuntimeError('You found a bug!')

    return Polygon(inter)


if __name__ == '__main__':
    main()

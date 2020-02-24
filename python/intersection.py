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


def main():
    quad0 = ConvexPolygon([(5, 14), (9, 8), (5, 3), (1, 7)])
    quad1 = ConvexPolygon([(9, 14), (13, 9), (9, 4), (6, 9)])

    poly0 = ConvexPolygon([(18, 76), (45, 91), (91, 69), (43, 13), (21, 23)])
    poly1 = ConvexPolygon([(76, 92), (88, 74), (89, 57), (45, 9), (36, 80)])

    lgn_avgs = []
    min_input_size = 5
    max_input_size = 50000

    size_range = range(min_input_size, max_input_size, 500)
    for input_size in size_range:
        lgn = []
        pt = Vertex((50.1, 50.5))
        #pt = Vertex((101.3, 50.5))

        for iters in range(5):
            #tmp_poly0 = gen_random_convex_poly(input_size, x_min=25, x_max=75)
            tmp_poly0 = gen_random_convex_poly(input_size)
            try:
                start = dt.utcnow()
                is_inside(pt, tmp_poly0)
                end = dt.utcnow()
                lgn.append(end-start)
                lgn[-1] = lgn[-1].seconds + lgn[-1].microseconds*1e-6
            except IndexError:
                print('error at {} vertices'.format(input_size))
            """
            if lgn[-1] > 0.0002:
                ax = plt.gca()
                tmp_poly0.plot(ax)
                ax.plot(pt.x(), pt.y(), 'o')
                plt.show()
            """

            # is_inside_n2() is too slow to keep up with trying to sample
            # is_inside() at larger input sizes.
            #start = dt.utcnow()
            #is_inside_n2(pt, tmp_poly0)
            #end = dt.utcnow()
            #n2.append(end-start)
            #n2[-1] = n2[-1].seconds + n2[-1].microseconds*1e-6
        try:
            lgn_avgs.append(sum(lgn)/float(len(lgn)))
        except Exception:
            lgn_avgs.append(0)

    ax = plt.gca()
    tmp_poly0.plot(ax)
    ax.plot(pt.x(), pt.y(), 'o')
    plt.show()

    xs = [x for x in size_range]
    plt.plot(xs, [5.5e-5 / 16.0 * math.log2(x) for x in xs])
    plt.plot(xs, lgn_avgs)
    plt.show()

    mismatches = []
    # Compare straight-forward O(n) implementation with O(lg(n)) implementation.
    grid_size = 10
    for x in range(0, 15*grid_size, 1):
        for y in range(0, 15*grid_size, 1):
            pt = Vertex((x/float(grid_size), y/float(grid_size)))
            res = is_inside(pt, quad0)
            res2 = is_inside_n2(pt, quad0)
            if res != res2:
                mismatches.append((pt, res, res2))
    print(len(mismatches))

    #run_and_plot(quad0, quad1)
    #run_and_plot(poly0, poly1)



def run_and_plot(poly0, poly1):
    ax = plt.gca()
    poly0.plot(ax)
    poly1.plot(ax)
    intersect_polys(poly0, poly1, is_inside).plot(ax)
    plt.show()


def gen_random_convex_poly(number_of_vertices, radius=50.0, x_min=0, x_max=100, y_min=0, y_max=100):
    offsets = sorted([random() * 2.0 * math.pi for i in range(number_of_vertices)])
    x_offset = x_min + (x_max-x_min)*random()
    y_offset = y_min + (y_max-y_min)*random()

    return ConvexPolygon([(radius*math.cos(theta) + x_offset, radius*math.sin(theta) + y_offset) for theta in offsets])


def gen_random_poly(number_of_vertices):
    # Note that this result is not necessarily convex.
    return Polygon([(int(random() * 100), int(random() * 100))
                    for i in range(number_of_vertices)])


class Vertex:

    def __init__(self, coordinates, **kwargs):
        self.coordinates = coordinates
        if 'prev' in kwargs:
            self.set_prev(kwargs['prev'])
        if 'next' in kwargs:
            self.set_next(kwargs['next'])

    def x(self):
        return self.coordinates[0]

    def y(self):
        return self.coordinates[1]

    def set_next(self, vertex):
        if type(vertex) != type(self):
            raise TypeError('Cannot point to a non-Vertex type.')
        self.next = vertex

    def next_vertex(self):
        """Return the next clockwise vertex in its corresponding
        polygon.
        """
        return self.next

    def set_prev(self, vertex):
        if type(vertex) != type(self):
            raise TypeError('Cannot point to a non-Vertex type.')
        self.prev = vertex

    def prev_vertex(self):
        """Return the previous clockwise vertex in its corresponding
        polygon.
        """
        return self.prev

    def __repr__(self):
        return '<Vertex {}>'.format(str(self))

    def __str__(self):
        return str(self.coordinates)


class PolygonIter:

    def __init__(self, polygon_ref, index):
        self.polygon = polygon_ref
        self.direction = 1
        self.index = 0
        self.index = self._advance(index)

    def reverse_direction(self):
        self.direction *= -1

    def get_value(self):
        return self.polygon.get_vertex_at(self.index)

    def get_index(self):
        return self.index

    def get_next_iterator(self):
        next_iter = PolygonIter(self.polygon, self.index+self.direction)
        next_iter.direction = self.direction
        return next_iter

    def __iter__(self):
        return self

    def _advance(self, skips):
        return_index = self.index + skips
        if (return_index >= len(self.polygon.vertices) or
            return_index <= -1):
            print(self.index, skips, return_index)
            #ax = plt.gca()
            #self.polygon.plot(ax)
            #plt.show()
            raise IndexError('Cannot advance beyond bounds of corresponding '
                             'polygon vertices container.')
        return return_index

    def __next__(self):
        next_iter = PolygonIter(self.polygon, self.index)
        next_iter.direction = self.direction
        self.index = self._advance(self.direction)
        return next_iter

    def __sub__(self, rhs):
        return self.index - rhs.index

    def __add__(self, rhs):
        return self.index + rhs.index

    def __eq__(self, rhs):
        return self.index == rhs.index

    def __repr__(self):
        return str(self.polygon.get_vertex_at(self.index))

    def __str__(self):
        return str(self.polygon.get_vertex_at(self.index))


class ConvexPolygon:
    """This class encapsulates properties convex polygons posess in the
    most efficient runtime complexity possible. Class methods should perform
    actions in-place."""

    def __init__(self, vertices):
        """Vertices are connected by line segments in clockwise order to
        represent a convex polygon.
        """
        self.vertices = []

        for v in vertices:
            self.vertices.append(Vertex(v))

        for i in range(len(self.vertices)):
            self.vertices[i - 1].set_next(self.vertices[i])
            self.vertices[i].set_prev(self.vertices[i - 1])

        self.c = 0

    def __repr__(self):
        return '->'.join([str(v) for v in self.vertices])

    def get_iterator(self, index=0):
        return PolygonIter(self, index)

    def get_iterator_wrapped_in_bounds(self, index):
        if index <= -1:
            iter_index = -index
            iter_index %= len(self.vertices)
            return self.get_iterator(index=len(self.vertices)-iter_index)
        return self.get_iterator(index=index % len(self.vertices))

    def get_front_iterator(self):
        return self.get_iterator(index=0)

    def get_back_iterator(self):
        return self.get_iterator(index=len(self.vertices)-1)

    def get_vertex_at(self, index):
        return self.vertices[index]

    def __iter__(self):
        return PolygonIter(self, 0)

    def __reversed__(self):
        iterator = PolygonIter(self, 0)
        iterator.reverse_direction()
        return iterator

    def plot(self, mplib_axes):
        xs = [v.x() for v in self.vertices] + [self.vertices[0].x()]
        ys = [v.y() for v in self.vertices] + [self.vertices[0].y()]
        mplib_axes.plot(xs, ys, 'o-')
        return mplib_axes

    def _search_vertex(self, comparator, start=0, end=None):
        """Retrieve vertex to exploit convexity within the polygon's vertex
        search space. Will use start and end as index values and search in the
        interval [start, end].

        Args:
            self: Polygon in question.
            comparator: Function that compares the middle vertex with its
                neighbors. Must return a boolean value.
            start: Index into vertex array to begin search.
            end: Index into vertex array to end search.

        Returns:
            Polygon iterator to vertex within vertex array.
        """
        if end is None:
            end = start + len(self.vertices) - 1

        if start == end:
            return self.get_iterator(index=start)

        if start - end == 1:
            if comparator(self.vertices[start], self.vertices[end]):
                return self.get_iterator(index=start)
            else:
                return self.get_iterator(index=end)

        mid = start + int((end - start) / 2.0)
        mid_vtx = self.vertices[mid]
        after_mid_vtx = self.vertices[mid + 1]
        before_mid_vtx = self.vertices[mid - 1]

        if comparator(after_mid_vtx, mid_vtx):
            return self._search_vertex(comparator, start=mid + 1, end=end)
        elif comparator(before_mid_vtx, mid_vtx):
            return self._search_vertex(comparator, start=start, end=mid - 1)
        else:
            return self.get_iterator(index=mid)

    def get_top_vertex(self):
        """Retrieve vertex with the max y-value of all vertices in polygon.
        Will use start and end as index values and search in the interval
        [start, end].

        Args:
            self: Polygon in question.
            start: Index into vertex array to begin search.
            end: Index into vertex array to end search.

        Returns:
            Polygon iterator to vertex within vertex array.
        """
        return self._search_vertex(lambda v0, v1: v0.y() > v1.y())

    def get_bottom_vertex(self):
        """Retrieve vertex with the min y-value of all vertices in polygon.
        Will use start and end as index values and search in the interval
        [start, end].

        Args:
            self: Polygon in question.
            start: Index into vertex array to begin search.
            end: Index into vertex array to end search.

        Returns:
            Polygon iterator to vertex within vertex array.
        """
        return self._search_vertex(lambda v0, v1: v0.y() < v1.y())

    def get_right_side(self):
        bottom = self.get_bottom_vertex()
        tmp = self.get_top_vertex()
        side = []
        while tmp != bottom:
            side.append(tmp)
            tmp = tmp.next
        side.append(tmp)
        return side

    def get_left_side(self):
        top = self.get_top_vertex()
        tmp = self.get_bottom_vertex()
        side = []
        while tmp != top:
            side.append(tmp)
            tmp = tmp.next
        side.append(tmp)
        return side

# BEGINNING EXTREMELY TERSE FUNCTIONS
def line_intersect(p0, p1, p2, p3, err=0):
    lhs = np.array([[p0.y() - p1.y(), p1.x() - p0.x()], [p2.y() - p3.y(), p3.x() - p2.x()]])
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


pt_eq = lambda pt0, pt1: (abs(pt0.x() - pt1.x()) < 1e-9) and (abs(pt0.y() - pt1.y()) <
                                                          1e-9)
or_op = lambda x, y: x or y
and_op = lambda x, y: x and y
pt_vertices = lambda pt, vertices: reduce(
    or_op, [pt_eq(pt, Vertex(v))
            for v in vertices]) if len(vertices) > 0 else False
is_left = lambda pt, v0, v1: ((v0.x() - pt.x()) * (v1.y() - pt.y()) - (v0.y() - pt.y()) *
                              (v1.x() - pt.x())) <= 0
is_inside_n2 = lambda pt, poly: reduce(and_op, [
    is_left(pt, poly.vertices[i - 1], poly.vertices[i])
    for i in range(len(poly.vertices))
])


def is_inside(point, polygon):
    """Determines if a point lies inside a convex polygon.

    Args:
        point: A Vertex object representing 2D point to query polygon for
            inclusion truth.
        polygon: A Polygon object which provides bounds to which insidedness is
            concerned.

    Returns:
        Boolean value where True indicates the point is inside the polygon.
    """
    # perform input conditioning here.
    top = polygon.get_top_vertex()
    bottom = polygon.get_bottom_vertex()

    if point.y() > top.get_value().y() or point.y() < bottom.get_value().y():
        return False

    # Vertex at tail of left polyline's line segment where vertical bounds
    # encompass point.
    left = _is_inside(point, polygon, bottom, top, lambda u, v: u.y() >= v.y()).get_value()

    # Vertex at tail of right polyline's line segment where vertical bounds
    # encompass point.
    right = _is_inside(point, polygon, top, bottom, lambda u, v: u.y() <= v.y()).get_value()

    return (is_left(point, left, left.next_vertex()) and
            is_left(point, right, right.next_vertex()))


def _is_inside(point, polygon, start_iter, end_iter, comparator):
        if start_iter == end_iter:
            return start_iter
        if start_iter - end_iter == 1:
            if comparator(start_iter.get_value(), end.get_value()):
                return start_iter
            return end_iter

        mid = start_iter + polygon.get_iterator_wrapped_in_bounds( int((end_iter - start_iter) / 2.0) )
        mid_iter = polygon.get_iterator(mid)
        after_mid_iter = polygon.get_iterator_wrapped_in_bounds(mid + 1)
        before_mid_iter = polygon.get_iterator_wrapped_in_bounds(mid - 1)

        if comparator(after_mid_iter.get_value(), point) and not comparator(mid_iter.get_value(), point):
            return mid_iter
        elif comparator(mid_iter.get_value(), point):
            return _is_inside(point, polygon, start_iter, before_mid_iter, comparator)
        else:
            return _is_inside(point, polygon, after_mid_iter, end_iter, comparator)


def intersect_polys(poly0, poly1, is_inside=is_inside_n2):
    inter = []

    poly0v = poly0.vertices[0]
    poly1v = poly1.vertices[0]

    max_iters = 2 * (len(poly0.vertices) + len(poly1.vertices))
    c = 0

    while True:
        if is_inside(poly0v, poly1):
            if pt_vertices(poly0v, inter):
                break
            inter.append(poly0v.coord)

        tmp_inters = []
        for p1v in poly1.vertices:
            tmp_inter = line_intersect(poly0v, poly0v.next_vertex(), p1v, p1v.next_vertex())
            if tmp_inter is not None:
                tmp_inters.append((tmp_inter[0], tmp_inter[1], p1v.next_vertex()))

        # sort by euclidean distance from poly0v
        tmp_inters = sorted(tmp_inters,
                            key=lambda x: (poly0v.x - x[0])**2 +
                            (poly0v.y - x[1])**2)

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

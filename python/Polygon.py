#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defines classes related to 2D polygons.
"""


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
        return '({:.2f}, {:.2f})'.format(*self.coordinates)


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
        next_iter = PolygonIter(self.polygon, self.index + self.direction)
        next_iter.direction = self.direction
        return next_iter

    def __iter__(self):
        return self

    def _advance(self, skips):
        return_index = self.index + skips
        if (return_index >= len(self.polygon.vertices) or return_index <= -1):
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


class Polygon:

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

    def __repr__(self):
        return '->'.join([str(v) for v in self.vertices])

    def get_iterator(self, index=0):
        return PolygonIter(self, index)

    def get_iterator_wrapped_in_bounds(self, index):
        if index <= -1:
            iter_index = -index
            iter_index %= len(self.vertices)
            return self.get_iterator(index=len(self.vertices) - iter_index)
        return self.get_iterator(index=index % len(self.vertices))

    def get_front_iterator(self):
        return self.get_iterator(index=0)

    def get_back_iterator(self):
        return self.get_iterator(index=len(self.vertices) - 1)

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
        """Retrieve a vertex that is the "most" of a property that can be
        extracted per vertex with the comparator. Will use start and end as
        index values and search in the interval [start, end].

        Args:
            self: Polygon in question.
            comparator: Function that compares two vertices to create a strict
                        lexical ordering. Must return a boolean value.
            start: Index into vertex array to begin search.
            end: Index into vertex array to end search.

        Returns:
            Polygon iterator to vertex within vertex array.
        """
        if end is None:
            end = start + len(self.vertices) - 1

        largest_lexical_value = self.get_iterator(index=start)

        for index in range(start, end):
            if comparator(self.vertices[index], largest_lexical_value.get_value()):
                largest_lexical_value = self.get_iterator(index=index)

        return largest_lexical_value

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


class ConvexPolygon(Polygon):
    """This class encapsulates properties convex polygons posess in the
    most efficient runtime complexity possible. Class methods should perform
    actions in-place."""

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

# distutils: language = c++
# cython: linetrace=True

import cython
cimport cython
import numpy as np
cimport numpy as np
from libcpp.map cimport map
from libcpp.deque cimport deque
from cython.operator import dereference, postincrement
from libcpp.vector cimport vector


cdef class DiGraph:
    cdef int n_edges
    cdef map[int, map[int, int]] edges

    def __init__(self):
        self.n_edges = 0
        # self.edges

    cpdef int number_of_nodes(self):
        return self.edges.size()

    cpdef int number_of_edges(self):
        return self.n_edges

    cpdef int add_nodes(self, int n):
        cdef int i
        for i in range(self.number_of_nodes(),
                       self.number_of_nodes() + n):
            self.edges[i] = map[int, int]()
        return self.number_of_nodes()

    cpdef int add_edge(self, int u, int v):
        self.edges[u][v] = self.n_edges
        self.n_edges += 1
        return self.n_edges

    cpdef int get_eid(self, int u, int v):
        return self.edges[u][v]

    cpdef map[int, int] successors(self, int u):
        return self.edges[u]

    cpdef out_edges(self, int u):
        cdef map[int, int].iterator it = self.edges[u].begin()
        while it != self.edges[u].end():
            print(dereference(it).first)
            print(dereference(it).second)
            postincrement(it)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef single_source_shortest_path(self, weight, int s):
        cdef double[:] w = weight
        cdef int parent, child, eid
        cdef double cost_via_parent
        cdef map[int, int].iterator child_iter, last_child
        cdef deque[int] frontier
        frontier.push_front(s)

        cdef np.ndarray[double, ndim=1] d = np.empty(self.number_of_nodes())
        d[:] = np.inf
        d[s] = 0

        while frontier.size() != 0:
            parent = frontier.front()
            frontier.pop_front()
            child_iter = self.edges[parent].begin()
            last_child = self.edges[parent].end()
            while child_iter != last_child:
                child = dereference(child_iter).first
                eid = dereference(child_iter).second
                cost_via_parent = d[parent] + w[eid]
                if cost_via_parent < d[child]:
                    d[child] = cost_via_parent
                    if (frontier.size() > 0
                            and cost_via_parent <= d[frontier.front()]):
                        frontier.push_front(child)
                    else:
                        frontier.push_back(child)
                postincrement(child_iter)
        return d
















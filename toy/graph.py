import torch
import torch.nn as nn

from device import device

class Graph:
    def __init__(self, n_vertex, n_edge_type):
        # A[e][j][i] = vertex i and j are linked via edge type e
        # triplets[i] = (v_from, e_type, v_to)
        self.A = torch.zeros((n_edge_type, n_vertex, n_vertex)).to(device)

        self.n_vertex = n_vertex
        self.n_edge_type = n_edge_type
        self.masked_edge = torch.zeros(size=self.A.shape).to(device)

    def tosparse(self):
        self.A = self.A.to_sparse()

    def _check_v(self, v):
        if (v < 0):
            return False
        if (v >= self.n_vertex):
            return False
        assert(type(v) == int)
        return True

    def _check_e(self, e_type):
        if (e_type < 0):
            return False
        if (e_type >= self.n_edge_type):
            return False
        return True

    def linked(self, v_from, v_to, e_type):
        return self.A[e_type][v_to][v_from] == 1.0

    def add_edge(self, v_from, v_to, e_type):
        assert(self._check_v(v_from)
            and self._check_v(v_to)
            and self._check_e(e_type))
        if (self.A[e_type][v_to][v_from] == 0.0):
            self.A[e_type][v_to][v_from] = 1.0
    
    def remove_edge(self, v_from, v_to, e_type):
        assert(self._check_v(v_from)
            and self._check_v(v_to)
            and self._check_e(e_type))
        assert(self.A[e_type][v_to][v_from] == 1.0)
        self.A[e_type][v_to][v_from] = 0.0
    
    def mask_edge(self, v_from, v_to, e_type):
        self.remove_edge(v_from, v_to, e_type)
        self.masked_edge[e_type][v_to][v_from] = 1.0

    def restore_masked_edge(self):
        self.A += self.masked_edge
        self.masked_edge = torch.zeros(size=self.A.shape).to(device)

    def write_to_file(self, filepath):
        with open(filepath, "w") as f:
            for e in range(self.n_edge_type):
                for i in range(self.n_vertex):
                    for j in range(self.n_vertex):
                        if (self.A[e][j][i] > 0):
                            f.write(str(i) + '\t' + str(e) + '\t' + str(j) + '\n')
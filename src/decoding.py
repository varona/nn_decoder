"""Decoder class: a decoder takes a syndrome and returns a recovery operation.
"""
import numpy as np
from numba import njit


@njit
def _find_index(array, item):
    """Finds first index where array==item. Else returns len(array)."""
    for idx in np.arange(array.size):
        if array[idx] == item:
            return idx
    return idx+1


@njit('int64[:](int64[:], int64[:], int64[:])')
def _winding_number_loop(line_h, line_v, edge):
    """This is equivalent to (but faster):
    ```
    for e in edge:
        if e in line_h:
            winding_num[1] += 1
        if e in line_v:
            winding_num[0] += 1
    ```
    """
    winding_num = np.zeros(2, dtype=np.int64)
    index = np.searchsorted(line_h, edge)
    i = _find_index(index, len(line_h))
    winding_num[1] = np.sum(line_h[index[:i]] == edge[:i])
    index = np.searchsorted(line_v, edge)
    i = _find_index(index, len(line_v))
    winding_num[0] = np.sum(line_v[index[:i]] == edge[:i])
    return winding_num


class Decoder:
    """A decoder receives an error syndrome and computes a recovery operation
    that tries to correct the error.
    """

    def __init__(self, decode_fun):
        self._decode_fun = decode_fun

    def decode(self, v_syn_ind, p_syn_ind, lattice):
        """Compute recovery operation.

        Args:
            v_syn_ind (1d array int): excited vertex indices.
            p_syn_ind (1d array int): excited plaquette indices.
            lattice (lattice object): the code lattice.

        Returns:
            x_recover (1d array int): edge indices where to apply a S^+.
            z_recover (1d array int): edge indices where to apply a Z-Pauli.
        """
        return self._decode_fun(v_syn_ind, p_syn_ind, lattice)

    def decode_result(self, x_edge, z_edge, v_syn_ind, p_syn_ind, lattice):
        """Logical errors as a consequence of applying recovery operators.

        Args:
            x_edge (1d array int): edges where X (S^+) error occurred.
            z_edge (1d array int): edges where Z error occurred.
            v_syn_ind (1d array int): excited vertex indices.
            p_syn_ind (1d array int): excited plaquette indices.
            lattice (lattice object): the code lattice.

        Returns:
            x_result_0 (1d array bool): True if X logical error in logical
                qubit 0.
            z_result_0 (1d array bool): logical Z error in qubit 0.
            x_result_1 (1d array bool): logical X error in qubit 1.
            z_result_1 (1d array bool): logical Z error in qubit 1.
        """
        x_recover, z_recover = self._decode_fun(v_syn_ind,
                                                p_syn_ind,
                                                lattice)
        x_total = self.operation_total(x_edge, x_recover)
        z_total = self.operation_total(z_edge, z_recover)

        x_result_0, x_result_1 = self._winding_result(
            x_total, lattice, direct=True)
        z_result_0, z_result_1 = self._winding_result(
            z_total, lattice, direct=False)

        return x_result_0, z_result_0, x_result_1, z_result_1

    def decode_succesful(self, x_edge, z_edge, v_syn_ind, p_syn_ind, lattice):
        """If no logical error occurred after recovery, returns True.
        """
        x_result_0, x_result_1, z_result_0, z_result_1 = self.decode_result(
            x_edge, z_edge, v_syn_ind, p_syn_ind, lattice)
        x_result = np.logical_or(x_result_0, x_result_1)
        z_result = np.logical_or(z_result_0, z_result_1)
        return not np.logical_or(x_result, z_result)

    @staticmethod
    def operation_total(e_edge, e_recover):
        """Join e_edge and e_recover, if edge appears an even number of times,
        delete from final list e_total.
        """
        e_total = []
        for i in range(len(e_edge)):
            e, count = np.unique(
                np.append(e_edge[i], e_recover), return_counts=True)
            e_total.append(e[count % 2 == 1])
        return e_total

    @staticmethod
    def _winding_number(edge, lattice, direct):
        """Given a cycle in direct/reciprocal lattice (if direct==True the 
        cycle is in direct lattice) as a list of edges, obtain its winding 
        number, i.e., number of times the cycle edge goes around the  crossing 
        a system gauge line."""
        line_h = lattice.gauge_line(not direct, 0)
        line_v = lattice.gauge_line(not direct, 1)
        return _winding_number_loop(line_h, line_v, np.sort(edge))

    def _winding_result(self, e_total, lattice, direct):
        """Obtains parity of the winding number for each of the cycles in
        e_total. If parity is even, return False; else return True."""
        e_result_0 = np.zeros(len(e_total), dtype=bool)
        e_result_1 = np.zeros(len(e_total), dtype=bool)
        for i, e in enumerate(e_total):
            if len(e) == 0:
                continue
            winding = self._winding_number(e, lattice, direct)
            if winding[0] % 2 == 1:
                e_result_0[i] = True
            if winding[1] % 2 == 1:
                e_result_1[i] = True
        return e_result_0, e_result_1


def decoder_simple_fun(v_syn_ind, p_syn_ind, lattice):
    """Decoder taking all excitations to plaquette and vertex 0.
    """
    # Decode vertex syndrome
    x_recover = np.array([], dtype=int)
    for s in v_syn_ind:
        x_recover = np.append(
            x_recover, lattice.direct_distance(0, s, pbc=False)[1])
    # Decode plaquette syndrome
    z_recover = np.array([], dtype=int)
    for s in p_syn_ind:
        z_recover = np.append(
            z_recover, lattice.reciprocal_distance(0, s, pbc=False)[1])
    # Simplify operators
    x_recover, count = np.unique(x_recover, return_counts=True)
    x_recover = x_recover[count % 2 == 1]
    z_recover, count = np.unique(z_recover, return_counts=True)
    z_recover = z_recover[count % 2 == 1]
    return x_recover, z_recover


decoder_simple = Decoder(decoder_simple_fun)

"""
Introduces errors into the lattice and computes syndromes.
"""

import os
import numpy as np
from bisect import bisect_left

import compute_pattern as cp
from hexagonal_lattice import HexagonalLattice


def plaquette_measurement(X_string, lattice):
    """Given a string of X errors, returns the possible plaquette excitations
    obtained when plaquettes are measured, with the probability distribution
    associated with them. 
    
    Also returns P_z operators that may cause these excitations (Pauli P_z
    coefficients are assigned to a plaquette excitation configuration, if they
     are non-zero).
    
    Args:
        X_string (1d array int): adjacent edge indices.
        lattice (lattice object): lattice where X_string lives.
    
    Returns:
        prob_dist (1d array float): probability of each of the plaquette 
            excitation patterns.
        plaquette (1d array int): plaquettes of the lattice involved.
        c_class (1d array int): plaquette excitation pattern to which each
            coefficient c(Z_Q) belongs, number of coefficients is
            2**(len(edge)).
        edge (1d array int): edges involved.
    """
    pattern = cp.X_error_pattern(X_string, lattice)
    # Label edges and plaquettes in conn path from left to right and bottom to
    # top. If group of plaquettes/edges goes through a border, the ordering 
    # will not work, thus we use move_away_boundary
    vertex = np.unique(lattice.e_vertex[X_string])

    plaquette = np.unique(lattice.v_plaquette[vertex])
    p_position = cp.move_away_boundary_p(
        lattice.N_row, 
        lattice.N_col, 
        np.atleast_2d(lattice.p_position[plaquette, :]))
    ind = np.lexsort((p_position[:,1], p_position[:,0]))
    plaquette = plaquette[ind]

    edge = lattice.vertex2edge(vertex, both=False)
    e_position = cp.move_away_boundary_e(
        lattice.N_row, lattice.N_col, lattice.edge_position(edge))
    ind = np.lexsort((e_position[:,1], e_position[:,0]))
    edge = edge[ind]

    pattern = cp.pattern2dec(pattern, cp.set_list, cp.base, add_len=True)
    i = bisect_left(cp.pattern_dec_list, pattern)
    if cp.pattern_dec_list[i] == pattern:
        return (cp.pattern_pdata_list[i], plaquette, cp.pattern_cdata_list[i], 
                edge)

    # Compute pattern
    if len(edge)>12:
        raise Exception(f'Pattern too long{pattern}, length {len(edge)}')
    prob_dist, c_class = cp.compute_pattern(pattern, 14, save=True)
    return prob_dist, plaquette, c_class, edge


def get_X_p_syndrome(X_string, lattice, one_string=True, KTC=False):
    """Computes plaquette syndrome coming from a string of X operators. 
    
    Edges in X_string must be contiguous. Syndrome is measured and one of the
    possible plaquette excitation pattern is chosen with np.random.choice. 
    Returns just one P_z if one_string is True.
    
    Args:
        X_string (1d array int): adjacent edge indices.
        lattice (lattice object).
        one_string (bool): optional, return just one of the equivalent Z's.
        KTC (bool): optional, Kitaev toric code instead of semion code.

    Returns:
        p_syndrome (1d array bool): with length lattice.N_plaquette. True 
            corresponds to a plaquette excitation.
        Z_operator (list of 1d arrays int): each 1d array containing edge
            indices where a Z error occurred.
    """
    p_syndrome = np.zeros(lattice.N_plaquette, dtype=bool)

    # Toric code, no plaquette syndrome cause by X errors
    if KTC:
        return p_syndrome, np.array([], dtype=object)

    # Semion code
    prob_dist, plaquette, c_class, edge = plaquette_measurement(
        X_string, lattice)
    case = np.random.choice(len(prob_dist), 1, p=prob_dist)
    p_syndrome_list = cp.de2bi(np.arange(2**len(plaquette)), len(plaquette))
    p_index = np.argwhere(np.squeeze(p_syndrome_list[case,:]))
    p_syndrome[plaquette[p_index]] = True

    # Case 0 corresponds to no plaquette excitations
    # Not interested in closed loops P_z, only open P_z.
    if case==0: return p_syndrome, np.array([], dtype=object) 
    c_choice_ind = np.argwhere(c_class == case)
    Z_operator = []
    for i in range(len(c_choice_ind)):
        ind = np.squeeze(np.asarray(
            cp.de2bi(c_choice_ind[i], len(edge)), bool))
        Z_operator.append(edge[ind])
        if one_string: break

    return p_syndrome, Z_operator


def get_Z_p_syndrome(Z_error, lattice):
    """Computes plaquette syndrome coming from Z errors.

    Args:
        Z_error (1d array bool): with length lattice.N_edge. True if an Z error
            occurred at that edge.
        lattice (lattice object).

    Returns:
        p_syndrome (1d array bool): with length lattice.N_plaquette. True 
            corresponds to a plaquette excitation.
        Z_edge (1d array int): edge indices where an Z error occurred.
    """
    p_syndrome = np.zeros(lattice.N_plaquette, dtype=bool)
    for i in range(lattice.N_edge):
        if Z_error[i]:
            p_index = np.intersect1d(
                lattice.v_plaquette[lattice.e_vertex[i, 0]], 
                lattice.v_plaquette[lattice.e_vertex[i, 1]])
            p_syndrome[p_index] = ~p_syndrome[p_index]
    return p_syndrome, np.squeeze(np.argwhere(Z_error))


def get_v_syndrome(X_error, lattice):
    """Returns vertex syndrome coming from X errors.
    
    Args:
        X_error (1d array bool): with length lattice.N_edge. True if an X error
            occurred at that edge.
        lattice (lattice object).

    Returns:
        v_syndrome (1d array bool) with length lattice.N_vertex. True 
            corresponds to a vertex excitation.
        X_edge (1d array int): edge indices where an X error occurred.
    """
    v_syndrome = np.zeros(lattice.N_vertex, dtype=bool)
    for i in range(lattice.N_edge):
        if X_error[i]:
            v_syndrome[lattice.e_vertex[i]] = np.logical_not(v_syndrome[lattice.e_vertex[i]])
    return v_syndrome, np.squeeze(np.argwhere(X_error))


def uncompress_Z_edge(Z_edge_compressed, one_string=True):
    """Obtain the Z_edge errors causing the observed plaquette excitations from
    the 'compressed' version of Z_edge.
    """

    if one_string:
        string = np.array([], int)
        for s in Z_edge_compressed:
            if len(s):
                string = np.append(string, s[0])
        string, count = np.unique(string, return_counts=True)
        string = string[count % 2 != 0]
        return [string]

    dim = [len(i) for i in Z_edge_compressed]
    Z_edge = []
    if len(Z_edge_compressed):
        for i in range(np.prod(dim)):
            ind = np.unravel_index(i,dim)
            string = np.array([], int)
            for j, d in enumerate(ind):
                string = np.append(string, Z_edge_compressed[j][d])
            string, count = np.unique(string, return_counts=True)
            string = string[count % 2 != 0]
            Z_edge.append(string)
    if Z_edge == []:
        Z_edge = [np.array([],dtype=int)]
    return Z_edge


def XYZ_noise(p_X, p_Y, p_Z, lattice, KTC=False, plot=False, seed=None,
              one_string=True):
    """Introduce random X-, Y- and Z-Pauli errors in a lattice, measure and 
    return syndrome.
    
    Args:
        p_X: float, X-error probability.
        p_Y: float, Y-error probability.
        p_Z: float, Z-error probability.
        lattice (lattice object).
        KTC (bool): whether to simulate Kitaev toric code instead of semion 
            code.
        plot (bool): wheter to plot lattice with error and syndromes.
        seed (int): random state seed.
        one_string (bool): when an X error occurs and the stabilizers are
            measured, several Z strings can produce the same plaquette
            excitations (all of these are equivalent in terms of decoding).
            Return all of them or just one.

    Returns:
        X_edge (list of 1d array int): the list only contains one element, 
            a 1d array with edge indices where an X error occurred.
        Z_edge (list of 1d arrays int): if one_string is False, returns
            several 1d arrays with all possible Z strings, else just one.
        v_syn_ind (1d array int): excited vertex indices.
        p_syn_ind (1d array int): excited plaquette indices.
    """

    if seed is not None:
        np.random.seed(seed)

    error = np.random.choice(4, lattice.N_edge, p=[
                             1-p_X-p_Y-p_Z, p_X, p_Y, p_Z])
    X_error = (error==1)
    Y_error = (error==2)
    Z_error = (error==3)
    # Decompose Y_error into X and Z error
    X_error = np.logical_xor(X_error, Y_error)
    Z_error = np.logical_xor(Z_error, Y_error)

    # Z-error plaquette syndrome
    p_syndrome, Z_edge = get_Z_p_syndrome(Z_error, lattice)
    if Z_edge.size:  # if Z_edge non-empty
        Z_edge = [[Z_edge]]
    else:
        Z_edge = [np.array([], dtype=int)]
    # X-error vertex syndrome
    v_syndrome, X_edge = get_v_syndrome(X_error, lattice)
    X_edge = [X_edge]
    # X-error plaquette syndrome
    X_error_group = cp.group_edges(X_error, lattice)
    for X_string in X_error_group:
        synd, P_z = get_X_p_syndrome(X_string, lattice, one_string, KTC)
        p_syndrome = np.logical_xor(synd, p_syndrome)
        if len(P_z)>0:
            Z_edge.append(P_z)

    Z_edge = uncompress_Z_edge(Z_edge, one_string)

    v_syn_ind = np.squeeze(np.argwhere(v_syndrome))
    p_syn_ind = np.squeeze(np.argwhere(p_syndrome))

    if plot:
        lattice.plot_lattice(e_numbers=True)
        lattice.plot_syndrome(p_syndrome, v_syndrome)
        lattice.plot_error(X_error, Z_error)
        from matplotlib import pyplot as plt
        plt.savefig('XYZ_noise.pdf')

    return X_edge, Z_edge, v_syn_ind, p_syn_ind

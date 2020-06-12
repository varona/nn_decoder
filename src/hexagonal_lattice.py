"""
Builds a hexagonal lattice of size (n_row, n_col) on a torus, periodic boundary
conditions.
"""

import numpy as np
from matplotlib import pyplot as plt
from numba import njit


# # # # Numba optimized functions # # # #

# Numba optimized functions are defined outside the HexagonalLattice class,
# since numba cannot handle self object easily and static methods cannot call
# other static methods.


@njit
def _mod(n_col, n_row, position):
    """Applies _mod_1d at each row of the array position.
    
    Args:
        n_col (int).
        n_row (int).
        position (2d array of int or float): each row contains a position in 
        the lattice.
        
    Returns:
        pos (2d array): same shape and type as position.
    """
    pos = position.copy()
    for i in range(position.shape[0]):
        pos[i, :] = _mod_1d(n_col, n_row, position[i])
    return pos


@njit
def _mod_1d(n_col, n_row, position):
    """mod function for the hexagonal lattice.

    Since we have periodic boundary conditions any position (x,y) can be 
    mapped to a position in the range 
        0 <= y - 3*x < 6*n_col
        0 <= y < 3*n_row
    
    Args:
        n_col (int).
        n_row (int).
        position (1d array of int or float).

    Returns:
        pos (1d array): same shape and type as position.
    """
    v_0 = np.array([2*n_col, 0])
    v_1 = np.array([n_row, 3*n_row])
    pos = position.copy()
    if pos[1] < 0:
        while pos[1] < 0:
            pos = pos + v_1
    elif pos[1] > 3*n_row:
        while pos[1] > 3*n_row:
            pos = pos - v_1
    if pos[1]-3*pos[0] > 0:
        while pos[1]-3*pos[0] > 0:
            pos = pos + v_0
    elif pos[1]-3*pos[0]+3*n_col*2 < 0:
        while pos[1]-3*pos[0]+3*n_col*2 < 0:
            pos = pos - v_0
    return pos


@njit('float64[:,:](int64,int64,int64[:])')
def _edge_position(n_col, n_vertex, edge):
    """Computes the position of the given edges.
    
    Args:
        edge (1d array int): edges.

    Returns:
        e_position (2d array float): with edge positions.
    """
    e_position = np.zeros((edge.size, 2), dtype=np.float64)
    for i, e in enumerate(edge):
        if e < n_vertex:
            n = e//(n_col*2)
            e_position[i, 1] = 3*n+.5
            m = e-n*2*n_col
            e_position[i, 0] = m+n+.5
        elif e >= n_vertex:
            k = e - n_vertex
            n = k//n_col
            e_position[i, 1] = 3*n+2
            m = k-n*n_col
            e_position[i, 0] = 2*m+1+n
    return e_position


@njit('int64[:](int64, int64, int64[:,:], int64[:], int64[:])')
def _vertex2edge(n_col, n_row, v_position, v_orientation, vertex):
    """Given a list of vertices, returns all edges connected to vertices.
    
    Args:
        n_col (int).
        n_row (int).
        v_position (2d array int): all vertex positions (lattice.v_position).
        v_orientation (1d array int): all vertex orientations.
        vertex (1d array int): array of vertices.

    Returns:
        (1d array int): edges connected to vertices in vertex.
    """
    edge = np.zeros(3*vertex.size, dtype=np.int64)
    shift_0 = np.array([[.5, -.5], [0., 1.], [-.5, -.5]], dtype=np.float64)
    shift_1 = np.array([[.5, .5], [0., -1.], [-.5, .5]], dtype=np.float64)
    for i in range(vertex.size):
        v_pos = np.zeros((3, 2), dtype=np.float64)
        v_pos[0, :] = (v_position[vertex[i]]).astype(np.float64)
        v_pos[1, :] = (v_position[vertex[i]]).astype(np.float64)
        v_pos[2, :] = (v_position[vertex[i]]).astype(np.float64)
        if v_orientation[vertex[i]] == 0:
            v_pos = v_pos + shift_0
        else:
            v_pos = v_pos + shift_1
        v_pos = _mod(n_col, n_row, v_pos)
        edge_v = np.zeros(3, dtype=np.int64)
        for j in range(3):
            if v_pos[j, 0] != np.floor(v_pos[j, 0]):  # semiinteger
                n = (v_pos[j, 1] - .5)/3
                m = v_pos[j, 0]-.5 - n
                edge_v[j] = m + n*2*n_col
            else:
                n = (v_pos[j, 1]-2)/3
                m = (v_pos[j, 0]-1-n)/2
                edge_v[j] = m + n*n_col + n_row*n_col*2
        edge[i*3:i*3+3] = edge_v
    return np.unique(edge)


@njit('int64[:](int64, int64, float64[:,:])')
def _e_position2edge(n_col, n_vertex, e_position):
    """Given edge positions, obtain edge indices.
    
    Args:
        n_col (int).
        n_vertex (int).
        e_position (2d array float): position of edges with shape (?, 2).

    Returns:
        edge (1d array int): edge indices.
    """
    n_edge = np.shape(e_position)[0]
    edge = np.zeros(n_edge, dtype=np.int64)
    for i in range(n_edge):
        if e_position[i, 0] != np.floor(e_position[i, 0]):  # semiinteger
            n = (e_position[i, 1] - .5)/3
            m = e_position[i, 0]-.5 - n
            edge[i] = np.int64(m + n*2*n_col)
        else:
            n = (e_position[i, 1]-2)/3
            m = (e_position[i, 0]-1-n)/2
            edge[i] = np.int64(m + n*n_col + n_vertex)
    return edge


@njit('Tuple((int64, int64[:]))(int64, int64, int64, int64[:], int64[:])')
def _direct_distance(n_col, n_row, v1_orientation, p1, p2):
    """Given two vertex positions compute distance and path between them in
    direct lattice. It does not take into account periodic boundary
    conditions in the lattice.
    
    Args:
        n_col (int).
        n_row (int).
        v1_orientation (int): orientation of first vertex (0 or 1).
        p1 (1d array int): position of first vertex.
        p2 (1d array int): position of second vertex.

    Returns:
        d (int): distance.
        e_path (1d array int): edges in the path.
    """
    p_new = p1.copy()
    d = np.int64(0)
    e_path = np.zeros(0, dtype=np.int64)
    while np.any(p_new != p2):
        p_old = p_new.copy()
        diff = np.sign(p2 - p_new)
        if v1_orientation == 0:
            v1_orientation = 1
            if diff[1] == 1:
                p_new[1] = p_new[1] + 2
            elif diff[0] == 1:
                p_new = p_new + \
                    np.array([1, 0], dtype=np.int64) - \
                    np.array([0, 1], dtype=np.int64)
            else:
                p_new = p_new - np.array([1, 1], dtype=np.int64)
        elif v1_orientation == 1:
            v1_orientation = 0
            if diff[1] == -1:
                p_new[1] = p_new[1] - 2
            elif diff[0] == 1:
                p_new = p_new + np.array([1, 1], dtype=np.int64)
            else:
                p_new = p_new - \
                    np.array([1, 0], dtype=np.int64) + \
                    np.array([0, 1], dtype=np.int64)
        d = d + np.int64(1)

        pos = (p_new + p_old)/2.  # pos is float64
        pos = _mod_1d(n_col, n_row, pos)
        edge = _e_position2edge(n_col, n_row*n_col*2, np.atleast_2d(pos))

        e_path = np.concatenate((e_path, edge))

    return d, e_path


@njit('Tuple((int64, int64[:]))(int64, int64, int64[:], int64[:])')
def _reciprocal_distance(n_col, n_row, p1, p2):
    """Given two plaquette positions, compute distance and path joining them
    in reciprocal lattice.

    Args:
        n_col (int).
        n_row (int).
        p1 (1d array int): position of first plaquette.
        p2 (1d array int): position of second plaquette.

    Returns:
        d (int): distance.
        e_path (1d array int): edges in the path.
    """
    p_new = p1.copy()
    d = np.int64(0)
    e_path = np.zeros(0, dtype=np.int64)
    while np.any(p_new != p2):
        p_old = p_new.copy()
        diff = np.sign(p2 - p_new)
        if diff[1] == 0:
            p_new = p_new + \
                np.multiply(diff, np.array([2, 0], dtype=np.int64))
        else:
            if diff[0] == 0:
                diff[0] = 1
            p_new = p_new + \
                np.multiply(diff, np.array([1, 3], dtype=np.int64))

        d = d + np.int64(1)

        pos = (p_new + p_old)/2.
        pos = _mod_1d(n_col, n_row, pos)
        edge = _e_position2edge(n_col, n_row*n_col*2, np.atleast_2d(pos))

        e_path = np.concatenate((e_path, edge))
    return d, e_path


@njit('int64[:,:](int64, int64, int64[:], int64[:])')
def _pbc_position(n_col, n_row, p1, p2):
    """Compute the four periodic positions of p2 with respect to p1.

    Sum lattice vectors v_0 and v_1 to p2 to obtain periodic positions. This
    positions are equivalent to the original one: p2 == _mod(p2_pbc). Necessary
    to compute distances between p1 and p2.

    Args:
        n_col (int).
        n_row (int).
        p1 (1d array int): position in the lattice.
        p2 (1d array int): position in the lattice.

    Returns:
        p2_pbc (2d array int): with shape (4, 2), the four different
            periodic positions of p1.
    """
    v_0 = np.array([2*n_col, 0])
    v_1 = np.array([n_row, 3*n_row])
    u_1 = np.array([1, 3])/np.sqrt(1+3**2)

    d1 = p2 - p1
    d2 = np.zeros(np.int(2), dtype=np.float64)
    d2[0] = np.float(d1[0])
    d2[1] = np.float(d1[1])
    d2[1] = np.dot(u_1, d2)
    d1[0] = np.int(np.sign(d2[0]))
    d1[1] = np.int(np.sign(d2[1]))

    p2_pbc = np.atleast_2d(p2)
    p2_pbc = np.concatenate((p2_pbc, np.atleast_2d(p2-d1[0]*v_0)), axis=0)
    p2_pbc = np.concatenate((p2_pbc, np.atleast_2d(p2-d1[1]*v_1)), axis=0)
    p2_pbc = np.concatenate(
        (p2_pbc, np.atleast_2d(p2-d1[1]*v_1-d1[0]*v_0)), axis=0)

    return p2_pbc


# # # # Hexagonal Lattice class # # # #

class HexagonalLattice:
    """A hexagonal graph formed by n_row x n_col hexagons on a torus.

    The lattice has n_row x n_col hexagons and has periodic boundary
    conditions, i.e., it's embedded on a torus.

    Attributes:
        n_row (int): number of hexagon rows.
        n_col (int): number of hexagon columns.
        n_edge (int): total number of edges.
        n_vertex (int): total number of vertices.
        n_plaquette (int): total number of plaquettes.
        p_position (2d array int): contains position of each plaquette.
        v_position (2d array int): contains position of each vertex.
        v_orientation (1d array int): contains orientation of each vertex.
        v_plaquette (2d array int): plaquettes connected to each vertex.
        e_vertex (2d array int): vertices connected to each edge.
        e_orientation (1d array int): edge orientation.
    """

    # Height, width and scale of each hexagon. These are only relevant when
    # plotting the lattice.
    size = 1.
    h = 2*size 
    w = np.sqrt(3)*size

    def __init__(self, n_row, n_col):
        """"Creates a HexagonalLattice object.

        Args:
            n_row (int): number of hexagon rows.
            n_col (int): number of hexagon columns.
        """

        self.n_row = n_row
        self.n_col = n_col
        self.n_edge = n_row*n_col*3
        self.n_vertex = n_row*n_col*2
        self.n_plaquette = n_row*n_col

        self.p_position = self.build_plaquette()
        self.v_position, self.v_orientation = self.build_vertex()
        self.v_plaquette = self.vertex2plaquette()
        self.e_vertex, self.e_orientation = self.build_edge()

    def build_plaquette(self):
        """Computes plaquette data.

        Returns:
            p_position (2d array int): contains position of each plaquette.
        """
        # Positions in units x,y -> w/2, h/4
        p_position = np.zeros([self.n_plaquette, 2], dtype=int)
        for i in range(self.n_row):  # iterate over rows
            for j in range(self.n_col):  # iterate over columns
                k = i*self.n_col + j
                p_position[k, 0] = 2 + i + 2*j
                p_position[k, 1] = 2 + i*3
        return p_position

    def build_vertex(self):
        """Construct vertex data.

        Returns:
            v_position (2d array int): contains position of each vertex.
            v_orientation (1d array int): contains orientation of each vertex.
        """
        v_orientation = np.zeros([self.n_vertex], dtype=int)  # Y orientation 1
        v_position = np.zeros([self.n_vertex, 2], dtype=int)
        for i in range(self.n_row):  # iterate over rows
            for j in range(2*self.n_col):  # iterate over columns
                k = i*2*self.n_col+j
                v_position[k, 0] = j + i
                v_position[k, 1] = 1 + i*3
                if j % 2 == 0:
                    v_position[k, 1] = i*3
                    v_orientation[k] = 1
                else:
                    v_position[k, 1] = i*3 + 1
                    v_orientation[k] = 0
        return v_position, v_orientation

    def build_edge(self):
        """Constructs edge data.
        
        Returns:
            e_vertex (2d array int): vertices connected to each edge.
            e_orientation (1d array int): edge orientation.
        """
        e_vertex = np.zeros([self.n_edge, 2], dtype=int)
        e_orientation = np.zeros([self.n_edge], dtype=int)
        for i in range(self.n_vertex):
            e_vertex[i, 0] = i
            if np.mod(i, self.n_col*2) == 2*self.n_col-1:
                e_vertex[i, 1] = i-2*self.n_col+1
            else:
                e_vertex[i, 1] = i+1
            if i % 2 == 0:
                e_orientation[i] = 1
            else:
                e_orientation[i] = 2
        for i in range(self.n_vertex, self.n_edge):
            j = i-self.n_vertex
            n_row = j//self.n_col
            if n_row == self.n_row-1:
                e_vertex[i, 0] = 2*j + 1
                e_vertex[i, 1] = 2*(j-self.n_col*(self.n_row-1))
            else:
                e_vertex[i, 0] = 2*j + 1
                e_vertex[i, 1] = 2*j + 2*self.n_col

        return e_vertex, e_orientation

    def vertex2plaquette(self):
        """Computes the plaquettes associated to each vertex.

        Returns:
            (2d array int): plaquettes connected to each vertex.
        """
        v_plaquette = np.zeros([self.n_vertex, 3], dtype=int)
        for k in range(self.n_vertex):
            p_pos = np.zeros([3, 2], dtype=int)
            if self.v_orientation[k] == 0:
                p_pos[0, ] = self.v_position[k, ] + np.array([1, 1])
                p_pos[1, ] = self.v_position[k, ] + np.array([-1, 1])
                p_pos[2, ] = self.v_position[k, ] + np.array([0, -2])
            else:
                p_pos[0, ] = self.v_position[k, ] + np.array([1, -1])
                p_pos[1, ] = self.v_position[k, ] + np.array([-1, -1])
                p_pos[2, ] = self.v_position[k, ] + np.array([0, 2])

            p_pos = self.mod(p_pos)
            v_plaquette[k, ] = self.p_position2plaquette(p_pos)
        return v_plaquette

    def vertex2edge(self, vertex, both=True):
        """Given a list of vertices, return edges whose associated vertices are
        both contained in the list if both is True. Else return all edges
        associated to vertices.
        
        Args:
            vertex (int or 1d array int): input vertices.
            both (bool): if true, returned edges are only those where both
                vertices are contained in vertex list.

        Returns:
            edge (1d array np.int64): edges.
        """
        vertex = np.atleast_1d(vertex)
        edge = _vertex2edge(self.n_col, self.n_row,
                            self.v_position, self.v_orientation, vertex)

        if both:
            edge_both = np.zeros(self.n_edge, dtype=bool)
            for i in edge:
                v1 = self.e_vertex[i, 0] in vertex
                v2 = self.e_vertex[i, 1] in vertex
                if v1 and v2:
                    edge_both[i] = True
            return np.squeeze(np.argwhere(edge_both))
        return edge

    def plaquette2vertex(self, plaquette):
        """Given a list of plaquettes, return vertices associated.
        
        Args:
            plaquette (integer or 1d array int): plaquettes.
        
        Returns:
            p_vertex (2d array np.int64): shape (len(plaquette), 6) vertices
            where each row corresponds to a plaquette. 6 vertices are 
            associated with each plaquette.
        """
        plaquette = np.atleast_1d(plaquette)
        p_vertex = np.zeros([len(plaquette), 6], dtype=int)
        for i, p in enumerate(plaquette):
            p_vertex[i, :] = np.squeeze(
                np.argwhere(np.any(self.v_plaquette == p, 1)))
        return p_vertex

    def plaquette2edge(self, plaquette):
        """Given a list of plaquettes, return belonging edges.
        
        Args:
            plaquette (int): or 1d array (int) of plaquettes.

        Returns:
            p_edge (2d array int): shape (len(plaquette), 6).
        """
        p_vertex = self.plaquette2vertex(plaquette)
        p_edge = np.zeros(p_vertex.shape, dtype=int)
        for i in range(len(p_vertex)):
            p_edge[i, :] = self.vertex2edge(p_vertex[i, :])
        return p_edge

    def edge2vertex(self, edge):
        """Given edge list, obtain vertex indices of the two vertices 
        joined by each edge.
        
        Args:
            edge (integer or 1d array int): edge indices.
            
        Returns:
            (2d array int): shape (len(edge), 2).
        """
        edge = np.atleast_1d(edge)
        return self.e_vertex[edge, :]

    def edge_position(self, edge):
        """Returns the position of the given edges.
        
        Args:
            edge (integer or 1d array int).

        Returns:
            e_position (2d array np.float64): edge positions.
        """
        edge = np.atleast_1d(edge)
        return _edge_position(self.n_col, self.n_vertex, edge)

    def e_position2edge(self, e_position):
        """Given edge positions, obtain edge indices.

        Args:
            e_position (1d array float or 2d array float): a single edge
            position or several positions of edges with shape (?, 2).

        Returns:
            edge (1d array int): of edge indices.
        """
        e_position = np.atleast_2d(e_position)
        return _e_position2edge(self.n_col, self.n_vertex, e_position)

    def p_position2plaquette(self, p_position):
        """Given plaquette positions, return plaquettes.
        Args:
            p_position (1d array int): with a single plaquette position or 2d
                array (float) of positions of plaquettes with shape (?, 2).

        Returns:
            plaqutte (1d array int): of plaquette indices.
        """
        p_position = np.atleast_2d(p_position)
        n_plaquette = p_position.shape[0]
        plaquette = np.zeros(n_plaquette, dtype=int)
        for i in range(n_plaquette):
            plaquette[i] = p_position[i, 1]//3*self.n_col + \
                (p_position[i, 0] - p_position[i, 1]//3)//2 - 1
        return plaquette

    def mod(self, position):
        """_mod wrapper
        """
        return _mod(self.n_col, self.n_row, position)

    def mod_1d(self, position):
        """_mod_1d wrapper
        """
        return _mod_1d(self.n_col, self.n_row, position)

    def reciprocal_distance(self, p1, p2, pbc=True):
        """Obtain distance and shortest path between two plaquettes.
        
        If pbc is true, we consider the periodic boundary conditions of the
        lattice, taking p1 as reference and compute distance with p2 and 
        periodic equivalents of p2. Return the minimum of these distances.
        
        Args:
            p1 (int): if plaquette index, 1d array (int) if plaquette
            position. First vertex.
            p2 (int): if plaquette index, 1d array (int) if plaquette
            position. Second vertex.
            pbc (bool): whether to take into account periodic boundary 
            conditions.

        Returns:
            distance (int).
            path (1d array int): edges.
        """
        if np.size(p1) == 1:  # if input is index of plaquette
            p1 = self.p_position[p1, :]
            p2 = self.p_position[p2, :]
        if not pbc:
            return _reciprocal_distance(self.n_col, self.n_row, p1, p2)

        p2_pbc = _pbc_position(self.n_col, self.n_row, p1, p2)

        for i, p in enumerate(p2_pbc):
            d, path = _reciprocal_distance(self.n_col, self.n_row, p1, p)
            if i == 0:
                d_min = d
                path_min = path
                continue
            if d < d_min:
                d_min = d
                path_min = path

        return d_min, path_min

    def direct_distance(self, p1, p2, v1_orientation=0, pbc=True):
        """Obtain distance and shortest path between two vertices. 
        
        Analogous to reciprocal_distance
        
        Args:
            p1 (int): if vertex index, 1d array (int) if vertex position. 
                First vertex.
            p2 (int): if vertex index, 1d array (int) if vertex position. 
                Second vertex.
            v1_orientation (int):, orientation of first vertex. Optional,
                not necessary if p1 and p2 are given as index.
            pbc (bool): whether to take into account periodic boundary
                conditions.

        Returns:
            distance (int):.
            path (1d array int): edges.
        """
        if np.size(p1) == 1:  # if vertex index given as input
            v1_orientation = self.v_orientation[p1]
            p1 = self.v_position[p1, :]
            p2 = self.v_position[p2, :]
        if not pbc:
            return _direct_distance(self.n_col, self.n_row, v1_orientation, 
                                    p1, p2)

        p2_pbc = _pbc_position(self.n_col, self.n_row, p1, p2)

        for i, p in enumerate(p2_pbc):
            d, path = _direct_distance(
                self.n_col, self.n_row, v1_orientation, p1, p)
            if i == 0:
                d_min = d
                path_min = path
                continue
            if d < d_min:
                d_min = d
                path_min = path

        return d_min, path_min

    def gauge_line(self, direct, axis):
        """Outputs string in reciprocal or direct lattice transversing the
        system in direction given by axis.
        
        Args:
            direct (bool): direct or reciprocal.
            axis (int): 0 horizontal, 1 vertical.
            
        Returns:
            line (1d array int).
        """
        if direct and axis == 0:
            line = np.arange(self.n_col*2)
        elif direct and axis == 1:
            line = np.arange(0, self.n_col*2*self.n_row, self.n_col*2)
            line = np.concatenate([line, np.arange(
                self.n_vertex, self.n_row*self.n_col+self.n_vertex, self.n_col)])
        elif not direct and axis == 0:
            line = np.arange(self.n_vertex, self.n_vertex+self.n_col)
        elif not direct and axis == 1:
            line = np.arange(1, self.n_row*self.n_col*2+1, self.n_col*2)
        return line

    def plot_edge(self, edge, **kwargs):
        """Plots edge of the lattice. 
        
        Args:
            edge (int): index corresponding to the edge.
        """
        pos = np.squeeze(self.edge_position(edge))
        if self.e_orientation[edge] == 0:
            delta_x = np.array([0., 0.])
            delta_y = np.array([-1., 1.])
        elif self.e_orientation[edge] == 1:
            delta_x = np.array([-0.5, 0.5])
            delta_y = np.array([-.5, .5])
        elif self.e_orientation[edge] == 2:
            delta_x = np.array([0.5, -0.5])
            delta_y = np.array([-.5, .5])
        plt.plot((pos[0]+delta_x)*.5*self.w,
                 (pos[1]+delta_y)*.25*self.h, **kwargs)

    def plot_lattice(self, p_numbers=True, v_numbers=True, e_numbers=False,
                     axis=False):
        """Plots the lattice.

        Args:
            p_numbers (bool): print plaquette indices.
            v_numbers (bool): print vertex indices.
            e_numbers (bool): print edge indices.
            axis (bool): show axis.
        """
        plt.figure()
        if not axis:
            plt.axis('off')
        # plt.xlabel("$x$")
        # plt.ylabel("$y$")
        if v_numbers:
            for i in range(self.n_vertex):  # Vertex numbers
                plt.text(self.v_position[i, 0]*.5*self.w, self.v_position[i, 1]
                         * .25*self.h, i, ha="left", va="top", color='b')
        if p_numbers:
            for i in range(self.n_plaquette):  # Plaquette numbers
                plt.text(self.p_position[i, 0]*.5*self.w,
                         self.p_position[i, 1]*.25*self.h, i, color='r')
        if e_numbers:
            for i in range(self.n_edge):  # Edge numbers
                plt.text(np.squeeze(self.edge_position(i))[
                         0]*.5*self.w, np.squeeze(self.edge_position(i))[1]*.25*self.h, i, color='c')
        # Draw vertices
        # plt.scatter(self.v_position[:,0]*.5*self.w, self.v_position[:,1]*.25*self.h, 10, marker='o')
        # Draw edges
        for i in range(self.n_edge):
            self.plot_edge(i, color='k')
        # Draw Plaquettes
        # plt.scatter(p_position[:,0], p_position[:,1], 10, marker='o')
        # plt.autoscale(enable=True, axis='both', tight=True)
        # plt.show()

    def plot_syndrome(self, p_syndrome=[], v_syndrome=[]):
        """Given a set of plaquettes and vertices, mark them with red dots in
        the plot. We can plot the error syndrome this way.

        Args:
            p_syndrome (1d array int): plaquette indices.
            v_syndrome (1d array int): vertex indices.
        """
        plt.scatter(self.p_position[p_syndrome, 0]*.5*self.w,
                    self.p_position[p_syndrome, 1]*.25*self.h, 14, marker='o', color='r')
        plt.scatter(self.v_position[v_syndrome, 0]*.5*self.w, self.v_position[v_syndrome, 1]
                    * .25*self.h, 14, marker='o', color='r', zorder=5)

    def plot_error(self, x_error, z_error=[]):
        """Given X and Z errors, plot them in the lattice with different 
        colors.

        No error -> black
        X error -> red
        X and Z error -> yellow
        Z error -> green

        Args:
            x_error (1d array int): list of edges where X occurred.
            z_error (1d array int): optional, list of edges where Z occurred.
        """
        if len(x_error) == self.n_edge or len(z_error) == self.n_edge:
            x = x_error
            z = z_error
            if len(x_error) == 0:
                x = np.zeros(self.n_edge, dtype=bool)
            if len(z_error) == 0:
                z = np.zeros(self.n_edge, dtype=bool)
        else:
            x = np.zeros(self.n_edge, dtype=bool)
            z = np.zeros(self.n_edge, dtype=bool)
            x[x_error] = True
            z[z_error] = True
        for i in range(self.n_edge):
            if not (x[i] or z[i]):
                color = 'k'
            elif x[i] and z[i]:
                color = 'yellow'
            elif x[i]:
                color = 'r'
            elif z[i]:
                color = 'lime'
            self.plot_edge(i, color=color)

"""Functions for data generation.
"""
import numpy as np
import os
import pickle
import lzma
from itertools import repeat
from multiprocessing import Pool

from error_model import xyz_noise
from decoding import decoder_simple
from compute_pattern import bi2de


def data_point(p_X, p_Y, p_Z, lattice, ktc=False, seed=None):
    """Generates one data point: syndrome and label. Also keeps record of the
    total number of failed attempts that occurred when calling xyz_noise. xyz_noise may fail for long error patterns. If p_X and p_Y are
    low, this happens very rarely and does not impact the final results.

    Args:
        p_X (float): x error rate.
        p_Y (float): y error rate.
        p_Z (float): z error rate.
        lattice (HexagonalLattice instance)
        ktc: whether to produce data for the toric code instead of the semion
            code.
        seed: random seed.

    Returns:
        x (1d array of bool): syndrome.
        y (int): label.
        failed: total number of failed attempts when calling xyz_noise.
    """
    failed = 0
    while True:
        try:
            x_edge, z_edge, v_syn_ind, p_syn_ind = xyz_noise(
                p_X, p_Y, p_Z, lattice, ktc=ktc, seed=seed)
            break
        # xyz_noise may fail if for long error patterns. If p_X and p_Y are
        # low this happens very rarely and does not impact the final results.
        except AssertionError:  # String goes all around system
            failed += 1
        except FileNotFoundError:  # Pattern data npz file does not exist
            failed += 1

    y = decoder_simple.decode_result(
        x_edge, z_edge, v_syn_ind, p_syn_ind, lattice)
    v_syn = np.zeros(lattice.n_vertex, dtype=bool)
    v_syn[v_syn_ind] = True
    p_syn = np.zeros(lattice.n_plaquette, dtype=bool)
    p_syn[p_syn_ind] = True
    x = np.concatenate([v_syn, p_syn])
    y = np.squeeze(bi2de(np.concatenate(y)))
    return x, y.astype('int8'), failed


def data_generator(p_X, p_Y, p_Z, lattice, n, ktc=False):
    """Generate n data points using data_point function.
    """
    np.random.seed()
    x = np.zeros((n, lattice.n_vertex+lattice.n_plaquette), dtype=bool)
    y = np.zeros(n, dtype='int8')
    failed = 0
    for i in range(n):
        x[i, :], y[i], f = data_point(
            p_X, p_Y, p_Z, lattice, ktc=ktc, seed=None)
        failed += f
    return (x, y), failed


def compute_and_save(p_X, p_Y, p_Z, lattice, n,
                     ktc, fname, i_fname, verbose=False):
    """Generate data with data_generator and save in fname.
    """
    if verbose:
        print(f'Computing {fname} {i_fname}')
    data, failed = data_generator(p_X, p_Y, p_Z, lattice, n, ktc)

    fname = fname + str(i_fname)
    if os.path.isfile(fname):
        fname = fname + '_'
    with lzma.open(fname, 'wb') as f:
        pickle.dump(data, f)

    return failed


def main_data_gen(lattice, p_error, noise_type, start, end, data_type,
                  ktc=False, path='../training_data/', verbose=False):
    """Produces data files from start to end number, with given 
    characteristics. Each file contains 1e5 data points.

    Args:
        lattice (object): code lattice.
        p_error (float): error rate.
        noise_type (str): uncorrelated or depolarizing.
        start (int): starting file number.
        end (int): ending file number.
        data_type (str): str appended to file name (i.e. 'validation').
        ktc (bool): if true, produces data for Kitaev's toric code.
        path (str): path where to save the data.
        verbose (bool): if true, more feedback is given.
    """
    n_file = int(10**5)
    print(f'Computing size={lattice.n_row}, p_error={p_error}, '
          f'noise_type={noise_type} ...')
    if ktc:
        print('ktc data generation')

    if noise_type == 'depolarizing':
        p_X = p_error
        p_Y = p_error
        p_Z = p_error
    elif noise_type == 'uncorrelated':
        p_X = p_error-p_error**2
        p_Y = p_error**2
        p_Z = p_error-p_error**2
    else:
        raise ValueError(f'Invalid noise type: "{noise_type}".')

    subdir_name = f'{lattice.n_row}_{lattice.n_col}_{noise_type}'
    data_name = f'{lattice.n_row}_{lattice.n_col}_{noise_type}_{p_error}'
    if ktc:
        data_name += '_ktc'
        subdir_name += '_ktc'

    if not os.path.exists(os.path.join(path, subdir_name)):
        os.makedirs(os.path.join(path, subdir_name))

    fname0 = os.path.join(path, subdir_name, data_name)+'_'+data_type
    i_list = list(range(start, end))

    with Pool() as pool:
        failed = pool.starmap(compute_and_save,
                              zip(repeat(p_X), repeat(p_Y), repeat(p_Z),
                                  repeat(lattice), repeat(n_file),
                                  repeat(ktc), repeat(fname0), i_list,
                                  repeat(verbose)))

    failed_ratio = np.sum(failed)/((end-start)*n_file)
    if failed_ratio > 0.001:
        print(f'Warning: {np.sum(failed)} failed attempts out of a total of '
              f'{(end-start)*n_file} generated examples.')

    print('Done!')

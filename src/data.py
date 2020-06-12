"""Functions for data generation.
"""
import numpy as np
import os
import pickle
import lzma
from itertools import repeat
from multiprocessing import Pool, cpu_count

from error_model import xyz_noise
from decoding import decoder_simple
from compute_pattern import bi2de


def data_point(p_X, p_Y, p_Z, lattice, KTC=False, seed=None):
    """Generates one data point: syndrome and label.
    """

    while True:
        try:
            x_edge, z_edge, v_syn_ind, p_syn_ind = xyz_noise(
                p_X, p_Y, p_Z, lattice, KTC=KTC, seed=seed)
            break
        # xyz_noise may fail if a long error pattern appears. If p_X and p_Y
        # are low this happens very rarely and does not impact the final
        # results.
        except:
            continue

    y = decoder_simple.decode_result(
        x_edge, z_edge, v_syn_ind, p_syn_ind, lattice)
    v_syn = np.zeros(lattice.n_vertex, dtype=bool)
    v_syn[v_syn_ind] = True
    p_syn = np.zeros(lattice.n_plaquette, dtype=bool)
    p_syn[p_syn_ind] = True
    x = np.concatenate([v_syn, p_syn])
    y = np.squeeze(bi2de(np.concatenate(y)))
    return x, y.astype('int8')


def data_generator(p_X, p_Y, p_Z, lattice, instances, KTC=False):
    """Generate several data points.
    """
    np.random.seed()
    x = np.zeros((instances, lattice.n_vertex+lattice.n_plaquette), dtype=bool)
    y = np.zeros(instances, dtype='int8')
    for i in range(instances):
        x[i, :], y[i] = data_point(
            p_X, p_Y, p_Z, lattice, KTC=KTC, seed=None)
    return x, y


def compute_and_save(p_X, p_Y, p_Z, noise_type, lattice, N_instance, 
                     KTC, fname, i_fname, verbose=False):
    """Generate data with data_generator and save in fname.
    """
    if verbose:
        print(f'Computing {fname} {i_fname}')
    data = data_generator(p_X, p_Y, p_Z, lattice, N_instance, KTC)

    fname = fname + str(i_fname)
    if os.path.isfile(fname):
        fname = fname + '_'
    with lzma.open(fname, 'wb') as f:
        pickle.dump(data, f)


def main_data_gen(lattice, p_error, noise_type, start, end, data_type, 
                  KTC=False, path='../training_data/', verbose=False):
    """Produces data files from start to end number, with given 
    characteristics. Each file contains 1e5 data points.

    Args:
        lattice (object): code lattice.
        p_error (float): error rate.
        noise_type (str): uncorrelated or depolarizing.
        start (int): starting file number.
        end (int): ending file number.
        data_type (str): str appended to file name (i.e. 'validation').
        KTC (bool): if true, produces data for Kitaev's toric code.
        path (str): path where to save the data.
        verbose (bool): if true, more feedback is given.
    """
    N_size = lattice.n_row
    print(f'Computing size={N_size}, p_error={p_error}, '
          f'noise_type={noise_type} ...')
    if KTC:
        print('KTC data generation')

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
    if KTC:
        data_name += '_KTC'
        subdir_name += '_KTC'

    if not os.path.exists(os.path.join(path, subdir_name)):
        os.makedirs(os.path.join(path, subdir_name))

    fname0 = os.path.join(path, subdir_name, data_name)+'_'+data_type
    i_list = list(range(start, end))

    with Pool() as pool:
        pool.starmap(compute_and_save, 
            zip(repeat(p_X), repeat(p_Y), repeat(p_Z), repeat(noise_type),
                repeat(lattice), repeat(int(10**5)), repeat(KTC), 
                repeat(fname0), i_list, repeat(verbose)))

    print('Done!')
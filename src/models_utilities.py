"""Utilities for deep learning models.
"""
import numpy as np
import lzma
import pickle
import os
import warnings

from time import time
from numba import njit
# pylint: disable=import-error
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


@njit
def data2image(data, lattice_shape, width=0):
    """Receives 1d syndrome data and returns a 2d array with the syndromes
    distributed according to Table II of arXiv:2002.08666.
    """
    n_vertex = 2*lattice_shape[0]*lattice_shape[1]
    n_plaquette = lattice_shape[0]*lattice_shape[1]

    data_image = np.zeros((len(data[1]), lattice_shape[0]*2+2*width,
                           lattice_shape[1]*2+2*width, np.int64(1)), 
                           dtype=np.bool_)
    perm = np.roll(np.arange(lattice_shape[1]*2),  lattice_shape[0])
    for ind in range(len(data[0])):
        image = np.zeros(
            (lattice_shape[0]*2+2 * width,  lattice_shape[1]*2+2 * width), 
            dtype=np.bool_)
        central_image = np.zeros(
            (lattice_shape[0]*2,  lattice_shape[1]*2), dtype=np.bool_)
        # inputs has first vertices then plaquettes
        for i in range(n_vertex):
            row = i//(lattice_shape[1]*2)
            col = i - lattice_shape[1]*2*row
            central_image[2*row,
                          np.mod(col+row, 2*lattice_shape[1])] = data[0][ind][i]
        for i in range(n_plaquette):
            row = i//(lattice_shape[1])
            col = i - lattice_shape[1]*2*row
            central_image[2*row+1, np.mod(2*col+row+2, 2*lattice_shape[1])] = \
                data[0][ind][i + n_vertex]

        image = np.zeros(
            (lattice_shape[0]*2+2 * width,  lattice_shape[1]*2+2 * width), 
            dtype=np.bool_)

        if width!=0:
            image[0: width,  width:2 * lattice_shape[1] +
                width] = central_image[- width:, perm]
            image[-width:,  width:2 * lattice_shape[1] +
                width] = central_image[0:width, perm]
            image[width:-width,  width:2 * lattice_shape[1] + width] = \
                central_image

            image[:, 0:width] = image[:, 
                2*lattice_shape[1]:width+2*lattice_shape[1]]
            image[:, width+2*lattice_shape[1]:2*width+2 *
                lattice_shape[1]] = image[:, width:width*2]
        else:
            image = central_image

        data_image[ind, :, :, 0] = image
    return (data_image, data[1])


class DataLoader(Sequence):
    """Sequence class to generate data for fit_generator. Loads data from
    files in path.
    """

    def __init__(self, 
                lattice_shape, 
                noise_type, 
                batch_size, 
                p_error=None, 
                ktc=False, 
                path='../training_data/', 
                data_modifyer=None, 
                data_type='data', 
                batch_ignore=0):
        """DataLoader __init__

        Args:
            lattice_shape (tuple): code shape (lattice.n_row, lattice.n_col)
            noise_type (str): noise type.
            batch_size (int): batch size for training.
            p_error (float): optional, load only data with this error rate.
            ktc (bool): if true, load data for Kitaev's toric code.
            path (str): path of data.
            data_modifyer (function): optional, modify returned data with this
                function.
            data_type (str): e.g., 'data', 'validation'.
            batch_ignore (int): ignore the first n batches of data,
                effectively reducing the dataset. 
        """
        path = os.path.join(path, 
            f'{lattice_shape[0]}_{lattice_shape[1]}_{noise_type}')
        if ktc:
            path = path + '_ktc'

        self.batch_size = batch_size
        self.lattice_shape = lattice_shape
        self.p_error = p_error
        self.noise_type = noise_type
        self.batch_ignore = batch_ignore
        self.data_modifyer = data_modifyer

        # List all files fulfilling the conditions
        self.data_fname = []
        for fname in sorted(os.listdir(path)):
            if data_type not in fname:
                continue
            if self.p_error is not None and not (f'_{self.p_error}_' in fname):
                continue
            self.data_fname.append(os.path.join(path, fname))

        with lzma.open(self.data_fname[0], 'rb') as f:
            data = pickle.load(f)
        self.inst_per_file = len(data[1])

        self.steps_per_epoch = int(np.floor(
            len(self.data_fname)*self.inst_per_file/batch_size)) \
            - self.batch_ignore
        self.loaded_data = None
        self.loaded_file = None

    def __len__(self):
        return self.steps_per_epoch

    def inx_to_file_line(self, idx):
        file_index = np.mod(idx*self.batch_size//self.inst_per_file,
                      len(self.data_fname))
        line = idx*self.batch_size % self.inst_per_file
        return file_index, line

    def __getitem__(self, idx):
        idx += self.batch_ignore
        file_start, line_start = self.inx_to_file_line(idx)
        file_end, line_end = self.inx_to_file_line(idx+1)
        assert np.abs(np.mod(file_end-file_start, len(self.data_fname))) <= 1

        if file_start != self.loaded_file:
            with lzma.open(self.data_fname[file_start], 'rb') as f:
                if self.data_modifyer is not None:
                    self.loaded_data = self.data_modifyer(pickle.load(f))
                else:
                    self.loaded_data = pickle.load(f)
            self.loaded_file = file_start

        if file_start == file_end:
            X = self.loaded_data[0][line_start:line_end, :]
            Y = self.loaded_data[1][line_start:line_end]
        elif line_end == 0:
            X = self.loaded_data[0][line_start:, :]
            Y = self.loaded_data[1][line_start:]
        else:
            X = self.loaded_data[0][line_start:, :]
            Y = self.loaded_data[1][line_start:]
            with lzma.open(self.data_fname[file_end], 'rb') as f:
                if self.data_modifyer is not None:
                    self.loaded_data = self.data_modifyer(pickle.load(f))
                else:
                    self.loaded_data = pickle.load(f)
            self.loaded_file = file_end
            X = np.concatenate([X, self.loaded_data[0][:line_end, :]])
            Y = np.concatenate([Y, self.loaded_data[1][:line_end]])

        return X, Y


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.

    It has been modified to work on batch end, executed periodically after a 
    certain number of training steps.

    Args:
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of periods that produced the monitored
            quantity with no improvement after which training will
            be stopped.
            Validation quantities may not be produced for every
            epoch, if the validation frequency
            (`model.fit(validation_freq=5)`) is greater than one.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        min_delta: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
        batch_period: monitor metric with this period.
    """

    def __init__(self, 
                 monitor='loss', 
                 factor=0.1, 
                 patience=10,
                 verbose=1, 
                 mode='auto', 
                 min_delta=1e-4, 
                 cooldown=0, 
                 min_lr=0,
                 batch_period=2000):
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

        self.batch_period = batch_period
        self.batch = 0

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_batch_end(self, batch, logs=None):
        if batch-self.batch < self.batch_period:
            return
        self.batch = batch

        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                # print('\n wait value = '+str(self.wait)+'\n')
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing '
                                  'learning rate to %s.' % (batch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0
        

class EarlyStopping(Callback):
    """Monitor val_acc, save model if it increases, early stop if it stops
    improving.

    Works on batch end, executed periodically after a certain number of
    training steps.

    Args:
        val_sequence (Sequence object): validation sequence.
        batch_period (int): monitor metric every number of batches.
        patience (int): number of periods that produced the monitored quantity 
            with no improvement after which training will be stopped.
        min_delta (float): threshold for measuring the new optimum.
    """

    def __init__(self, 
                 monitor='acc',
                 batch_period=2000, 
                 patience=2, 
                 min_delta=0.0002):

        self.monitor = monitor
        self.batch_period = batch_period  # batch period for early stopping
        self.batch = 0

        self.patience = patience
        self.wait = 0
        self.min_delta = min_delta

        self._reset()

    def _reset(self):
        """Resets wait counter.
        """
        if 'acc' not in self.monitor:
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.wait = 0

    def on_batch_end(self, batch, logs={}):
        if batch-self.batch < self.batch_period:
            return
        self.batch = batch

        logs = logs or {}
        current = logs.get(self.monitor)

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:  # Early stopping
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f'\nEarly stopping: {self.monitor} did not improve. '\
                f'Before: {self.best:.4f}, afer: {current:.4f}, '\
                f'diff = {current-self.best:.4f}.\n')


class CustomSaver(Callback):
    """Computes val_acc and saves the model if it improves.
    
    Args:
        val_sequence (sequence object): validation data.
        model_name (str): name to save model.
        batch_period (int): monitor metric every number of batches.
    """

    def __init__(self, val_sequence, model_name, batch_period=2000):
        self.val_sequence = val_sequence
        self.model_name = model_name
        self.batch_period = batch_period
        self.batch = 0
        self.history = {'val_acc': [], 'val_loss': [], 'batch': []}
        self.val_acc = .0

    def on_batch_end(self, batch, logs={}):
        if batch-self.batch < self.batch_period:
            return
        self.batch = batch

        # Compute val_acc and save history
        val_loss, val_acc = self.model.evaluate_generator(
            self.val_sequence, verbose=0)
        self.history['val_acc'] += [val_acc]
        self.history['val_loss'] += [val_loss]
        self.history['batch'] += [batch]

        # Save model if val_acc improved
        if val_acc > self.val_acc:
            print(
                f'\nval_loss: {val_loss:.4f},'
                f' val_acc: {val_acc:.4f}, batch: {batch}. Saved.\n')
            self.model.save(self.model_name+'.h5')
            self.val_acc = val_acc
        else:
            print(
                f'\nval_loss: {val_loss:.4f},'
                f' val_acc: {val_acc:.4f}, batch: {batch}.\n')

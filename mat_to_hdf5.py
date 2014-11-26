#!/usr/bin/env python2.7

from collections import namedtuple
from multiprocessing import Pool
from common.data import jsdict
from common.time import Timer
from seizure_prediction import hdf5
from seizure_prediction.data import accumulate_data
from seizure_prediction.settings import load_settings
import numpy as np
import scipy.io
import scipy.signal
import os.path
import sys


Reader = namedtuple('Reader', ['read', 'exists', 'filename'])


class Metadata(object):

    def __init__(self):
        self.shape = None
        self.data_length_sec = None
        self.sampling_frequency = None
        self.channels = None
        self.sequences = []

    def add_shape(self, shape):
        if self.shape is None:
            self.shape = shape
        else:
            assert shape == self.shape

    def add_data_length_sec(self, data_length_sec):
        if self.data_length_sec is None:
            self.data_length_sec = data_length_sec
        else:
            assert data_length_sec == self.data_length_sec

    def add_sampling_frequency(self, sampling_frequency):
        if self.sampling_frequency is None:
            self.sampling_frequency = sampling_frequency
        else:
            assert sampling_frequency == self.sampling_frequency

    def add_channels(self, channels):
        if self.channels is None:
            self.channels = channels
        else:
            assert np.alltrue(channels == self.channels)

    def add_sequence(self, sequence):
        if sequence is not None:
            self.sequences.append(sequence)

    def __str__(self):
        seq_groups = []
        prev = None
        prev_start = None
        for seq in self.sequences:
            if prev_start is None:
                prev_start = seq
            else:
                if seq != prev + 1:
                    if prev_start == prev:
                        seq_groups.append('%d' % prev)
                    else:
                        seq_groups.append('%d-%d' % (prev_start, prev))
                    prev_start = seq
            prev = seq
        if prev_start is not None:
            seq_groups.append('%d-%d' % (prev_start, prev))

        seq_mega_groups = []
        prev = None
        count = 1
        for group in seq_groups:
            if prev is not None:
                if prev != group:
                    seq_mega_groups.append(('%d of %s' % (count, prev)) if count > 1 else prev)
                    count = 1
                else:
                    count += 1
            prev = group
        if prev is not None:
            seq_mega_groups.append('%d of %s' % (count, prev) if count > 1 else prev)

        return str({
            'shape': self.shape,
            'data_length_sec': self.data_length_sec,
            'sampling_frequency': self.sampling_frequency,
            'channels': len(self.channels) if self.channels is not None else None,
            'sequences': seq_mega_groups
        })


def process_data_sub_job(settings, filename_in_fmt, filename_out_fmt, id, num_jobs):

    pid = os.getpid()
    reader = mat_reader(target, settings.data_dir)

    num_processed = 0
    for i in xrange(id + 1, sys.maxint, num_jobs):
        out_index = i - 1
        filename_in = filename_in_fmt % i
        filename_out = filename_out_fmt % out_index if filename_out_fmt is not None else None
        filename_out_temp = '%s.pid.%d.tmp' % (filename_out, pid) if filename_out is not None else None

        if filename_out is not None and os.path.exists(filename_out):
            num_processed += 1
            continue

        if not reader.exists(filename_in):
            if i == id + 1:
                print 'Could not find file', reader.filename(filename_in)
                return 0
            break

        print 'Runner %d processing %s' % (id, reader.filename(filename_in))

        segment = reader.read(filename_in)
        data = process_data(segment)
        hdf5.write(filename_out_temp, data)

        os.rename(filename_out_temp, filename_out)

        num_processed += 1

    return num_processed


def process_data(segment):
    data_key = [key for key in segment.keys() if not key.startswith('_')][0]
    data = segment[data_key][0][0]

    X = data[0]
    data_length_sec = int(data[1][0][0])
    sampling_frequency = float(data[2][0][0])
    channels = [ch[0] for ch in data[3][0]]
    sequence = int(data[4][0][0]) if len(data) >= 5 else None

    min_freq = 195.0
    def find_q():
        q = 2
        while True:
            f = sampling_frequency / q
            if f < min_freq:
                return q - 1
            q += 1

    if sampling_frequency > min_freq:
        q = find_q()
        if q > 1:
            # if X.dtype != np.float64:
            #     X = X.astype(np.float64)
            # X -= X.mean(axis=0)
            X = scipy.signal.decimate(X, q, ftype='fir', axis=X.ndim-1)
            X = np.round(X).astype(np.int16)
            # if X.dtype != np.float32:
            #     X = X.astype(np.float32)
            sampling_frequency /= q

    channels = np.array(channels, dtype=str(channels[0].dtype).replace('U', 'S'))
    out = {
        'X': X,
        'data_length_sec': data_length_sec,
        'sampling_frequency': sampling_frequency,
        'num_channels': X.shape[0],
        'channels': channels,
        'target': target,
        'data_type': data_type,
    }
    if sequence is not None:
        out['sequence'] = sequence

    return jsdict(out)


#used for verifying and printing
def collect_metadata(data, metadata_accum):
    metadata_accum.add_shape(data.X.shape)
    metadata_accum.add_data_length_sec(data.data_length_sec)
    metadata_accum.add_sampling_frequency(data.sampling_frequency)
    metadata_accum.add_channels(data.channels)
    if 'sequence' in data:
        metadata_accum.add_sequence(data.sequence)


def process_and_merge_segments(target, data_type, out_dir, metadata, N_jobs):
    filename_out = os.path.join(out_dir, '%s_%s.hdf5' % (target, data_type))

    if os.path.exists(filename_out):
        return 0

    print 'Processing %s ...' % filename_out

    filename_in_fmt = '%s_%s_segment_%%.4d' % (target, data_type)
    filename_out_fmt = '%s/%s_%s_segment_%%d.hdf5' % (out_dir, target, data_type)

    # process_data_sub_job(settings, filename_in_fmt, filename_out_fmt, 0, 1)
    pool = Pool(N_jobs)
    results = [pool.apply_async(process_data_sub_job, [settings, filename_in_fmt, filename_out_fmt, id, N_jobs])
        for id in range(N_jobs)]
    pool.close()
    pool.join()

    num_processed = np.sum([r.get() for r in results])
    for i in xrange(num_processed):
        data = hdf5.read(filename_out_fmt % i)
        collect_metadata(data, metadata)

    _, accum_meta = accumulate_data(settings, target, data_type, tag=None,
        output_to_original_data_dir=True, quiet=True)

    return accum_meta.num_segments


def mat_reader(target, dir):
    ext = '.mat'
    expand_filename = lambda filename: os.path.join(dir, target, filename + ext)
    read = lambda filename: scipy.io.loadmat(expand_filename(filename))
    exists = lambda filename: os.path.exists(expand_filename(filename))
    return Reader(read=read, exists=exists, filename=expand_filename)


def process_mat_into_hdf5(settings, target, data_type, N_jobs):
    assert data_type in ('preictal', 'interictal', 'test')

    print 'Loading data ...'
    timer = Timer()

    out_dir = os.path.join(settings.data_dir)
    metadata = Metadata()
    segments_processed = process_and_merge_segments(target, data_type, out_dir, metadata, N_jobs)

    print 'Processed %d segments in %s' % (segments_processed, timer.pretty_str())
    print data_type, 'Metadata', metadata


if __name__ == "__main__":

    settings = load_settings()
    N_jobs = 8

    data_types = [
        'preictal',
        'interictal',
        'test'
    ]

    targets = [
        'Dog_1',
        'Dog_2',
        'Dog_3',
        'Dog_4',
        'Dog_5',
        'Patient_1',
        'Patient_2'
    ]

    for target in targets:
        for data_type in data_types:
            process_mat_into_hdf5(settings, target, data_type, N_jobs)

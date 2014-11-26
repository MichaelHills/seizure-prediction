import numpy as np
from common.data import jsdict
from common.time import Timer
import os.path
from multiprocessing import Pool
import h5py
import sys
import re
import glob


def read_hdf5_segment(file, key, start=None, end=None):
    dset = file[key]
    meta = {}
    for key, value in dset.attrs.iteritems():
        meta[key] = value

    if start is None and end is None:
        X = dset[:]
    else:
        if start >= dset.shape[0]:
            return None
        if (start + 1 == end):
            X = dset[start]
        else:
            X = dset[start:end]

    return X, meta


def write_hdf5_segment(file, key, data, meta=None):
    dset = file.create_dataset(key, data=data)

    if meta is not None:
        for key, value in meta.iteritems():
            dset.attrs[key] = value
            # print key, value


# NOTE(mike): just doing np.array(list_of_numpy_arrays) seems really slow,
# This seems to be a bit faster. However I really need to do some benchmarking
# to determine what is the fastest method.
def to_np_array(X):
    if isinstance(X[0], np.ndarray):
        # return np.vstack(X)
        out = np.empty([len(X)] + list(X[0].shape), dtype=X[0].dtype)
        for i, x in enumerate(X):
            out[i] = x
        return out

    return np.array(X)

# The worker method for a process to work on it's subset of the data. It will push
# the data through the pipeline working on 1 segment at a time. Segments are pulled
# in 1 at a time to keep working-set of memory to a minimum.
def process_data_sub_job(filename_in, filename_out_fmt, id, num_jobs, process_data_fn):
    if not os.path.exists(filename_in):
        return 0

    pid = os.getpid()

    num_processed = 0
    for i in xrange(id, sys.maxint, num_jobs):

        filename_out = filename_out_fmt % i if filename_out_fmt is not None else None
        # Use temp filename then rename the completed file to the proper name.
        # This is more or less an atomic update. Cancelling the program should
        # never leave data in a half-written state. Hence only the tempfile
        # will be in a half-written state and the pid determines when the process
        # is still alive and still processing the data. An inactive pid means the
        # tempfile is trash and can be deleted.
        filename_out_temp = '%s.pid.%d.tmp' % (filename_out, pid) if filename_out is not None else None

        if filename_out is not None and os.path.exists(filename_out):
            num_processed += 1
            continue

        with h5py.File(filename_in, 'r') as f:
            segment = read_hdf5_segment(f, 'X', start=i, end=i+1)
            if segment is None:
                break
            X, meta = segment

        data_obj = {}
        for k, v in meta.iteritems():
            data_obj[k] = v

        # save disk space
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        X = process_data_fn(X, jsdict(data_obj))

        if filename_out is not None:
            with h5py.File(filename_out_temp, 'w', libver='latest') as f:
                if X.dtype != np.float32:
                    X = X.astype(np.float32)
                write_hdf5_segment(f, 'X', X)

            os.rename(filename_out_temp, filename_out)

        num_processed += 1

    return num_processed

# filenames for single accumulated file
def single_filename_builder(target, data_type, dir, tag=None):
    if tag is not None:
        filename = '%s_%s_%s.hdf5' % (target, data_type, tag)
    else:
        filename = '%s_%s.hdf5' % (target, data_type)

    return os.path.join(dir, filename)


# filenames for individual segments before they get accumulated into one big file
def segment_filename_builder(target, data_type, dir, tag=None):
    if tag is not None:
        filename = '%s_%s_%s_segment_%%d.hdf5' % (target, data_type, tag)
    else:
        filename = '%s_%s_segment_%%d.hdf5' % (target, data_type)

    return os.path.join(dir, filename)

# glue code around process_data_sub_job to setup input/output destinations and the
# processing method (applying pipeline on input data)
def process_data_job(settings, target, data_type, id, num_jobs, pipeline):

    def process(data, meta):
        out = pipeline.apply(data, meta)
        return out

    input_source = pipeline.get_input_source()
    input_source_pipeline = input_source.get_pipeline()
    input_tag = input_source_pipeline.get_name() if input_source_pipeline is not None else None
    input_data_dir = settings.data_dir if input_tag is None else settings.cache_dir
    filename_in = single_filename_builder(target, data_type, input_data_dir, input_tag)
    filename_out_fmt = segment_filename_builder(target, data_type, settings.cache_dir, pipeline.get_name())
    return process_data_sub_job(filename_in, filename_out_fmt, id, num_jobs, process_data_fn=process)

# Accumulates N segments into a single file as it is faster to load data this way.
def accumulate_data(settings, target, data_type, tag, output_to_original_data_dir=False, quiet=False, meta_only=False):
    output_dir = settings.data_dir if output_to_original_data_dir else settings.cache_dir
    filename_out = single_filename_builder(target, data_type, output_dir, tag)
    orig_filename_in = single_filename_builder(target, data_type, settings.data_dir)

    def collect_meta(filename):
        meta = {}
        with h5py.File(filename, 'r') as f:
            meta['num_segments'] = f['X'].shape[0]
            if 'sequence' in f.keys():
                meta['sequence'] = f['sequence'][:]
            for k, v in f['X'].attrs.iteritems():
                meta[k] = v
        return meta

    # load already processed output file
    if os.path.exists(filename_out):
        # pull meta off original data
        meta = collect_meta(orig_filename_in)

        # pull X data off processed data
        with h5py.File(filename_out, 'r') as f:
            meta['X_shape'] = f['X'].shape
            X = f['X'][:] if not meta_only else None
            if not quiet: print 'from cache ...',
            return X, jsdict(meta)
    else:
        # get ready to process all segments into 1 file, starting with getting the meta-data ready
        if not quiet: print 'processing ...',
        pid = os.getpid()
        filename_in_fmt = segment_filename_builder(target, data_type, output_dir, tag)

        orig_filename_in = single_filename_builder(target, data_type, settings.data_dir)

        # meta-data is collected differently when doing the first data conversion from mat to hdf5
        if output_to_original_data_dir:
            print 'Collecting metadata...'
            # Creating original files... pull metadata off first one, and also collect sequences
            meta = None
            sequence = []
            num_segments = 0
            for i in xrange(0, sys.maxint, 1):
                filename = filename_in_fmt % i
                if not os.path.exists(filename):
                    if num_segments == 0:
                        print 'Could not find file ', filename
                        sys.exit(1)
                    break

                with h5py.File(filename, 'r') as f_in:
                    meta_attrs = f_in['__metadata'].attrs
                    if 'sequence' in meta_attrs:
                        sequence.append(meta_attrs['sequence'])

                    if meta is None:
                        meta = {}
                        meta['channels'] = f_in['channels'][:]
                        for key in meta_attrs.keys():
                            if key != 'sequence':
                                meta[key] = meta_attrs[key]
                num_segments += 1

            if len(sequence) > 0:
                meta['sequence'] = sequence

            meta['num_segments'] = num_segments

            print 'Accumulating segments...'
        else:
            # pull metadata off the original data files
            meta = collect_meta(orig_filename_in)

        # now accumulate X data to a single file
        num_segments = meta['num_segments']
        filename_out_temp = '%s.pid.%d.tmp' % (filename_out, pid) if filename_out is not None else None
        with h5py.File(filename_out_temp, 'w-', libver='latest') as f_out:
            X_out = None
            for i in xrange(num_segments):
                with h5py.File(filename_in_fmt % i, 'r') as f_in:
                    X_in = f_in['X']
                    # init X_out
                    if X_out is None:
                        X_out = f_out.create_dataset('X', shape=[num_segments] + list(X_in.shape), dtype=X_in.dtype)
                        meta['X_shape'] = X_out.shape
                        for k, v in meta.iteritems():
                            X_out.attrs[k] = v

                    X_out[i] = X_in[:]
            X = X_out[:]

        # finalize
        os.rename(filename_out_temp, filename_out)
        # clean up
        for i in xrange(num_segments):
            try:
                os.remove(filename_in_fmt % i)
            except:
                pass

        return X, jsdict(meta)


# helper to check whether data exists in the data cache
def data_exists(settings, target, data_type, pipeline):
    filename_out = single_filename_builder(target, data_type, settings.cache_dir, pipeline.get_name())
    return os.path.exists(filename_out)


# Multi-process data loading, data segments are processed through the given pipeline, then are accumulated
# to a single file.
#
# check_only: returns True if data exists else false
# quiet: suppress prints if True
# meta_only: Actual X data is not fetched if meta_only is True, useful for light-weight data-loading
#            to check number of training samples or number of features.
def load_data_mp(settings, target, data_type, pipeline, check_only=False, quiet=False, meta_only=False):
    filename_out = single_filename_builder(target, data_type, settings.cache_dir, pipeline.get_name())
    filename_out_exists = os.path.exists(filename_out)
    if check_only:
        return filename_out_exists

    input_source = pipeline.get_input_source()
    input_source_pipeline = input_source.get_pipeline()
    if input_source_pipeline is not None:
        if not load_data_mp(settings, target, data_type, input_source_pipeline, check_only=True, quiet=quiet, meta_only=meta_only):
            if not quiet: print 'Preparing input source', input_source_pipeline.get_name()
            load_data_mp(settings, target, data_type, input_source_pipeline, check_only=False, quiet=quiet, meta_only=meta_only)
            if not quiet: print 'Input source ready'


    if not quiet: print 'Loading %s data ...' % data_type,
    timer = Timer()

    # TODO(mike): re-implement tmpfile cleanup that isn't really slow in the face of the genetic algorithm
    # spamming the disk with cross-validation score files.

    # clear cache of tmp files
    # regex = re.compile(r""".*\.pid\.(\d+)""")
    # for file in glob.glob(os.path.join(settings.cache_dir, '*.tmp')):
    #     match = regex.match(file)
    #     assert match is not None
    #     pid = int(match.group(1))
    #     try:
    #         os.getpgid(pid)
    #     except:
    #         print 'Removing', file
    #         os.remove(file)

    if not filename_out_exists:
        # DEBUG
        debug = False
        # debug = True
        if debug:
            print 'DEBUG'
            process_data_job(settings, target, data_type, 0, 1, pipeline)
            print 'Done'
        else:
            pool = Pool(settings.N_jobs)
            [pool.apply_async(process_data_job, [settings, target, data_type, i, settings.N_jobs, pipeline]) for i in range(settings.N_jobs)]
            pool.close()
            pool.join()

    accum, accum_meta = accumulate_data(settings, target, data_type, pipeline.get_name(), quiet=quiet, meta_only=meta_only)

    if not quiet: print 'prepared %d segments in %s %s %s' % (accum_meta.num_segments, timer.pretty_str(), accum_meta.X_shape, pipeline.get_name())

    return accum, accum_meta


import numpy as np
from scipy.signal import hilbert
from sklearn import preprocessing
import scipy.stats
import pandas as pd

from data import to_np_array


# optional modules for trying out different transforms
try:
    import pywt
except ImportError, e:
    pass

try:
    from scikits.talkbox.features import mfcc
except ImportError, e:
    pass

# for auto regressive model
try:
    import spectrum
except ImportError, e:
    pass



# NOTE(mike): Some transforms operate on the raw data in the shape (NUM_CHANNELS, NUM_FEATURES).
# Others operate on windowed data in the shape (NUM_WINDOWS, NUM_CHANNELS, NUM_FEATURES).
# I've been a bit lazy and just made the ApplyManyTransform base class helper... so if you intend
# a transform to work on pre-windowed data, just write a plain transform with apply method, if
# you intend to work on windowed-data, derive from ApplyManyTransform and implement apply_one method.
# Really this is just a problem of number of axes, and np.apply_along_axis could probably be used to
# clean up this mess. :) I haven't bothered updating it as things are working as they are.

class ApplyManyTransform(object):
    def apply(self, datas, meta):
        if datas.ndim >= 3:
            out = []
            for d in datas:
                out.append(self.apply_one(d, meta))

            return to_np_array(out)
        else:
            return self.apply_one(datas, meta)


class FFT:
    """
    Apply Fast Fourier Transform to the last axis.
    """
    def get_name(self):
        return "fft"

    def apply(self, data, meta=None):
        axis = data.ndim - 1
        return np.fft.rfft(data, axis=axis)


class Slice:
    """
    Take a slice of the data on the last axis.
    e.g. Slice(1, 48) works like a normal python slice, that is 1-47 will be taken
    """
    def __init__(self, start, end=None):
        self.start = start
        self.end = end

    def get_name(self):
        return "slice%d%s" % (self.start, '-%d' % self.end if self.end is not None else '')

    def apply(self, data, meta=None):
        s = [slice(None),] * data.ndim
        s[-1] = slice(self.start, self.end)
        return data[s]


class MFCC:
    """
    Mel-frequency cepstrum coefficients
    """
    def get_name(self):
        return "mfcc"

    def apply(self, data, meta=None):
        all_ceps = []
        for ch in data:
            ceps, mspec, spec = mfcc(ch)
            all_ceps.append(ceps.ravel())

        return to_np_array(all_ceps)


class Magnitude:
    """
    Take magnitudes of Complex data
    """
    def get_name(self):
        return "mag"

    def apply(self, data, meta=None):
        return np.abs(data)


class Log:
    """
    Apply LogE
    """
    def get_name(self):
        return "log"

    def apply(self, data, meta=None):
        indices = np.where(data <= 0)
        data[indices] = np.max(data)
        data[indices] = (np.min(data) * 0.1)
        return np.log(data)


class Log2:
    """
    Apply Log2
    """
    def get_name(self):
        return "log2"

    def apply(self, data, meta=None):
        indices = np.where(data <= 0)
        data[indices] = np.max(data)
        data[indices] = (np.min(data) * 0.1)
        return np.log2(data)


class Log10:
    """
    Apply Log10
    """
    def get_name(self):
        return "log10"

    def apply(self, data, meta=None):
        indices = np.where(data <= 0)
        data[indices] = np.max(data)
        data[indices] = (np.min(data) * 0.1)
        return np.log10(data)


class Stats(ApplyManyTransform):
    """
    Subtract the mean, then take (min, max, standard_deviation) for each channel.
    """
    def get_name(self):
        return "stats"

    def apply_one(self, data, meta=None):
        # data[ch][dim]
        shape = data.shape
        out = np.empty((shape[0], 3))
        for i in range(len(data)):
            ch_data = data[i]
            ch_data -= np.mean(ch_data)
            outi = out[i]
            outi[0] = np.std(ch_data)
            outi[1] = np.min(ch_data)
            outi[2] = np.max(ch_data)

        return out


class MomentPerChannel(ApplyManyTransform):
    """
    Calculate the Nth moment per channel.
    """
    def __init__(self, n):
        self.n = n

    def get_name(self):
        return "moment%d" % self.n

    def apply_one(self, data, meta=None):
        return scipy.stats.moment(data, moment=self.n, axis=data.ndim-1)


class UnitScale:
    """
    Scale across the last axis.
    """
    def get_name(self):
        return 'unit-scale'

    def apply(self, data, meta=None):
        return preprocessing.scale(data, axis=data.ndim-1)


class UnitScaleFeat:
    """
    Scale across the first axis, i.e. scale each feature.
    """
    def get_name(self):
        return 'unit-scale-feat'

    def apply(self, data, meta=None):
        return preprocessing.scale(data.astype(np.float64), axis=0)


class CorrelationMatrix(ApplyManyTransform):
    """
    Calculate correlation coefficients matrix across all EEG channels.
    """
    def get_name(self):
        return 'corr-mat'

    def apply_one(self, data, meta=None):
        return np.corrcoef(data)


class Eigenvalues(ApplyManyTransform):
    """
    Take eigenvalues of a matrix, and sort them by magnitude in order to
    make them useful as features (as they have no inherent order).
    """
    def get_name(self):
        return 'eigen'

    def apply_one(self, data, meta=None):
        w, v = np.linalg.eig(data)
        w = np.absolute(w)
        w.sort()
        return w


# Take the upper right triangle of a matrix
def upper_right_triangle(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            accum.append(matrix[i, j])

    return to_np_array(accum)


class UpperRightTriangle(ApplyManyTransform):
    """
    Take the upper right triangle of a matrix.
    """
    def get_name(self):
        return 'urt'

    def apply_one(self, data, meta=None):
        assert data.ndim == 2 and data.shape[0] == data.shape[1]
        return upper_right_triangle(data)


class FreqCorrelation(ApplyManyTransform):
    """
    Correlation in the frequency domain. First take FFT with (start, end) slice options,
    then calculate correlation co-efficients on the FFT output, followed by calculating
    eigenvalues on the correlation co-efficients matrix.

    The output features are (fft, upper_right_diagonal(correlation_coefficients), eigenvalues)

    Features can be selected/omitted using the constructor arguments.
    """
    def __init__(self, start_hz, end_hz, option, use_phase=False, with_fft=False, with_corr=True, with_eigen=True):
        self.start_hz = start_hz
        self.end_hz = end_hz
        self.option = option
        self.with_fft = with_fft
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        self.use_phase = use_phase
        assert option in ('us', 'usf', 'none', 'fft_in')
        assert with_corr or with_eigen

    def get_name(self):
        selections = []
        if self.option in ('us', 'usf', 'fft_in'):
            selections.append(self.option)
        if self.with_fft:
            selections.append('fft')
        if not self.with_corr:
            selections.append('nocorr')
        if not self.with_eigen:
            selections.append('noeig')
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        return 'freq-corr%s-%s-%s%s' % ('-phase' if self.use_phase else '', self.start_hz, self.end_hz, selection_str)

    def apply_one(self, data, meta=None):
        num_time_samples = data.shape[-1] if self.option != 'fft_in' else (data.shape[-1] - 1) * 2 # revert FFT shape change
        if self.start_hz == 1 and self.end_hz is None:
            freq_slice = Slice(self.start_hz, self.end_hz)
        else:
            # FFT range is from 0Hz to 101Hz
            def calc_index(f):
                return int((f / (meta.sampling_frequency/2.0)) * num_time_samples) if f is not None else num_time_samples
            freq_slice = Slice(calc_index(self.start_hz), calc_index(self.end_hz))
            # print data.shape, freq_slice.start, freq_slice.end
            # import sys
            # sys.exit(0)

        data1 = data
        if self.option != 'fft_in':
            data1 = FFT().apply(data1)
        data1 = freq_slice.apply(data1)
        if self.use_phase:
            data1 = np.angle(data1)
        else:
            data1 = Magnitude().apply(data1)
            data1 = Log10().apply(data1)

        data2 = data1
        if self.option == 'usf':
            data2 = UnitScaleFeat().apply(data2)
        elif self.option == 'us':
            data2 = UnitScale().apply(data2)

        data2 = CorrelationMatrix().apply_one(data2)

        if self.with_eigen:
            w = Eigenvalues().apply_one(data2)

        out = []
        if self.with_corr:
            data2 = upper_right_triangle(data2)
            out.append(data2)
        if self.with_eigen:
            out.append(w)
        if self.with_fft:
            data1 = data1.ravel()
            out.append(data1)
        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)


class Correlation(ApplyManyTransform):
    """
    Correlation in the time domain. Calculate correlation co-efficients
    followed by calculating eigenvalues on the correlation co-efficients matrix.

    The output features are (upper_right_diagonal(correlation_coefficients), eigenvalues)

    Features can be selected/omitted using the constructor arguments.
    """
    def __init__(self, scale_option, with_corr=True, with_eigen=True):
        self.scale_option = scale_option
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        assert scale_option in ('us', 'usf', 'none')
        assert with_corr or with_eigen

    def get_name(self):
        selections = []
        if self.scale_option != 'none':
            selections.append(self.scale_option)
        if not self.with_corr:
            selections.append('nocorr')
        if not self.with_eigen:
            selections.append('noeig')
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        return 'corr%s' % (selection_str)

    def apply_one(self, data, meta=None):
        data1 = data
        if self.scale_option == 'usf':
            data1 = UnitScaleFeat().apply(data1)
        elif self.scale_option == 'us':
            data1 = UnitScale().apply(data1)

        data1 = CorrelationMatrix().apply_one(data1)

        # patch nans
        data1[np.where(np.isnan(data1))] = -2

        if self.with_eigen:
            w = Eigenvalues().apply_one(data1)

        out = []
        if self.with_corr:
            data1 = upper_right_triangle(data1)
            out.append(data1)
        if self.with_eigen:
            out.append(w)

        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)


class FlattenChannels(object):
    """
    Reshapes the data from (..., N_CHANNELS, N_FEATURES) to (..., N_CHANNELS * N_FEATURES)
    """
    def get_name(self):
        return 'fch'

    def apply(self, data, meta=None):
        if data.ndim == 2:
            return data.ravel()
        elif data.ndim == 3:
            s = data.shape
            return data.reshape((s[0], np.product(s[1:])))
        else:
            raise NotImplementedError()


class Windower:
    """
    Breaks the time-series data into N second segments, for example 60s windows
    will create 10 windows given a 600s segment. The output is the reshaped data
    e.g. (600, 120000) -> (600, 10, 12000)
    """
    def __init__(self, window_secs=None):
        self.window_secs = window_secs
        self.name = 'w-%ds' % window_secs if window_secs is not None else 'w-whole'

    def get_name(self):
        return self.name

    def apply(self, X, meta=None):
        if self.window_secs is None:
            return X.reshape([1] + list(X.shape))

        num_windows = meta.data_length_sec / self.window_secs
        samples_per_window = self.window_secs * int(meta.sampling_frequency)
        samples_used = num_windows * samples_per_window
        samples_dropped = X.shape[-1] - samples_used
        X = Slice(samples_dropped).apply(X)
        out = np.split(X, num_windows, axis=X.ndim-1)
        out = to_np_array(out)
        return out

class PreictalWindowGenerator:
    """
    Experimental windower that generates overlapping windows for preictal segments only.
    The window_secs parameter describes how long each window is, and gen_factor describes
    how many extra windows you want as a multiplier.

    For example given a 600s segment, a window size of 60s will give you 10 windows,
    this number is then multiplied by gen_factor, e.g. 20 windows if gen_factor is 2.
    The window size is fixed and the starting point for each window will be evenly-spaced.

    It's been a while since I've used this, not even sure if it works properly...
    """
    def __init__(self, window_secs, gen_factor):
        self.window_secs = window_secs
        self.gen_factor = gen_factor
        self.name = 'wg-%ds-%d' % (window_secs, gen_factor)
        self.windower = Windower(window_secs)

    def get_name(self):
        return self.name

    def apply(self, X, meta):
        if meta.data_type == 'preictal':
            num_windows = (meta.data_length_sec / self.window_secs) * self.gen_factor
            samples_per_window = self.window_secs * int(meta.sampling_frequency) / self.gen_factor
            samples_used = num_windows * samples_per_window
            samples_dropped = X.shape[-1] - samples_used
            X = Slice(samples_dropped).apply(X)
            pieces = np.split(X, num_windows, axis=X.ndim-1)
            pieces_per_window = self.gen_factor
            gen = [np.concatenate(pieces[i:i+pieces_per_window], axis=pieces[0].ndim - 1) for i in range(0, num_windows - self.gen_factor + 1)]
            gen = to_np_array(gen)
            return gen
        else:
            return self.windower.apply(X, meta)


class Hurst:
    """
    Hurst exponent per-channel, see http://en.wikipedia.org/wiki/Hurst_exponent

    Another description can be found here: http://www.ijetch.org/papers/698-W10024.pdf
    Kavya Devarajan, S. Bagyaraj, Vinitha Balasampath, Jyostna. E. and Jayasri. K.,
    "EEG-Based Epilepsy Detection and Prediction," International Journal of Engineering
    and Technology vol. 6, no. 3, pp. 212-216, 2014.

    """
    def get_name(self):
        return 'hurst'

    def apply(self, X, meta):
        def apply_one(x):
            x -= x.mean()
            z = np.cumsum(x)
            r = (np.maximum.accumulate(z) - np.minimum.accumulate(z))[1:]
            s = pd.expanding_std(x)[1:]

            # prevent division by 0
            s[np.where(s == 0)] = 1e-12
            r += 1e-12

            y_axis = np.log(r / s)
            x_axis = np.log(np.arange(1, len(y_axis) + 1))
            x_axis = np.vstack([x_axis, np.ones(len(x_axis))]).T

            m, b = np.linalg.lstsq(x_axis, y_axis)[0]
            return m

        return np.apply_along_axis(apply_one, -1, X)


class PFD(ApplyManyTransform):
    """
    Petrosian fractal dimension per-channel

    Implementation derived from reading:
    http://arxiv.org/pdf/0804.3361.pdf
    F.S. Bao, D.Y.Lie,Y.Zhang,"A new approach to automated epileptic diagnosis using EEG
    and probabilistic neural network",ICTAI'08, pp. 482-486, 2008.
    """
    def get_name(self):
        return 'pfd'

    def pfd_for_ch(self, ch):
        diff = np.diff(ch, n=1, axis=0)

        asign = np.sign(diff)
        sign_changes = ((np.roll(asign, 1) - asign) != 0).astype(int)
        N_delta = np.count_nonzero(sign_changes)

        n = len(ch)
        log10n = np.log10(n)
        return log10n / (log10n + np.log10(n / (n + 0.4 * N_delta)))

    def apply_one(self, X, meta=None):
        return to_np_array([self.pfd_for_ch(ch) for ch in X])


def hfd(X, kmax):
    N = len(X)
    Nm1 = float(N - 1)
    L = np.empty((kmax,))
    L[0] = np.sum(abs(np.diff(X, n=1))) # shortcut :)
    for k in xrange(2, kmax + 1):
        Lmks = np.empty((k,))
        for m in xrange(1, k + 1):
            i_end = (N - m) / k # int
            Lmk_sum = np.sum(abs(np.diff(X[np.arange(m - 1, m + (i_end + 1) * k - 1, k)], n=1)))
            Lmk = Lmk_sum * Nm1 / (i_end * k)
            Lmks[m-1] = Lmk

        L[k - 1] = np.mean(Lmks)

    a = np.empty((kmax, 2))
    a[:, 0] = np.log(1.0 / np.arange(1.0, kmax + 1.0))
    a[:, 1] = 1.0

    b = np.log(L)

    # find x by solving for ax = b
    x, residues, rank, s = np.linalg.lstsq(a, b)
    return x[0]


class HFD(ApplyManyTransform):
    """
    Higuchi fractal dimension per-channel

    Implementation derived from reading:
    http://arxiv.org/pdf/0804.3361.pdf
    F.S. Bao, D.Y.Lie,Y.Zhang,"A new approach to automated epileptic diagnosis using EEG
    and probabilistic neural network",ICTAI'08, pp. 482-486, 2008.
    """
    def __init__(self, kmax):
        self.kmax = kmax

    def get_name(self):
        return 'hfd-%d' % self.kmax

    def apply_one(self, data, meta=None):
        return to_np_array([hfd(ch, self.kmax) for ch in data])


class Diff(ApplyManyTransform):
    """
    Wrapper for np.diff
    """
    def __init__(self, order):
        self.order = order

    def get_name(self):
        return 'diff-%d' % self.order

    def apply_one(self, data, meta=None):
        return np.diff(data, n=self.order, axis=data.ndim-1)


class SpectralEntropy(ApplyManyTransform):
    """
    Calculates Shannon entropy between the given frequency ranges.
    e.g. The probability density function of FFT magnitude is calculated, then
    given range [1,2,3], Shannon entropy is calculated between 1hz and 2hz, 2hz and 3hz
    in this case giving 2 values per channel.

    NOTE(mike): Input for this transform must be from (FFT(), Magnitude())
    """
    def __init__(self, freq_ranges, flatten=True):
        self.freq_ranges = freq_ranges
        self.flatten = flatten

    def get_name(self):
        return 'spec-ent-%s%s' % ('-'.join([str(f) for f in self.freq_ranges]), '-nf' if not self.flatten else '')

    def apply_one(self, fft_mag, meta):
        num_time_samples = (fft_mag.shape[-1] - 1) * 2 # revert FFT shape change

        X = fft_mag ** 2
        for ch in X:
            ch /= np.sum(ch + 1e-12)

        psd = X # pdf

        out = []

        #[0,1,2] -> [[0,1], [1,2]]
        for start_freq, end_freq in zip(self.freq_ranges[:-1], self.freq_ranges[1:]):
            start_index = np.floor((start_freq / meta.sampling_frequency) * num_time_samples)
            end_index = np.floor((end_freq / meta.sampling_frequency) * num_time_samples)
            selected = psd[:, start_index:end_index]

            entropies = - np.sum(selected * np.log2(selected + 1e-12), axis=selected.ndim-1) / np.log2(end_index - start_index)
            if self.flatten:
                out.append(entropies.ravel())
            else:
                out.append(entropies)

        if self.flatten:
            return np.concatenate(out)
        else:
            return to_np_array(out)


class PIBSpectralEntropy(ApplyManyTransform):
    """
    Similar to the calculations in SpectralEntropy transform, but instead power-in-band
    is calculated over the given freq_ranges, finally Shannon entropy is calculated on that.
    The output is a single entropy value per-channel.

    NOTE(mike): Input for this transform must be from (FFT(), Magnitude())
    """
    def __init__(self, freq_ranges):
        self.freq_ranges = freq_ranges

    def get_name(self):
        return 'pib-spec-ent-%s' % '-'.join([str(f) for f in self.freq_ranges])

    def apply_one(self, data, meta=None):
        num_channels = data.shape[0]
        num_time_samples = float((data.shape[-1] - 1) * 2) # revert FFT shape change

        def norm(X):
            for ch in X:
                ch /= np.sum(ch + 1e-12)
            return X

        psd = data ** 2
        psd = norm(psd)

        # group into bins
        def binned_psd(psd, out):
            prev = freq_ranges[0]
            for i, cur in enumerate(freq_ranges[1:]):
                prev_index = np.floor((prev / meta.sampling_frequency) * num_time_samples)
                cur_index = np.floor((cur / meta.sampling_frequency) * num_time_samples)
                out[i] = np.sum(psd[prev_index:cur_index])
                prev = cur

        freq_ranges = self.freq_ranges
        out = np.empty((num_channels, len(freq_ranges) - 1,))
        for ch in range(num_channels):
            binned_psd(psd[ch], out[ch])

        psd_per_bin = norm(out)

        def entropy_per_channel(psd):
            entropy_components = psd * np.log2(psd + 1e-12)
            entropy = -np.sum(entropy_components) / np.log2(psd.shape[-1])
            return entropy

        out = np.empty((num_channels,))
        for i, ch in enumerate(psd_per_bin):
            out[i] = entropy_per_channel(ch)

        return out


class FreqBinning(ApplyManyTransform):
    """
    Given spectral magnitude data, select a range of bins, and then choose a consolidation function
    to use to calculate each bin. The sum can be used, or the mean, or the standard deviation.

    NOTE(mike): Input for this transform must be from (FFT(), Magnitude())
    """
    def __init__(self, freq_ranges, func=None):
        self.freq_ranges = freq_ranges
        assert func is None or func in ('sum', 'mean', 'std')
        self.func = func

    def get_name(self):
        return 'fbin%s%s' % ('' if self.func is None else '-' + self.func, '-' + '-'.join([str(f) for f in self.freq_ranges]))

    def apply_one(self, X, meta):
        num_channels = X.shape[0]
        num_time_samples = (X.shape[-1] - 1) * 2 # revert FFT shape change

        if self.func == 'mean':
            func = np.mean
        elif self.func == 'std':
            func = np.std
        else:
            func = np.sum

        # group into bins
        def binned_freq(data, out):
            prev = freq_ranges[0]
            for i, cur in enumerate(freq_ranges[1:]):
                prev_index = np.floor((prev / meta.sampling_frequency) * num_time_samples)
                cur_index = np.floor((cur / meta.sampling_frequency) * num_time_samples)
                out[i] = func(data[prev_index:cur_index])
                prev = cur

        freq_ranges = self.freq_ranges
        out = np.empty((num_channels, len(freq_ranges) - 1,))
        for ch in range(num_channels):
            binned_freq(X[ch], out[ch])

        return out


class AR(ApplyManyTransform):
    """
    Auto-regressive model as suggested by:
    http://hdl.handle.net/1807/33224
    https://tspace.library.utoronto.ca/bitstream/1807/33224/1/Green_Adrian_CA_201211_MASc_thesis.pdf

    It is suggested to use a model order of 8.
    """
    def __init__(self, order):
        self.order = order

    def get_name(self):
        return 'ar%d' % self.order

    def calc_for_ch(self, ch):
        ar_coeffs, dnr, reflection_coeffs = spectrum.aryule(ch, self.order)
        return np.abs(ar_coeffs)

    def apply_one(self, X, meta):
        return np.concatenate([self.calc_for_ch(ch) for ch in X], axis=0)


class SubMean:
    """
    For each feature, subtract from each channel the mean across all channels.
    This is to perform average reference montage.
    """
    def get_name(self):
        return 'subm'

    def apply(self, X, meta):
        assert X.ndim == 2
        X -= X.mean(axis=0)
        return X


def index_for_hz(X, hz, sampling_frequency):
    return int((float(hz) / sampling_frequency) * X.shape[-1])


class Preprocess:
    """
    Data preprocessing stage to normalize the data across all patients.
    Data that has not had average reference montage applied needs it applied.
    """
    def get_name(self):
        return 'pp'

    def apply(self, X, meta):
        # NOTE(mike): Patient 1 and 2 have not subtracted the average reference from their raw data
        # whereas Dogs 1 to 5 have. So bring these two patients into line to normalize the preprocessing
        # across ALL patients.
        if meta.target in ('Patient_1', 'Patient_2'):
            X = SubMean().apply(X, meta)
        return X


class PhaseSynchrony(ApplyManyTransform):
    """
    Calculate phase synchrony between channels using Hilbert transform and Shannon entropy.

    Method described in:
    http://www.researchgate.net/publication/222567264_Comparison_of_Hilbert_transform_and_wavelet_methods_for_the_analysis_of_neuronal_synchrony/links/0deec52baa808a3812000000
    Le Van Quyen M, Foucher J, Lachaux J-P, Rodriguez E, Lutz A, Martinerie JM, Varela FJ (2001)
        Comparison of Hilbert transform and wavelet methods for the analysis of neural synchrony.
        J Neurosci Methods 111:83-98

    NOTE(mike): This seemed to work well in cross-validation, but I never got an increased
    on the leaderboard.
    """
    def __init__(self, with_eigen=False, with_raw=True):
        assert with_eigen or with_raw
        self.with_raw = with_raw
        self.with_eigen = with_eigen

    def get_name(self):
        return 'phase-synchrony-%s%s' % ('-eigen' if self.with_eigen else '', '-noraw' if not self.with_raw else '')

    def apply_one(self, X, meta):
        h = X + (1j * hilbert(X))
        phase = np.angle(h)

        num_bins = int(np.exp(0.626 + 0.4 * np.log(X.shape[-1] - 1)))
        Hmax = np.log(num_bins)

        num_channels = X.shape[0]
        if self.with_eigen:
            M = np.ones((num_channels, num_channels), dtype=np.float64)
        out = np.empty((num_channels * (num_channels - 1) / 2,), dtype=np.float64)
        count = 0
        for i in range(num_channels):
            for j in range(i + 1, num_channels):
                ch1_phase = phase[i]
                ch2_phase = phase[j]

                phase_diff = np.mod(np.abs(ch1_phase - ch2_phase), np.pi * 2.0)

                # convert phase_diff into a pdf of num_bins
                hist = np.histogram(phase_diff, bins=num_bins)[0]
                pdf = hist.astype(np.float64) / np.sum(hist)

                H = np.sum(pdf * np.log(pdf + 1e-12))

                p = (H + Hmax) / Hmax

                if self.with_eigen:
                    M[i][j] = p
                    M[j][i] = p
                out[count] = p
                count += 1

        if self.with_eigen:
            eigen = Eigenvalues().apply_one(M)

        if self.with_eigen and self.with_raw:
            return np.concatenate((out, eigen))

        if self.with_eigen:
            return eigen
        else:
            return out

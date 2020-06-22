from __future__ import print_function, division
import numpy as np # manipulate data
import scikits.audiolab as al # play sounds, only for python 2
from scipy import signal # window signal

fs = 44100  # sampling rate Hz

### building blocks

def play(f, fs):
    """TODO: filter high frequencies or raise error when f > 800 Hz"""
    # if spectrum(f)
    return al.play(f, fs)


def sinewave(freq, time):
    """Creates sine wave"""
    y = np.sin(2 * np.pi * freq * time)
    w = signal.tukey(len(y))  # window the signal
    return y * w


def notesInSeries(freqs, time, synth=None):
    """play notes with frequencies freqs, each with a duration specified by time
    freqs: ndarray
        frequencies of the sine waves
    time: ndarray
        time
    synth: callable
        wave sythesizer, takes frequency and time
        default sinewave
    """
    if synth is None: # no decay of the harmonics
        synth = sinewave

    n = len(freqs)
    m = len(time)
    y = np.zeros(n*m)
    print(n, len(y))
    for i, freq in enumerate(freqs):
        yi = synth(freq, time)
        y[i*m:(i+1)*m] = yi

    return y

def harmonic_i2dec(freq, time, n=5):
    """sine wave with n harmonics with quadratic decay
    time: ndarray"""
    quad_dec = np.arange(1, n+1)**2

    return harmonic(freq, time, n=n, decay=quad_dec)

def harmonic(freq, time, n=5, decay=None):
    """sine wave with n harmonics
    freq: num
        fundamental frequency
    time: ndarray
        time sampled
    n: number of harmonics
    decay: ndarray of size n
        Decay of the harmonics
        default None means no decay"""

    if decay is None: # no decay of the harmonics
        decay = np.ones(n)

    #print(decay)
    y = np.zeros_like(time)

    for i in np.arange(1, n + 1):
        y += np.sin(2 * np.pi * freq * i * time) / decay[i-1]
        #print(decay[i-1])
    w = signal.tukey(len(y))  # window the signal
    return y * w


def experimental_harmony(freqs, time, decay=None):
    """sine waves with specific in frequencies (freqs)
    freqs: ndarray
        frequencies of the sine waves
    time: ndarray"""
    n = len(freqs)

    if decay is None: # no decay of the harmonics
        decay = np.ones(n)

    y = np.zeros_like(time)
    for i, freq in enumerate(freqs):
        y += np.sin(2 * np.pi * freq * time) / decay[i-1]
    w = signal.tukey(len(y))  # window the signal
    return y * w


def time_arr(tf, fs):
    """creates a time array with duration tf and sampling rate fs"""
    return np.linspace(0, tf, fs * tf)


def _round2int(x):
    return int(np.round(x))


round2int = np.vectorize(_round2int)


def playwave(f, tf=1, timber=harmonic):
    """plays sine wave with frequency f and duration tf"""
    t = time_arr(tf, fs)
    y = timber(f, t)
    al.play(y, fs=fs)


#### pitch and music theory

def key2frequency(n_key):
    """Returns the frequency given the key number"""
    return 440. * 2. ** ((n_key - 49.) / 12.)


keys2frequencies = np.vectorize(key2frequency)


def frequency2key(n_key):
    """Returns the frequency of the n-th key"""
    return 12. * np.log2(f / 440.) + 49.


frequencies2keys = np.vectorize(frequency2key)


def linear_piano_key2frequency(n_key):
    """Simulates the frequencies of a piano with linear intervals between notes
    Calibrate linear piano with A440, the 49th key and A880 the 61th key
    Parameters
    ----------
    n_key: int
        piano key
    Returns
    -------
    f: float
        frequency of n_key in the linear piano
    """
    f = 440. / 12. * n_key + 440. * (1. - 49. / 12.)
    return f


linear_piano_keys2frequencies = np.vectorize(linear_piano_key2frequency)


## Music theory

def majorScaleKeys(n0=49, n_octaves=1):
    """Returns the keys of the major scale starting at n0
    TODO: generalise to more than one octave"""
    intervals = np.array([2, 2, 1, 2, 2, 2, 1])
    intervals_from_key = np.cumsum(intervals)
    keys = np.hstack((np.array([n0]), n0 + intervals_from_key))
    return keys


def majorScaleFreqs(n0=49, n_octaves=1):
    """Returns the frequencies of the major starting at the n0-th key"""
    keys = majorScaleKeys(n0=n0, n_octaves=n_octaves)
    return keys2frequencies(keys)


def linear_majorScaleFreqs(n0=49, n_octaves=1):
    """Returns the frequencies of the major starting at the n0-th key"""
    keys = majorScaleKeys(n0=n0, n_octaves=n_octaves)
    return linear_piano_key2frequency(keys)


def pioanokey2note():
    """TODO returns notes A A# Bb B..."""
    return 0


def note2pianokey(note, octave=None, key=None):
    """return key index
    Parameters
    ----------
    note: str
        musical note Ab3, A3, A#, B, etc
    octave: int,
        default None, infer from the note, A#3 (last character), octave = 3
        if an integer is given, then is assumed that the notes are not passed with the octave.
        TODO: check possible errors here, eg. octave = 3 and note = A#3

    Example: note2pianokey('A#4') --> 50
    """
    assert isinstance(note, str), "note must be str"
    assert len(note) <= 3, "note should at most three characters long"
    if octave is None:
        octave = int(note[-1])
        n = note[:-1]
    else:
        n = note

    assert n[0] in "ABCDEFG", "note should be in A-G"
    assert octave in np.arange(8), "note octave should be int in 1-8"
    # mapping between notes and key numbers
    key2n = dict(zip(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], np.arange(1, 13)))
    key2n.update({'Bb': 11, 'Db': 2, 'Eb': 4, 'Gb': 7, 'Ab': 9})

    return 3 + 12 * (octave - 1) + key2n[n]


def calibrate_piano(note_key1, note_key2):
    """TODO
    Simulates the frequencies of a piano with linear intervals between notes
    Calibrate linear piano with A440, A880
    TODO note_ket = (key, freq)"""
    assert isinstance(note_key1[0], int)
    assert isinstance(note_key2[0], int)
    assert isinstance(note_key1[1], float)
    assert isinstance(note_key2[1], float)
    return 0
    f1, k1 = note_key1
    f2, k2 = note_key2

    f = (f2 - f1) / (k2 - k1) * n_key + b

    return f
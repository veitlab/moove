# utils/audio_utils.py
import numpy as np


def seconds_to_index(seconds, sample_rate):
    """Convert time in seconds to index based on the sample rate."""
    index_size = int(seconds * sample_rate // 1000)
    return index_size


def seconds_to_chunk_index(seconds, chunk_size, sample_rate):
    """Convert time in seconds to chunk index based on the sample rate and chunk size."""
    index_size = (seconds * sample_rate) // chunk_size
    return index_size


def index_to_seconds(index, chunk_size, sample_rate):
    """Convert index to time in seconds based on the chunk size and sample rate."""
    seconds = (index * chunk_size) / sample_rate
    return seconds


def decibel(x, xref=1.0):
    """Convert amplitude values to decibel (dB)."""
    x_copy = np.copy(x)
    x_copy[x_copy < 1e-10] = 1e-10  # Avoid log(0) by setting a minimum threshold
    db = 20.0 * np.log10(x_copy / xref)
    return db

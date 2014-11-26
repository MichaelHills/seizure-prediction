import sklearn.utils

def collect_sequence_ranges(sequences):
    assert len(sequences) > 0
    seq_starts = [0]
    prev = sequences[0]
    for i, seq in enumerate(sequences[1:]):
        if seq != prev + 1:
            seq_starts.append(i + 1)
        prev = seq

    seq_ranges = []
    prev_start = seq_starts[0]
    for start in seq_starts[1:]:
        seq_ranges.append((prev_start, start))
        prev_start = start

    seq_ranges.append((prev_start, len(sequences)))

    return seq_ranges

def collect_sequence_ranges_from_meta(meta, shuffle=True):
    sequences = meta.sequence
    seq_ranges = collect_sequence_ranges(sequences)
    if shuffle:
        seq_ranges = sklearn.utils.shuffle(seq_ranges, random_state=2)
    return seq_ranges

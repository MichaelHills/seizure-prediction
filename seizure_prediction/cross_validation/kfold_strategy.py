import numpy as np
import sklearn
from seizure_prediction.cross_validation.sequences import collect_sequence_ranges_from_meta

class KFoldStrategy:
    """
    Create a k-fold strategy focused on preictal segments. The idea is to create a small number of folds
    that maximise coverage of the training set. Small number of folds as to keep performance in check.
    If there are 3 preictal sequences, then do 3 folds of (0,1), (0,2), (1,2). If there are 6 sequences,
    do 3 folds (0,1), (2,3), (4,5). The sequences are shuffled before being allocated to folds.

    However, interictal sequences are partitioned randomly as there are a lot more of them that random
    should more or less be fine.
    """

    def get_name(self):
        return 'kfold'

    def get_folds(self, preictal_meta):
        """
        :param preictal_meta: metadata from preictal segments
        :return: iterable of fold numbers to pass to split_train_cv
        """
        num_seqs = len(collect_sequence_ranges_from_meta(preictal_meta))
        assert num_seqs >= 2
        if num_seqs <= 2:
            num_folds = 2
        elif num_seqs <= 6:
            num_folds = 3
        else:
            num_folds = num_seqs / 2

        return xrange(num_folds)

    def get_sequence_ranges(self, meta, fold_number, interictal=False, shuffle=True):
        seq_ranges = collect_sequence_ranges_from_meta(meta, shuffle=shuffle)
        num_seqs = len(seq_ranges)

        # calculate the split numbers for a fold
        def get_num_train_seqs(num_seqs):
            if num_seqs <= 3:
                return 2
            else:
                return 3

        if interictal:
            interictal_ratio = 0.8 if num_seqs <= 20 else 0.4
            train_ranges, cv_ranges = sklearn.cross_validation.train_test_split(seq_ranges, train_size=interictal_ratio, random_state=fold_number)
        else:
            train_size = get_num_train_seqs(num_seqs)
            if num_seqs == 3:
                combinations = [[0, 1], [0, 2], [1, 2]]
            else:
                first_pass = [range(i, i + train_size) for i in range(0, num_seqs, train_size) if (i + train_size) <= num_seqs]
                remainder = num_seqs % train_size
                if remainder == 0:
                    gap = []
                else:
                    seq = range(num_seqs - remainder, num_seqs)
                    needed = train_size - remainder
                    gap_fillers = [i * train_size for i in range(needed)]
                    gap_fillers = [x for x in gap_fillers if x < num_seqs]
                    # print 'gf', gap_fillers
                    if len(gap_fillers) < train_size:
                        gap_fillers = [i * (train_size-1) for i in range(needed)]
                        gap_fillers = [x for x in gap_fillers if x < num_seqs]
                    gap = [gap_fillers + seq]
                second_pass = [range(i, i + train_size**2, train_size) for i in range(num_seqs)]
                second_pass = [x for x in second_pass if len(x) == train_size and x < num_seqs]
                third_pass = [range(i, i + train_size) for i in range(1, num_seqs, train_size) if (i + train_size) <= num_seqs]
                # third_pass = [range(i, i + train_size) for i in range(2, num_seqs, train_size) if (i + train_size) < num_seqs]
                combinations = first_pass + gap + second_pass + third_pass
            indices = combinations[fold_number]
            # print 'indices', indices
            train_ranges = [seq_ranges[i] for i in indices]
            cv_ranges = np.delete(seq_ranges, indices, axis=0)

        return train_ranges, cv_ranges

    def split_train_cv(self, data, meta, fold_number, interictal=False):
        train_ranges, cv_ranges = self.get_sequence_ranges(meta, fold_number, interictal)

        train_data = []
        for start, end in train_ranges:
            train_data.append(data[start:end])
        train_data = np.concatenate(train_data, axis=0)

        cv_data = []
        for start, end in cv_ranges:
            cv_data.append(data[start:end])
        cv_data = np.concatenate(cv_data, axis=0)

        return train_data, cv_data

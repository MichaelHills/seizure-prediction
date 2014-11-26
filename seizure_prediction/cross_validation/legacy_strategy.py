import numpy as np
import sklearn.cross_validation
from seizure_prediction.cross_validation.sequences import collect_sequence_ranges_from_meta


class LegacyStrategy:
    """
    Hand-picked random folds maintaining sequence integrity with 80% train/cv split.
    See k_fold_strategy for docs on each method.
    """

    def get_name(self):
        return 'legacy'

    def get_folds(self, preictal_meta):
        # hand-picked on my system to give a nice spread when num_sequences = 3,
        # i.e. (0, 1), (0, 2), (1, 2) when using 3 folds
        # The new way is to use k_fold.py instead of this
        return [8, 11, 14]

    def get_sequence_ranges(self, meta, fold_number, interictal=None, shuffle=None):
        train_size = 0.8
        seq_ranges = collect_sequence_ranges_from_meta(meta, shuffle=False)
        return sklearn.cross_validation.train_test_split(seq_ranges, train_size=train_size, random_state=fold_number)

    def split_train_cv(self, data, meta, fold_number, interictal=False):

        train_ranges, cv_ranges = self.get_sequence_ranges(meta, fold_number, interictal=interictal)

        train_data = []
        for start, end in train_ranges:
            train_data.append(data[start:end])
        train_data = np.concatenate(train_data, axis=0)

        cv_data = []
        for start, end in cv_ranges:
            cv_data.append(data[start:end])
        cv_data = np.concatenate(cv_data, axis=0)

        return train_data, cv_data

#!/usr/bin/env python2.7

import numpy as np

from seizure_prediction.cross_validation.kfold_strategy import KFoldStrategy
from seizure_prediction.cross_validation.legacy_strategy import LegacyStrategy
from seizure_prediction.cross_validation.sequences import collect_sequence_ranges
from seizure_prediction.pipeline import Pipeline, InputSource
from seizure_prediction.settings import load_settings
from seizure_prediction.tasks import load_pipeline_data


targets = [
    'Dog_1',
    'Dog_2',
    'Dog_3',
    'Dog_4',
    'Dog_5',
    'Patient_1',
    'Patient_2'
]

class Zero:
    def get_name(self):
        return 'zero'

    def apply(self, X, meta):
        return np.zeros(list(X.shape[:-1]) + [1])

settings = load_settings()
pipeline = Pipeline(InputSource(), Zero())

strategies = [
    LegacyStrategy(),
    KFoldStrategy(),
]

for strategy in strategies:
    print 'Strategy', strategy.get_name()
    for target in targets:
        _, preictal_meta = load_pipeline_data(settings, target, 'preictal', pipeline, check_only=False, quiet=True, meta_only=True)
        # _, interictal_meta = load_pipeline_data(settings, target, 'interictal', pipeline, check_only=False, quiet=True, meta_only=True)
        fold_numbers = strategy.get_folds(preictal_meta)
        data = np.arange(0, preictal_meta.X_shape[0]).astype(np.int)
        sequence_ranges = collect_sequence_ranges(preictal_meta.sequence)
        print '%s\n%d folds from %d sequences %s' % (target, len(fold_numbers), len(sequence_ranges), sequence_ranges)
        for fold_number in fold_numbers:
            train_folds, cv_folds = strategy.get_sequence_ranges(preictal_meta, fold_number, interictal=False, shuffle=False)
            print [list(f) for f in train_folds]
        print


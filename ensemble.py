#!/usr/bin/env python2.7

from multiprocessing import Pool
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from seizure_prediction.classifiers import make_svm, make_simple_lr, make_lr
from seizure_prediction.feature_selection import generate_feature_masks
from seizure_prediction.fft_bins import *
from seizure_prediction.pipeline import Pipeline, FeatureConcatPipeline, InputSource
from seizure_prediction.scores import get_score_summary, print_results
from seizure_prediction.tasks import make_csv_for_target_predictions, write_submission_file, \
    cross_validation_score, check_training_data_loaded, check_test_data_loaded, make_submission_predictions
from seizure_prediction.transforms import Windower, Correlation, FreqCorrelation, FFT, \
    Magnitude, PIBSpectralEntropy, Log10, FreqBinning, FlattenChannels, Preprocess, HFD, PFD, Hurst
from seizure_prediction.settings import load_settings
from main import run_prepare_data_for_cross_validation


def run_make_submission(settings, targets_and_pipelines, split_ratio):
    pool = Pool(settings.N_jobs)
    for i, (target, pipeline, feature_masks, classifier, classifier_name) in enumerate(targets_and_pipelines):
        for j, feature_mask in enumerate(feature_masks):
            progress_str = 'T=%d/%d M=%d/%d' % (i+1, len(targets_and_pipelines), j+1, len(feature_masks))
            pool.apply_async(make_submission_predictions, [settings, target, pipeline, classifier, classifier_name],
                {'feature_mask': feature_mask, 'progress_str': progress_str, 'quiet': True})
    pool.close()
    pool.join()

    guesses = ['clip,preictal']
    num_masks = None
    classifier_names = []
    for target, pipeline, feature_masks, classifier, classifier_name in targets_and_pipelines:
        classifier_names.append(classifier_name)
        if num_masks is None:
            num_masks = len(feature_masks)
        else:
            assert num_masks == len(feature_masks)

        test_predictions = []

        for feature_mask in feature_masks:
            data = make_submission_predictions(settings, target, pipeline, classifier, classifier_name, feature_mask=feature_mask)
            test_predictions.append(data.mean_predictions)

        predictions = np.mean(test_predictions, axis=0)
        guesses += make_csv_for_target_predictions(target, predictions)

    output = '\n'.join(guesses)
    write_submission_file(settings, output, 'ensemble n=%d split_ratio=%s' % (num_masks, split_ratio), None, str(classifier_names), targets_and_pipelines)


def run_prepare_data(settings, targets, pipelines, train=True, test=False, quiet=False):
    for pipeline in pipelines:
        for target in targets:
            print 'Preparing data for', target
            if train:
                check_training_data_loaded(settings, target, pipeline, quiet=quiet)
            if test:
                check_test_data_loaded(settings, target, pipeline, quiet=quiet)


def run_cross_validation(settings, targets, pipelines, mask_range, split_ratios, classifiers):
    pool = Pool(settings.N_jobs)
    for i, pipeline in enumerate(pipelines):
        for j, (classifier, classifier_name) in enumerate(classifiers):
            for k, target in enumerate(targets):
                pool.apply_async(cross_validation_score, [settings, target, pipeline, classifier, classifier_name], {'quiet': True})
                for split_num, split_ratio in enumerate(split_ratios):
                    masks = generate_feature_masks(settings, target, pipeline, np.max(mask_range), split_ratio, random_state=0, quiet=True)
                    for mask_num, mask in enumerate(masks):
                        progress_str = 'P=%d/%d C=%d/%d T=%d/%d S=%d/%d M=%d/%d' % (i+1, len(pipelines), j+1, len(classifiers), k+1, len(targets), split_num+1, len(split_ratios), mask_num+1, len(masks))
                        cross_validation_score(settings, target, pipeline, classifier, classifier_name, feature_mask=mask, quiet=True, return_data=False, pool=pool, progress_str=progress_str)
    pool.close()
    pool.join()
    print 'Finished cross validation mp'

    summaries = []
    for p_num, pipeline in enumerate(pipelines):
        for classifier, classifier_name in classifiers:
            scores_full = []
            scores_masked = [[[] for y in mask_range] for x in split_ratios]
            for i, target in enumerate(targets):
                run_prepare_data_for_cross_validation(settings, [target], [pipeline], quiet=True)
                data = cross_validation_score(settings, target, pipeline, classifier, classifier_name, pool=None, quiet=True)
                scores_full.append(data.mean_score)

                for split_index, split_ratio in enumerate(split_ratios):
                    masks = generate_feature_masks(settings, target, pipeline, np.max(mask_range), split_ratio, random_state=0, quiet=True)
                    for mask_index, num_masks in enumerate(mask_range):
                        predictions = []
                        y_cvs = None
                        for mask in masks[0:num_masks]:
                            data = cross_validation_score(settings, target, pipeline, classifier, classifier_name, feature_mask=mask, pool=None, quiet=True)
                            predictions.append(data.mean_predictions)
                            if y_cvs is None:
                                y_cvs = data.y_cvs
                            else:
                                for y_cv_1, y_cv_2 in zip(y_cvs, data.y_cvs):
                                    assert np.alltrue(y_cv_1 == y_cv_2)

                        predictions = np.mean(predictions, axis=0)
                        scores = [roc_auc_score(y_cv, p) for p, y_cv in zip(predictions, y_cvs)]
                        score = np.mean(scores)
                        scores_masked[split_index][mask_index].append(score)

            summary = get_score_summary('%s p=%d full' % (classifier_name, p_num), scores_full)
            summaries.append((summary, np.mean(scores_full)))
            for split_index, split_ratio in enumerate(split_ratios):
                for mask_index, num_masks in enumerate(mask_range):
                    scores = scores_masked[split_index][mask_index]
                    summary = get_score_summary('%s p=%d split_ratio=%s masks=%d' % (classifier_name, p_num, split_ratio, num_masks), scores)
                    summaries.append((summary, np.mean(scores)))
                    print summary

    print_results(summaries)


def main():
    settings = load_settings()

    pipelines = [
        FeatureConcatPipeline(
            Pipeline(InputSource(), Preprocess(), Windower(75), Correlation('none')),
            Pipeline(InputSource(), Preprocess(), Windower(75), FreqCorrelation(1, None, 'none')),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), FreqBinning(winning_bins, 'mean'), Log10(), FlattenChannels()),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([0.25, 1, 1.75, 2.5, 3.25, 4, 5, 8.5, 12, 15.5, 19.5, 24])),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([0.25, 2, 3.5, 6, 15, 24])),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([0.25, 2, 3.5, 6, 15])),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([0.25, 2, 3.5])),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([6, 15, 24])),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([2, 3.5, 6])),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([3.5, 6, 15])),
            Pipeline(InputSource(), Preprocess(), Windower(75), HFD(2)),
            Pipeline(InputSource(), Preprocess(), Windower(75), PFD()),
            Pipeline(InputSource(), Preprocess(), Windower(75), Hurst()),
        ),
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

    classifiers = [
        make_svm(gamma=0.0079, C=2.7),
        make_svm(gamma=0.0068, C=2.0),
        make_svm(gamma=0.003, C=150.0),
        make_lr(C=0.04),
        make_simple_lr(),
    ]


    make_submission = len(sys.argv) >= 2 and sys.argv[1] == 'submission'
    do_cv = not make_submission

    if do_cv:
        mask_range = [3]
        split_ratios = [0.4, 0.525, 0.6]
        run_prepare_data_for_cross_validation(settings, targets, pipelines)
        run_cross_validation(settings, targets, pipelines, mask_range, split_ratios, classifiers)

    if make_submission:
        num_masks = 10
        split_ratio = 0.525
        classifiers = [
            # make_svm(gamma=0.0079, C=2.7),
            make_svm(gamma=0.0068, C=2.0),
            # make_svm(gamma=0.003, C=150.0),
            # make_lr(C=0.04),
            # make_simple_lr(),
        ]

        targets_and_pipelines = []
        pipeline = pipelines[0]
        for classifier, classifier_name in classifiers:
            for i, target in enumerate(targets):
                run_prepare_data(settings, [target], [pipeline], test=True)
                feature_masks = generate_feature_masks(settings, target, pipeline, num_masks, split_ratio, random_state=0, quiet=True)
                targets_and_pipelines.append((target, pipeline, feature_masks, classifier, classifier_name))

        run_make_submission(settings, targets_and_pipelines, split_ratio)


if __name__ == "__main__":
    main()

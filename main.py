#!/usr/bin/env python2.7

from multiprocessing import Pool
import sys

import numpy as np

from seizure_prediction.classifiers import make_svm, make_lr, make_simple_lr
from seizure_prediction.cross_validation.kfold_strategy import KFoldStrategy
from seizure_prediction.cross_validation.legacy_strategy import LegacyStrategy
from seizure_prediction.pipeline import Pipeline, FeatureConcatPipeline, InputSource
from seizure_prediction.scores import get_score_summary, print_results
from seizure_prediction.tasks import make_submission_csv, cross_validation_score, \
    write_submission_file, check_training_data_loaded, check_test_data_loaded
from seizure_prediction.transforms import FFT, Magnitude, Log10, Windower, \
    Correlation, FreqCorrelation, FlattenChannels, \
    Hurst, PFD, PIBSpectralEntropy, FreqBinning, HFD, Preprocess
from seizure_prediction.settings import load_settings
from seizure_prediction.fft_bins import *


# cross_validation_strategy = KFoldStrategy()
cross_validation_strategy = LegacyStrategy()


def run_prepare_data_for_cross_validation(settings, targets, pipelines, quiet=False):
    if not quiet: print '\n'.join([p.get_name() for p in pipelines])
    for i, pipeline in enumerate(pipelines):
        for j, target in enumerate(targets):
            if not quiet: print 'Running prepare data', 'P=%d/%d T=%d/%d' % (i+1, len(pipelines), j+1, len(targets))
            check_training_data_loaded(settings, target, pipeline)


def run_prepare_data_for_submission(settings, targets, pipelines):
    for pipeline in pipelines:
        for target in targets:
            print 'Running %s pipeline %s' % (target, pipeline.get_name())
            check_training_data_loaded(settings, target, pipeline)
            check_test_data_loaded(settings, target, pipeline)


def run_cross_validation(settings, targets, classifiers, pipelines):
    print 'Cross-validation task'
    print 'Targets', ', '.join(targets)
    print 'Pipelines:\n ', '\n  '.join([p.get_name() for p in pipelines])
    print 'Classifiers', ', '.join([c[1] for c in classifiers])

    run_prepare_data_for_cross_validation(settings, targets, pipelines)

    # run on pool first, then show results after
    pool = Pool(settings.N_jobs)
    for i, pipeline in enumerate(pipelines):
        for j, (classifier, classifier_name) in enumerate(classifiers):
            for k, target in enumerate(targets):
                progress_str = 'P=%d/%d C=%d/%d T=%d/%d' % (i+1, len(pipelines), j+1, len(classifiers), k+1, len(targets))
                cross_validation_score(settings, target, pipeline, classifier, classifier_name,
                    strategy=cross_validation_strategy, pool=pool, progress_str=progress_str, return_data=False, quiet=True)
    pool.close()
    pool.join()

    summaries = []
    best = {}
    for p_num, pipeline in enumerate(pipelines):
        for c_num, (classifier, classifier_name) in enumerate(classifiers):
            mean_scores = []
            median_scores = []
            datas = []
            for target in targets:
                print 'Running %s pipeline %s classifier %s' % (target, pipeline.get_name(), classifier_name)
                data = cross_validation_score(settings, target, pipeline, classifier, classifier_name,
                    strategy=cross_validation_strategy, quiet=True)
                datas.append(data)
                if data.mean_score != data.median_score:
                    print '%.3f (mean)' % data.mean_score, data.mean_scores
                    print '%.3f (median)' % data.median_score, data.median_scores
                else:
                    print '%.3f' % data.mean_score
                mean_scores.append(data.mean_score)
                median_scores.append(data.median_score)

                best_score = best.get(target, [0, None, None, None])[0]
                cur_score = max(data.mean_score, data.median_score)
                if cur_score > best_score:
                    best[target] = [cur_score, pipeline, classifier, classifier_name]

            name = 'p=%d c=%d %s mean %s' % (p_num, c_num, classifier_name, pipeline.get_name())
            summary = get_score_summary(name, mean_scores)
            summaries.append((summary, np.mean(mean_scores)))
            print summary
            name = 'p=%d c=%d %s median %s' % (p_num, c_num, classifier_name, pipeline.get_name())
            summary = get_score_summary(name, median_scores)
            summaries.append((summary, np.mean(median_scores)))
            print summary

    print_results(summaries)

    print '\nbest'
    for target in targets:
        pipeline = best[target][1]
        classifier_name = best[target][3]
        print target, best[target][0], classifier_name, pipeline.get_names()


def run_make_submission(settings, targets, classifiers, pipelines):
    print 'Submissions task'
    print 'Targets', ', '.join(targets)
    print 'Pipelines', ', '.join([p.get_name() for p in pipelines])
    print 'Classifiers', ', '.join([c[1] for c in classifiers])

    run_prepare_data_for_submission(settings, targets, pipelines)

    pool = Pool(settings.N_jobs)
    for pipeline in pipelines:
        for classifier, classifier_name in classifiers:
            for target in targets:
                pool.apply_async(make_submission_csv, [settings, target, pipeline, classifier, classifier_name])
    pool.close()
    pool.join()

    use_median_submissions = False

    for pipeline in pipelines:
        for classifier, classifier_name in classifiers:
            guesses_mean = ['clip,preictal']
            guesses_median = ['clip,preictal']
            for target in targets:
                print 'Target %s pipeline %s classifier %s' % (target, pipeline.get_name(), classifier_name)
                predictions_mean, predictions_median = make_submission_csv(settings, target, pipeline, classifier, classifier_name)
                guesses_mean += predictions_mean
                guesses_median += predictions_median

            mean_output = '\n'.join(guesses_mean)
            median_output = '\n'.join(guesses_median)

            out = []
            if use_median_submissions and mean_output != median_output:
                out.append((mean_output, 'mean'))
                out.append((median_output, 'median'))
            else:
                out.append((mean_output, None))

            for guesses, name in out:
                write_submission_file(settings, guesses, name, pipeline, classifier_name)



def main():

    settings = load_settings()

    targets = [
        'Dog_1',
        'Dog_2',
        'Dog_3',
        'Dog_4',
        'Dog_5',
        'Patient_1',
        'Patient_2'
    ]

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

    classifiers = [
        make_svm(gamma=0.0079, C=2.7),
        make_svm(gamma=0.0068, C=2.0),
        make_svm(gamma=0.003, C=150.0),
        make_lr(C=0.04),
        make_simple_lr(),
    ]

    submission_pipelines = [
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

    submission_classifiers = [
        make_simple_lr(),
    ]

    if len(sys.argv) >= 2 and sys.argv[1] == 'submission':
        run_make_submission(settings, targets, submission_classifiers, submission_pipelines)
    else:
        run_cross_validation(settings, targets, classifiers, pipelines)


if __name__ == "__main__":
    main()


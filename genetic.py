#!/usr/bin/env python2.7

import random
from multiprocessing import Pool
import sys

import numpy as np
from deap import creator, base, tools

from seizure_prediction.classifiers import make_svm
from seizure_prediction.cross_validation.legacy_strategy import LegacyStrategy
from seizure_prediction.feature_selection import generate_feature_masks
from seizure_prediction.pipeline import Pipeline, FeatureConcatPipeline, InputSource
from seizure_prediction.scores import get_score_summary, print_results
from seizure_prediction.tasks import load_training_data, make_csv_for_target_predictions, write_submission_file, \
    cross_validation_score, check_training_data_loaded, check_test_data_loaded, make_submission_predictions, \
    calc_feature_mask_string
from seizure_prediction.transforms import Windower, Correlation, FreqCorrelation, FFT, \
    Magnitude, PIBSpectralEntropy, Log10, FreqBinning, FlattenChannels, PFD, HFD, Hurst, Preprocess
from seizure_prediction.settings import load_settings
from main import run_prepare_data_for_cross_validation
from seizure_prediction.fft_bins import *


cross_validation_strategy = LegacyStrategy()


def evaluate_fitness_score(settings, target, pipeline, classifier, classifier_name, quiet, arg):
    individual, best_score = arg
    if np.sum(individual) == 0:
        score = 0.0
    else:
        score = float(cross_validation_score(settings, target, pipeline, classifier, classifier_name,
            strategy=cross_validation_strategy, feature_mask=individual, quiet=True).mean_score)

    if score > best_score:
        if not quiet: print score, np.sum(individual)
    return score,


creator.create("RocAucMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.RocAucMax)


def random_bool(threshold):
    return 1 if random.random() <= threshold else 0


def get_pipeline_data(settings, target, pipeline):
    data = load_training_data(settings, target, pipeline, check_only=False, quiet=True)
    num_features = data.X_train.shape[data.X_train.ndim-1]
    return num_features, data.num_train_segments


def process_target(settings, target, pipeline, classifier, classifier_name, ratio, ngen, quiet, threshold=400):
    # make results repeatable
    random.seed(0)

    num_features, num_training_examples = get_pipeline_data(settings, target, pipeline)

    # Using sub-feature selection for the human patients appears to perform worse than
    # using full feature set. My guess is that perhaps there is not enough training samples
    # for this technique to work effectively. So do not run GA if there are too few training
    # samples. The threshold parameter can be tweaked with more testing.
    if num_training_examples < threshold:
        score = float(cross_validation_score(settings, target, pipeline, classifier, classifier_name,
            strategy=cross_validation_strategy, quiet=True).mean_score)
        return score, [[1] * num_features]

    num_wanted_features = int(num_features * ratio)
    if not quiet: print 'ratio', ratio
    if not quiet: print 'num features', num_features
    if not quiet: print 'num wanted features', num_wanted_features

    if not quiet: print target, classifier_name

    pool = Pool(settings.N_jobs)

    toolbox = base.Toolbox()
    toolbox.register("map", pool.map)
    toolbox.register("attr_bool", random_bool, ratio)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_fitness_score, settings, target, pipeline, classifier, classifier_name, quiet)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=30)
    CXPB, MUTPB, NGEN = 0.5, 0.2, ngen

    best_score = 0
    best_feature_mask = None
    all_feature_masks = {}

    # Evaluate the entire population
    if not quiet: print 'evaluating pop %d' % len(pop)
    fitnesses = toolbox.map(toolbox.evaluate, [(ind, 1.0) for ind in pop])
    if not quiet: print 'done evaluating'

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        all_feature_masks[calc_feature_mask_string(ind)] = (list(ind), fit[0])

    # calc first best
    fits = [ind.fitness.values[0] for ind in pop]
    best_index = np.argmax(fits)
    score = fits[best_index]
    if score > best_score:
        best_score = score
        best_feature_mask = pop[best_index]
        if not quiet: print 'new best', best_score, np.sum(best_feature_mask)

    # Begin the evolution
    for g in range(NGEN):
        if not quiet: print("-- %s: Generation %i --" % (target, g))

        # Select the next generation individuals
        offspring = toolbox.select(pop, int(len(pop)))
        # Clone the selected individuals
        offspring = list(toolbox.map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, [(ind, best_score) for ind in invalid_ind])
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            all_feature_masks[calc_feature_mask_string(ind)] = (list(ind), fit[0])

        if not quiet: print("  Evaluated %i individuals (pop size %d)" % (len(invalid_ind), len(offspring)))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        best_index = np.argmax(fits)
        all_f = [np.sum(ind) for ind in pop]
        if not quiet: print '  %s, %s, %s (%d-%d)' % (target, fits[best_index], np.sum(pop[best_index]), np.min(all_f), np.max(all_f))

        length = len(pop)
        mean = sum(fits) / length

        if not quiet: print("  Min %s" % min(fits))
        if not quiet: print("  Max %s" % max(fits))
        if not quiet: print("  Avg %s" % mean)

        score = fits[best_index]
        if score > best_score:
            best_score = score
            best_feature_mask = pop[best_index]
            if not quiet: print 'new best', best_score, np.sum(best_feature_mask)

    if not quiet: print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    if not quiet: print "-- Finished --\n%s\n%s\n%s" % (target, best_ind.fitness.values[0], best_ind)

    pop = list(all_feature_masks.values())
    pop.sort(cmp=lambda x1, x2: cmp(x2[1], x1[1]))
    sorted_pop = [ind for ind, score in pop]
    print target, 'best', pop[0][1], 'worst', pop[-1][1]

    return best_score, sorted_pop


def run_make_submission(settings, targets_and_pipelines, classifier, classifier_name):
    pool = Pool(settings.N_jobs)
    for i, (target, pipeline, feature_masks) in enumerate(targets_and_pipelines):
        for j, feature_mask in enumerate(feature_masks):
            progress_str = 'T=%d/%d M=%d/%d' % (i+1, len(targets_and_pipelines), j+1, len(feature_masks))
            pool.apply_async(make_submission_predictions, [settings, target, pipeline, classifier, classifier_name], {'feature_mask': feature_mask, 'quiet': True, 'progress_str': progress_str})
    pool.close()
    pool.join()

    guesses = ['clip,preictal']
    for target, pipeline, feature_masks in targets_and_pipelines:
        test_predictions = []

        for feature_mask in feature_masks:
            data = make_submission_predictions(settings, target, pipeline, classifier, classifier_name, feature_mask=feature_mask)
            test_predictions.append(data.mean_predictions)

        predictions = np.mean(test_predictions, axis=0)
        guesses += make_csv_for_target_predictions(target, predictions)

    output = '\n'.join(guesses)
    submission_targets_and_pipelines = [(target, pipeline, feature_masks, classifier, classifier_name)
        for target, pipeline, feature_masks in targets_and_pipelines]
    write_submission_file(settings, output, None, None, classifier_name, submission_targets_and_pipelines)


def run_prepare_data(settings, targets_and_pipelines, train=True, test=False):
    for target, pipeline, feature_masks in targets_and_pipelines:
        if train:
            check_training_data_loaded(settings, target, pipeline)
        if test:
            check_test_data_loaded(settings, target, pipeline)


def extract_masks_for_pipeline_and_masks(settings, target, pipeline, masks):
    outs = [{} for mask in masks]
    offset = 0
    for p in pipeline.get_pipelines():
        num_features, _ = get_pipeline_data(settings, target, p)
        for i, mask in enumerate(masks):
            p_mask = mask[offset:offset + num_features]
            outs[i][p.get_name()] = p_mask
        offset += num_features
    for mask in masks:
        assert offset == len(mask)
    return outs


def merge_dicts(*dicts):
    x = dicts[0].copy()
    for d in dicts[1:]:
        x.update(d)
    return x


def get_submission_targets_and_masks(settings, targets, classifier, classifier_name, pipeline_groups, random_pipelines, random_ratio=0.525, ngen=10, limit=2, random_limit=2):
    assert random_limit % limit == 0
    random_multiplier = random_limit / limit
    quiet = True

    random_pipeline = FeatureConcatPipeline(*random_pipelines)

    all_pipelines = []
    all_pipelines.extend(random_pipelines)
    for pg, ratio in pipeline_groups:
        all_pipelines.extend(pg)
    full_pipeline = FeatureConcatPipeline(*all_pipelines)
    run_prepare_data(settings, [(target, full_pipeline, []) for target in targets], test=True)

    def get_pipeline_and_feature_masks(target, pipelines, classifier, classifier_name, ratio, ngen):
        print target, 'fetching GA pipelines', [p.get_name() for p in pipelines]
        pipeline = FeatureConcatPipeline(*pipelines)
        score, best_N = process_target(settings, target, pipeline, classifier, classifier_name, ratio=ratio, ngen=ngen, quiet=quiet)
        return pipeline, best_N

    targets_and_pipelines = []
    for target in targets:
        # NOTE(mike): All this stuff is a bit nasty. It gets the random-masks and the genetic-masks
        # for different pipelines, and then pulls out the mask for each individual pipeline. A single
        # FeatureConcatPipeline is then created to represent all the features, and the masks for each
        # member of the FCP are merged together to form the single feature mask across the whole FCP.

        random_masks = generate_feature_masks(settings, target, random_pipeline, random_limit, random_ratio, random_state=0, quiet=quiet)
        # contains a list of pairs, (pipeline, mask)
        ga_groups = [get_pipeline_and_feature_masks(target, p, classifier, classifier_name, ratio, ngen) for p, ratio in pipeline_groups]
        ga_groups = [(p, masks[0:limit]) for p, masks in ga_groups]

        print target, 'extracting GA per-pipeline masks...'
        # contains a list of mask dictionaries
        ga_dicts = [extract_masks_for_pipeline_and_masks(settings, target, pipeline, masks) for pipeline, masks in ga_groups]
        ga_dicts = [mask_dicts * random_multiplier for mask_dicts in ga_dicts]

        r_dicts = extract_masks_for_pipeline_and_masks(settings, target, random_pipeline, random_masks)
        # this contains a list of dictionaries which maps pipeline names to masks
        # e.g. [r_dicts, ga_dicts0, ga_dicts1, ...]
        zip_group = [r_dicts] + ga_dicts

        print target, 'merging all masks...'
        feature_mask_dicts = [merge_dicts(*x) for x in zip(*zip_group)]

        feature_masks = []
        for feature_mask_dict in feature_mask_dicts:
            mask = []
            for p in full_pipeline.get_pipelines():
                mask.extend(feature_mask_dict[p.get_name()])
            feature_masks.append(mask)

        targets_and_pipelines.append((target, full_pipeline, feature_masks))
    return targets_and_pipelines


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

    # The genetic algorithm will be run individually on each pipeline group
    pipeline_groups = [
        ([
            Pipeline(InputSource(), Preprocess(), Windower(75), PFD()),
        ], 0.55),
        ([
            Pipeline(InputSource(), Preprocess(), Windower(75), Hurst()),
        ], 0.55),
        ([
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([0.25, 1, 1.75, 2.5, 3.25, 4, 5, 8.5, 12, 15.5, 19.5, 24])),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([0.25, 2, 3.5, 6, 15, 24])),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([0.25, 2, 3.5, 6, 15])),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([0.25, 2, 3.5])),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([6, 15, 24])),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([2, 3.5, 6])),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), PIBSpectralEntropy([3.5, 6, 15])),
            Pipeline(InputSource(), Preprocess(), Windower(75), HFD(2)),
        ], 0.55),
    ]

    make_submission = len(sys.argv) >= 2 and sys.argv[1] == 'submission'
    run_ga = not make_submission

    # This classifier is used in the genetic algorithm
    ga_classifier, ga_classifier_name = make_svm(gamma=0.0079, C=2.7)

    if run_ga:
        quiet = False
        summaries = []
        for ngen in [10]:
            for pipelines, ratio in pipeline_groups:
                out = []
                for target in targets:
                    print 'Running target', target
                    run_prepare_data_for_cross_validation(settings, [target], pipelines, quiet=True)
                    pipeline = FeatureConcatPipeline(*pipelines)
                    score, best_N = process_target(settings, target, pipeline, ga_classifier, ga_classifier_name, ratio=ratio, ngen=ngen, quiet=quiet)
                    print target, score, [np.sum(mask) for mask in best_N[0:10]]
                    out.append((target, score, pipeline, best_N))

            scores = np.array([score for _, score, _, _ in out])
            summary = get_score_summary('%s ngen=%d' % (ga_classifier_name, ngen), scores)
            summaries.append((summary, np.mean(scores)))
            print summary

        print_results(summaries)

    if make_submission:
        random_pipelines = [
            Pipeline(InputSource(), Preprocess(), Windower(75), Correlation('none')),
            Pipeline(InputSource(), Preprocess(), Windower(75), FreqCorrelation(1, None, 'none')),
            Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), FreqBinning(winning_bins, 'mean'), Log10(), FlattenChannels()),
        ]

        # These classifiers are used to make the final predictions
        final_classifiers = [
            # make_svm(gamma=0.0079, C=2.7),
            make_svm(gamma=0.0068, C=2.0),
            # make_svm(gamma=0.003, C=150.0),
            # make_lr(C=0.04),
            # make_simple_lr(),
        ]
        targets_and_pipelines = get_submission_targets_and_masks(settings, targets, ga_classifier, ga_classifier_name, pipeline_groups, random_pipelines)
        for classifier, classifier_name in final_classifiers:
            run_make_submission(settings, targets_and_pipelines, classifier, classifier_name)


if __name__ == "__main__":
    main()

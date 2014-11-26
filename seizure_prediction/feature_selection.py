import numpy as np

from seizure_prediction.tasks import load_pipeline_data


# Generate random feature masks using split_ratio as the rough guide to how many features are ON and how many are OFF.
def generate_feature_masks(settings, target, pipeline, num_masks, split_ratio, random_state, threshold=500, quiet=False):
    if not quiet: print target
    def get_pipeline_data(pipeline):
        _, preictal_meta = \
            load_pipeline_data(settings, target, 'preictal', pipeline, check_only=False, meta_only=True, quiet=quiet)
        _, interictal_meta = \
            load_pipeline_data(settings, target, 'interictal', pipeline, check_only=False, meta_only=True, quiet=quiet)
        num_features = preictal_meta.X_shape[-1]
        num_train_segments = preictal_meta.num_segments + interictal_meta.num_segments
        return num_features, num_train_segments

    if len(pipeline.get_pipelines()) == 0:
        return []

    num_features, num_training_examples = get_pipeline_data(pipeline)

    # NOTE(mike): Seemingly some patients benefit from these feature masks and some don't.
    # Currently the only pattern is number of training examples but this may or may not hold
    # true without doing further testing. Some manual testing against public leaderboard showed
    # a negative effect on Patient 1 and 2 but positive effects on Dogs 3 and 4. Dog 1 seemed to
    # have little to no effect and maybe a very slight positive effect on Dog 2.
    if num_training_examples < threshold:
        ratio = 1.0
    else:
        ratio = split_ratio

    if not quiet: print 'num features', num_features
    if not quiet: print 'ratio', ratio
    if not quiet: print 'num wanted features', int(num_features * ratio)

    if ratio == 1.0:
        masks = np.ones((num_masks, num_features))
    else:
        prng = np.random.RandomState(random_state)
        masks = (prng.random_sample((num_masks, num_features)) <= ratio)

    masks = list(masks.astype(np.int))

    if not quiet: print np.shape(masks)
    if not quiet: print 'generated', [np.sum(mask) for mask in masks]
    return list(masks)


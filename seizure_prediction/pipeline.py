
class Pipeline(object):
    """
    A Pipeline is an object representing the data transformations to make
    on the input data, finally outputting extracted features.

    input_source: Where to source the data from, InputSource() for original data or
                  InputSource(some_pipeline) to load the output of a pipeline.
    transforms: List of transforms to apply one by one to the input data.
    """
    def __init__(self, input_source, *transforms):
        self.input_source = input_source
        input_source_pipeline = input_source.get_pipeline()
        self.transforms = transforms
        full_pipeline = [input_source_pipeline] + list(transforms) if input_source_pipeline is not None else transforms
        names = [t.get_name() for t in full_pipeline]
        self.name = 'empty' if len(names) == 0 else '_'.join(names)

    def get_name(self):
        return self.name

    def get_names(self):
        return [self.name]

    def apply(self, data, meta):
        for transform in self.transforms:
            data = transform.apply(data, meta)
        return data

    def get_input_source(self):
        return self.input_source

    def get_pipelines(self):
        return [self]


class FeatureConcatPipeline(object):
    """
    Represents a list of pipelines with their features concatenated together.
    Useful for combining separate feature sets to see if they combine well.
    """
    def __init__(self, *pipelines, **options):
        pipelines = list(pipelines)
        if 'sort' not in options or options['sort'] == True:
            pipelines.sort(lambda x, y: cmp(x.get_name(), y.get_name()))
        self.pipelines = pipelines
        self.names = [p.get_name() for p in pipelines]
        self.name = 'FCP_' + '_cc_'.join(self.names)
        for p in pipelines:
            assert isinstance(p, Pipeline)

    def get_name(self):
        return self.name

    def get_names(self):
        return self.names

    def apply(self, data, meta):
        raise NotImplementedError()

    def get_pipelines(self):
        return self.pipelines

    def get_input_source(self):
        raise NotImplementedError()


class InputSource:
    """
    Wraps a pipeline to represent it as a data-source.
    """
    def __init__(self, *transforms):
        self.pipeline = Pipeline(InputSource(), *transforms) if len(transforms) > 0 else None

    def get_pipeline(self):
        return self.pipeline


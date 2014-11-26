# Seizure Prediction

This repository contains the code I used for the American Epilepsy Society Seizure's
Prediction Challenge on Kaggle.

http://www.kaggle.com/c/seizure-prediction

As a side note this won't generate my exact submission as the randomness was affected
after cleaning up the code. It doesn't score as well which demonstrates the fragility
of my approach. I have also included the linear regression approach as used by
Jonathan Tapson. It makes my genetic algorithm and random feature mask ensembling a
little redundant, hence I use his approach in `main.py`, but demonstrate my own approaches
in `genetic.py` and `ensemble.py`

I discuss further down my genetic algorithm approach and the features I used. Taking a
look at the code might also yield more insights.

You probably need 100-150GB free disk space to run this code.

###Hardware / OS platform used

 * 15" Retina MacBook Pro (Late 2013) 2.7GHz Core i7, 16GB RAM
 * OS X Mavericks
 * 512GB SSD

###Dependencies

####Required

 * Python 2.7 (I used built-in OS X Python 2.7.6)
 * scikit\_learn-0.15.2
 * numpy-1.9.0
 * pandas-0.14.1
 * scipy-0.14.0
 * h5py-2.3.1
 * hdf5 (see http://www.hdfgroup.org/HDF5)
 * deap-1.0

####Optional (to try out various data transforms)

 * spectrum (for auto-regressive model)

### SETTINGS.json

```
{
  "competition-data-dir": "data",
  "data-cache-dir": "data-cache",
  "submission-dir": "submissions",
  "num-jobs": "auto"
}
```

 * `competition-data-dir`: directory containing the downloaded competition data
 * `data-cache-dir`: directory the task framework will store cached data
 * `submission-dir`: directory submissions are written to
 * `num-jobs`: "auto" or integer specifying number of processes to use in multiprocessing Pool

### Getting started

#### Preprocess data into hdf5 format

First place the competition data under ./data/ (or as specified in SETTINGS.json)

```
$ ./mat_to_hdf5.py
Loading data ...
Processing data/Dog_1_preictal.hdf5 ...
Runner 0 processing data/Dog_1/Dog_1_preictal_segment_0001.mat
Runner 1 processing data/Dog_1/Dog_1_preictal_segment_0002.mat
Runner 2 processing data/Dog_1/Dog_1_preictal_segment_0003.mat
Runner 3 processing data/Dog_1/Dog_1_preictal_segment_0004.mat
Runner 4 processing data/Dog_1/Dog_1_preictal_segment_0005.mat
Runner 5 processing data/Dog_1/Dog_1_preictal_segment_0006.mat
Runner 6 processing data/Dog_1/Dog_1_preictal_segment_0007.mat
Runner 7 processing data/Dog_1/Dog_1_preictal_segment_0008.mat
...
```

This took ~38 minutes to run on my machine to process all the patients. After this is done you
can feel free to delete the original matlab files as my code generates hdf5 files to replace them.

All patients have their signals decimated down to 200Hz to save disk space and improve processing times.

#### Run cross-validation with full-features
```
./main.py
```

#### Make a submission
```
./main.py submission
```

This takes ~30 minutes on my machine with an empty data-cache.

### Three build variants (main/ensemble/genetic)

### main.py

This file contains the initial standard setup training per-patient models and not doing any
sub-feature selection. The default selected classifier for submission is linear regression.
A list of classifiers are used in cross-validation to compare scores.

```
./main.py
./main.py submission
```

### ensemble.py

This file contains the ensemble variant, generating N random feature masks, training N models
per-patient, and then averaging those N models predictions. I did not find the cross-validation
to be of much use but I left it in anyway. For submission this lead to better scores than
`main.py` when using SVC with specific parameters `gamma=0.0079` and `C=2.7`. For these
parameters `main.py` would achieve around 0.796 on public LB, and this ensembling approach
would achieve around 0.829. I later learned that using different parameters `gamma=0.003` and
`C=150.0` I could achieve similar scores around 0.829 without any ensembling.

```
./ensemble.py
./ensemble.py submission
```

### genetic.py

This file contains my genetic algorithm approach. This is what I used for my 5th place submission.
However the code as it is right now will not generate my exact submission as I renamed some of the
transforms which changed some orderings and randomness which led to different CV results and
ultimately different selected feature masks. It doesn't score too far off though.

The genetic algorithm starts with population size of 30 and runs for 10 generations. The population
is initialised with random feature masks consisting of roughly 55% features activated and the other
45% masked away. The fitness function is simply CV ROC AUC score.

This is quite slow, taking on the order of 1-2 hours to run. I also ran 3 sets of genetic algorithm,
each using a different subset of the features. I believe this to more or less just be myself optimising
random chance against the public LB.

Other than the 3 feature groups, other features which appeared to not benefit from the genetic algorithm
instead used random feature masks. Two masks were used for each feature group, 2 of the best masks
for each of the GA groups and 2 random masks for the random groups. Again optimising against the
leaderboard, a 52.5% active features ratio was used for the random feature masks.

To be honest this is all a bit of voodoo, and using the linear regression approach more or less makes
all of this a waste of time. Later testing showed that only Dog\_3 and Dog\_4 really benefited from
the sub-feature mask ensembling. Dog\_1 showed little change, Dog\_2 a very minor improvement, I
didn't test Dog\_5. Patient\_1 and Patient\_2 actually always performed worse when using sub-feature
masks whether genetic or random. There was correlation in training sample size and having a benefit
from feature masks, so I didn't use feature masks when the number of training samples was less than
500 (excludes Patient 1 and 2). More testing needs to be done to actually verify that's the right
thing to do.

```
./genetic.py
./genetic.py submission
```

### Features used

 * Time correlation matrix upper right triangle and sorted eigenvalues
 * Frequency correlation matrix upper right triangle and sorted eigenvalues (omits 0Hz bucket)
 * FFT Magnitude Log10 buckets for various ranges (see code below), where the power-in-band is calculated between the specified frequencies. The power-in-band is actually the average and not the sum. I saw minor boosts to perform Log10 after calculating power-in-band.
 * Power-in-band spectral entropies
 * Higuchi fractal dimension with kmax=2
 * Petrosian fractal dimension
 * Hurst exponent

Code doc in `seizure_prediction/transforms.py` contains more information.

In the code all these features are specified and joined together like so:
```
FeatureConcatPipeline(
    Pipeline(InputSource(), Preprocess(), Windower(75), Correlation('none')),
    Pipeline(InputSource(), Preprocess(), Windower(75), FreqCorrelation(1, None, 'none')),
    Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), FreqBinning([0.5, 2.25, 4, 5.5, 7, 9.5, 12, 21, 30, 39, 48], 'mean'), Log10(), FlattenChannels()),
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
)

```

### Pipelines and data transformations

#### Pipeline

A `Pipeline` is a series of transforms to be applied to the source data. All the transforms I've implemented
can be found under `seizure_prediction/transforms.py`

Once data has been passed through the pipeline, the output is saved in the `data-cache` directory and
can be reloaded almost instantly next time (a few millseconds on my machine).

```
Pipeline(Windower(75), Correlation())
```

One particularly useful pipeline is the FFT magnitude. It is generally the first step of many spectral
transforms such as just raw magnitudes or spectral entropy. Recalculating the FFT for all of these
pipelines over and over again is slow and wasteful. Which leads me to...

#### InputSource

It's much faster to load up previously processed data and reuse it than to compute it every time.
The `InputSource()` class lets you specify where you want the data to be loaded from. No argument
means the original time-series data. If you specify a pipeline, it will load it from there instead.
If you look up a bit in the features section you can see the InputSource being used to load
previously-computed FFT data.

I haven't found another use for this yet other than the FFT data, but it was worth it alone for that.
The only time I don't use it for FFT data is for frequency correlation. I store everything in the data
cache as float32, and this seems to cause issues with the `Correlation` transformation having more
issues with NaNs etc. So for now `FreqCorrelation` does duplicate FFT work.

Replacing:

```
Pipeline(InputSource(Preprocess(), Windower(75), FFT(), Magnitude()), Slice(1, None), Correlation('none')),
```
with
```
Pipeline(InputSource(), Preprocess(), Windower(75), FreqCorrelation(1, None, 'none')),
```

is low-hanging fruit. It just needs to be verified that the classification performance is not worse.
I was lazy in replacing it as I had already computed these transforms weeks earlier so it didn't
bother me too much. It does however slow down from-scratch data processing which needs to do the
extra work, such as when you clone this repo or if you clear the data cache to free up some disk space.

More examples:
```
InputSource()
InputSource(Preprocess(), FFT(), Magnitude())
InputSource(Preprocess(), Windower(75), FFT(), Magnitude())
```

Also note that this can chew up a lot of disk space for caching these results.

#### FeatureConcatPipeline

It's nice and clean to specify individual transforms and pipelines. However it's very practical to combine features. The `FeatureConcatPipeline` does exactly this. It will load each pipeline individually, then concatenate all the features together.

```
FeatureConcatPipeline(
    Pipeline(Windower(75), Correlation()),
    Pipeline(Windower(75), Hurst())
)
```

#### Safe to kill whenever you like

You can kill the program without fear of losing much progress. A unit of work for the data processing is a single segment (equivalent to one of the original matlab file segments) and a unit of work for the cross-validation is one fold. Results are saved to the data cache and things can pick up where they left off last time automatically.

There is one caveat however, there's a bug with Python multiprocessing pools and KeyboardInterrupt. I run my code from IntelliJ 14 Ultimate so I don't have a problem, but if you Ctrl-C from the commandline the pool doesn't exit properly so killing from the commandline is a bit of a pain and I have just been using `killall Python` for the time being to get around it. Not ideal, but not generally an issue for me given I use IntelliJ.

### Cross-validation strategies

I have implemented two cross-validation strategies, both based on using folds.

#### LegacyStrategy

Found in `seizure_prediction/cross_validation/legacy_strategy.py`

This strategy uses 3 folds per target, using hand-picked random seeds that seemed to give good
results on my system. I'm not sure this will even work well on other peoples' systems if the
random seeds generate different folds. This is what I used for the whole competition hence the
legacy name so I've left it in there.

#### KFoldStrategy

Found in `seizure_prediction/cross_validation/kfold_strategy.py`

This was a post-competition half-hearted attempt to build a more robust K-fold cross-validation
setup. The selected sequences do not rely on random seeds, and instead I roughly hand-picked
(via an algorithm) a good number of folds and also a good selection across the preictal sequences
that somewhat maximises the coverage of the preictal set.

For example, given 3 sequences in the preictal set it will use 3 folds `[(0, 1), (0, 2), (1,2)]`.
For 6 sequences and 3 folds it will use `[(0, 1), (2, 3), (4, 5)]`.

It seems to roughly work okay now, but I've never had much trust in the cross-validation scores
versus the leaderboard scores given that the test set is generally much bigger than the given
training data.

### Misc

I haven't fully cleaned up the code as much as I could, nor documented it as much as I could.
I cleaned it up enough and tried to describe enough that you could take this code base and try
out new transforms etc without too much difficulty.

If you clone this repo, you will probably want to start looking at `main.py` and it should
hopefully be straightforward to get things going.

Feel free to message me with any questions.

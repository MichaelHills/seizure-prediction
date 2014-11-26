from collections import namedtuple
import json
import multiprocessing
import os

Settings = namedtuple('Settings', ['data_dir', 'cache_dir', 'submission_dir', 'N_jobs'])


def load_settings():
    with open('SETTINGS.json') as f:
        settings = json.load(f)

    data_dir = str(settings['competition-data-dir'])
    cache_dir = str(settings['data-cache-dir'])
    submission_dir = str(settings['submission-dir'])
    N_jobs = str(settings['num-jobs'])
    N_jobs = multiprocessing.cpu_count() if N_jobs == 'auto' else int(N_jobs)

    for d in (cache_dir, submission_dir):
        try:
            os.makedirs(d)
        except:
            pass

    return Settings(data_dir=data_dir, cache_dir=cache_dir, submission_dir=submission_dir, N_jobs=N_jobs)

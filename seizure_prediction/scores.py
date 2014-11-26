import numpy as np

# helper methods for printing scores


def get_score_summary(name, scores):
    summary = 'mean=%.3f std=%.3f' % (np.mean(scores), np.std(scores))
    score_list = ['%.3f' % score for score in scores]
    return '%s [%s] %s' % (summary, ','.join(score_list), name)


def print_results(summaries):
    summaries.sort(cmp=lambda x,y: cmp(x[1], y[1]))
    if len(summaries) > 1:
        print 'summaries'
        for s, mean in summaries:
            print s

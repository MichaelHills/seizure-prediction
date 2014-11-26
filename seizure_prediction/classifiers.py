import numpy as np
import sklearn
import sklearn.pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# NOTE(mike): doesn't handle multi-class
class SimpleLogisticRegression(LinearRegression):
    def predict_proba(self, X):
        predictions = self.predict(X)
        predictions = sklearn.preprocessing.scale(predictions)
        predictions = 1.0 / (1.0 + np.exp(-0.5 * predictions))
        return np.vstack((1.0 - predictions, predictions)).T


def make_svm(gamma, C):
    cls = sklearn.pipeline.make_pipeline(StandardScaler(),
        SVC(gamma=gamma, C=C, probability=True, cache_size=500, random_state=0))
    name = 'ss-svc-g%.4f-C%.1f' % (gamma, C)
    return (cls, name)


def make_lr(C):
    cls = sklearn.pipeline.make_pipeline(StandardScaler(), LogisticRegression(C=C))
    name = 'ss-lr-C%.4f' % C
    return (cls, name)


def make_simple_lr():
    return (sklearn.pipeline.make_pipeline(StandardScaler(), SimpleLogisticRegression()), 'ss-slr')

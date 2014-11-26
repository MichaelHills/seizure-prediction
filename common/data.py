import os
import os.path


def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass

class jsdict(dict):
    def __init__(self, *args, **kwargs):
        super(jsdict, self).__init__(*args, **kwargs)
        self.__dict__ = self

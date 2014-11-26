from datetime import datetime

def unix_time(dt):
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()


def unix_time_millis(dt):
    return int(unix_time(dt) * 1000.0)


def get_millis():
    return unix_time_millis(datetime.now())


def get_seconds():
    return get_millis() / 1000.0


class Timer:
    def __init__(self):
        self.start = get_millis()

    def elapsed_millis(self):
        return get_millis() - self.start

    def elapsed_seconds(self):
        return long(self.elapsed_millis() / 1000.0)

    def pretty_str(self):
        ms = self.elapsed_millis()
        if ms > 5000:
            return '%ds' % long(ms / 1000.0)
        return '%dms' % ms

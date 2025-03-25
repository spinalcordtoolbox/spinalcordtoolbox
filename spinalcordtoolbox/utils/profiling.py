import atexit
import time


# Wrapper class which manages our timing
class Timer:
    def __init__(self):
        self._t0 = time.time()

    def __del__(self):
        self.stop()

    def stop(self):
        print("Elapsed time: %.3f seconds" % (time.time() - self._t0))


def begin_timer():
    # Initiate the timer (which starts its timer)
    timer = Timer()
    # noinspection PyTypeChecker
    atexit.register(timer.stop)

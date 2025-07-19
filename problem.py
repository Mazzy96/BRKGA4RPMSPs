#!/usr/bin/python3

import threading


class Problem(object):
    def __init__(self, args, dimensions):
        self.evaluations = 0
        self.eval_lock = threading.Lock()
        self.dimensions = dimensions
        self.args = args

    def batch_evaluate(self, keys, threads=1, print_sol=False):
        # Evaluation should proceed without changing the class state (so that
        # batch evaluation is possible)
        self.eval_lock.acquire()
        try:
            self.evaluations += len(keys)
        finally:
            self.eval_lock.release()



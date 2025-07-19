from loguru import logger
import argparse
import copy
import numpy as np
import numpy.random as npr
import sys
import time

from solver import Solver, min_max_arg

def add_parser_args(parser):
    parser.add_argument('--batch_size', type=int, default=1, help='Number of random solutions to evaluate in each iteration. Does not actually change anything other than the algorithm speed.')

class random(Solver):
    def __init__(self, problem, args):
        super().__init__(problem, args)
        self.ngen = 0

    def terminate(self):
        return super().terminate()

    def solve(self):
        best = None
        best_val = 1e99

        while not self.terminate():
            rbatch = self.random_population(self.args.batch_size)
            self.ngen += self.args.batch_size
            fitness = self.problem.batch_evaluate(rbatch, self.args.threads)

            # Update global best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_val:
                best = rbatch[best_idx].copy()
                best_val = fitness[best_idx]
                self.status_new_best(best_val, f"After {self.ngen} random solutions")

        return best_val, best




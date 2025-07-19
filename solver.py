#!/usr/bin/python3

from loguru import logger
import argparse
import numpy.random as npr
import numpy as np
import signal
import time

def min_max_arg(name, min_val=-1e99, max_val=1e99):
    class MinMaxSize(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if values < min_val or values > max_val:
                parser.error(f"{name} out of range ({values} should be in [{min_val}, {max_val}]")
            setattr(namespace, self.dest, values)
    return MinMaxSize

class Solver(object):
    def __init__(self, problem, args):
        self.args = args
        self.problem = problem
        self.wall_start = time.time()
        self.caught_signal = False

        orig_sigint = signal.getsignal(signal.SIGINT)
        orig_sigterm = signal.getsignal(signal.SIGTERM)

        def sighandler_lambda(sig, frame):
            logger.critical(f"Signal {sig} received; stopping at next opportunity.")
            self.signal_handler(sig, frame)
            signal.signal(signal.SIGINT, orig_sigint)
            signal.signal(signal.SIGTERM, orig_sigterm)
            return True

        signal.signal(signal.SIGINT, sighandler_lambda)
        signal.signal(signal.SIGTERM, sighandler_lambda)

    def signal_handler(self, sig, frame):
        self.caught_signal = True

    def solve(self):
        pass

    def terminate(self):
        if time.time() - self.wall_start > self.args.max_wall:
            return True
        elif time.process_time() > self.args.max_cpu:
            return True
        elif self.problem.evaluations > self.args.max_evals:
            return True
        elif self.caught_signal:
            return True
        return False

    def warm_population_UPMSR_PS(self, size): #UPMSR-PS
        population = npr.uniform(0, 1, (size, self.problem.dimensions))

        main_result = self.problem.warm_start()
        setup_start_time = main_result['setup_start'] 
        setup_end_time = main_result['setup_end'] 
        process_start_time = main_result['process_start'] 
        process_end_time = main_result['process_end'] 
        # Normalize the extracted times
        normalized_setup_start = (setup_start_time - np.min(setup_start_time)) / (np.max(setup_start_time) - np.min(setup_start_time))
        normalized_setup_end = (setup_end_time - np.min(setup_end_time)) / (np.max(setup_end_time) - np.min(setup_end_time))
        normalized_process_start = (process_start_time - np.min(process_start_time)) / (np.max(process_start_time) - np.min(process_start_time))
        normalized_process_end = (process_end_time - np.min(process_end_time)) / (np.max(process_end_time) - np.min(process_end_time))

        # Assign normalized values to the first four population members
        population[0] = normalized_setup_start
        population[1] = normalized_setup_end
        population[2] = normalized_process_start
        population[3] = normalized_process_end
        
        # # Generate population[4] and population[5] by adding Gaussian noise
        # noise = npr.normal(loc=0, scale=0.005, size=self.problem.n_jobs)
        # population[4] = np.clip(normalized_setup_start + noise, 0, 1)
        # noise = npr.normal(loc=0, scale=0.005, size=self.problem.n_jobs)
        # population[5] = np.clip(normalized_setup_end + noise, 0, 1)
        # noise = npr.normal(loc=0, scale=0.005, size=self.problem.n_jobs)
        # population[6] = np.clip(normalized_process_start + noise, 0, 1)
        # noise = npr.normal(loc=0, scale=0.005, size=self.problem.n_jobs)
        # population[7] = np.clip(normalized_process_end + noise, 0, 1)

        return population

    def warm_population_IPMR_P(self, size): #IPMR-P
        population = npr.uniform(0, 1, (size, self.problem.dimensions))

        # Generate the specific random key vector based on normalized delivery_times
        normalized_delivery_times = (self.problem.delivery_times - np.min(self.problem.delivery_times)) / (
                    np.max(self.problem.delivery_times) - np.min(self.problem.delivery_times))
        
        # Generate the specific random key vector based on normalized delivery_times
        normalized_ready_times = (self.problem.ready_times - np.min(self.problem.ready_times)) / (
                    np.max(self.problem.ready_times) - np.min(self.problem.ready_times))

        population[0] = normalized_delivery_times  # Assign to first individual

        population[1] = normalized_ready_times  # Assign to first individual

        # # Generate population[2] by adding Gaussian noise
        # noise = npr.normal(loc=0, scale=0.005, size=self.problem.n_jobs)  # Small perturbation
        # population[2] = np.clip(normalized_ready_times + noise, 0, 1)  # Ensure values remain in [0,1]

        # # Generate population[2] by adding Gaussian noise
        # noise = npr.normal(loc=0, scale=0.005, size=self.problem.n_jobs)  # Small perturbation
        # population[3] = np.clip(normalized_delivery_times + noise, 0, 1)  # Ensure values remain in [0,1]

        return population

    def status_new_best(self, val, msg=''):
        if msg:
            msg = f'; {msg}'
        logger.info(f"(Evals {self.problem.evaluations}; CPU {time.process_time():.2f}) New best objective: {val}{msg}")

    def random_population(self, size):
        # Compute flow times
        population = npr.uniform(0, 1, (size, self.problem.dimensions))
        return population

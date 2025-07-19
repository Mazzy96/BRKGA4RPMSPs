#!/usr/bin/python3

from loguru import logger
import argparse
import importlib
import numpy as np
import numpy.random as npr
import os.path
import random
import sys
import time
import traceback


def grka(solver):
    logger.info(f"Solving start (CPU {time.process_time()})")
    ret = solver.solve()
    logger.info(f"Solving finished (CPU {time.process_time()})")
    return ret


def main():
    parser = argparse.ArgumentParser(description='Generalized Random-Key Algorithm')
    parser.add_argument('problem', choices=['UPMSR_PS', 'UPMSR_PS_grasp', 'PMSP', 'IPMR_P'], help='Problem type to solve')
    parser.add_argument('instance', type=str, help='Instance path')
    parser.add_argument('--solver', choices=['brkga','random'], default='cmaes', help='Solver to use to solve the problem')
    parser.add_argument('--max_cpu', type=float, default=1e99, help='Maximum CPU time in seconds')
    parser.add_argument('--max_wall', type=float, default=1e99, help='Maximum wall time in seconds')
    parser.add_argument('--max_evals', type=int, default=int(1e15), help='Maximum number of objective function evaluations')
    parser.add_argument('--threads', type=int, default=1, help='Maximum number of threads to use')
    parser.add_argument('-s', '--seed', type=int, default=-1, help='Random seed (default = -1, meaning no seed given)')
    parser.add_argument('--tuner', action='store_true', default=False, help='Enable tuning mode; no output will be printed to stdout except for ')
    parser.add_argument('--ignore', type=str, default="", help='Used for tuning, just ignore it.')
    parser.add_argument('--truncate_final', action='store_true', default=False, help='Truncates the final solution provided to an integer. Useful when using a modified objective function that shouldn\'t be output to the tuner or a user.')

    args, unparsed_args = parser.parse_known_args(sys.argv[1:])
    

    # TODO Need to provide CLI help for algs and problems!

    logger.remove()
    if args.tuner: #  colorize=True,
        logger.add(sys.stderr, level='ERROR', format="<blue>{time:YY-MM-dd HH:mm:ss.SSS}</blue> |{level}| <green>{module}:{function}:{line}</green> - <level>{message}</level>")
    else:
        logger.add(sys.stdout, level=0, format="<blue>{time:YY-MM-dd HH:mm:ss.SSS}</blue> |{level}| <green>{module}:{function}:{line}</green> - <level>{message}</level>")
#     logger.add(sys.stderr, colorize=True, level='ERROR', backtrace=True, format="Error location: <green>|{file}:{line}|</green> ")

    if args.seed >= 0:
        random.seed(args.seed)
        npr.seed(args.seed)

    prob_mod = f"problem_{args.problem}"
    try:
        problem_mod = importlib.__import__(prob_mod)
    except:
        logger.exception(f"Error loading problem module {prob_mod}")
        sys.exit(2)


    solve_mod = f"solver_{args.solver}"
    try:
        solver_mod = importlib.__import__(solve_mod)
    except ModuleNotFoundError:
        logger.exception(f"Error loading solver module {solve_mod}")
        sys.exit(3)


    # reparse arguments now for specific problem and solver

    problem_mod.add_parser_args(parser)
    solver_mod.add_parser_args(parser)

    # TODO use unparsed_args here and merge?
    args = parser.parse_args(sys.argv[1:])

    prob_ = getattr(problem_mod, args.problem)
    solver_ = getattr(solver_mod, args.solver)

    instance = problem_mod.read_instance(args.instance)
    problem = prob_(args, instance)
    solver = solver_(problem, args)

    logger.info(f"Solving problem {args.problem} with solver {args.solver} on instance {args.instance}")
    logger.info(f"Max CPU: {args.max_cpu}; Max wall: {args.max_wall}; Max evals: {args.max_evals}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Threads: {args.threads}")

    try:
        best_val, best = grka(solver)
        if args.truncate_final:
            best_val = int(best_val)
    except Exception as ee:
        if args.tuner:
            print(f"GGA CRASHED 9999")
#             from datetime import datetime
#             with open(f'tmp-{int(datetime.now().timestamp())}.err', 'w') as fp:
#                 fp.write(f'GGA crash: {" ".join(sys.argv)}')
            sys.exit(1)
        else:
            raise ee

    logger.info("=== Printing solution ===")
    if not args.tuner:
        problem.batch_evaluate(np.array([best]), print_sol=True)
    logger.info("=== /Printing solution ===")

    logger.info(f"Total evaluations: {problem.evaluations}")
    logger.info(f"Elapsed CPU (s): {time.process_time():.2f}")
    logger.info(f"Elapsed wall (s): {time.time() - solver.wall_start:.2f}")
    logger.info(f"Evals/CPU s: {problem.evaluations / time.process_time():.2f}")
    logger.info(f"Best objective found: {best_val}")

    if args.tuner:
        if best_val is np.nan or best_val > 1e9:
            best_val = 9999 if args.use_gap else 1e9
        print(f"GGA SUCCESS {best_val}")


if __name__ == "__main__":
    main()



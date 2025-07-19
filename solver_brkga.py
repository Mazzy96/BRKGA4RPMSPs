from loguru import logger
import argparse
import copy
import numpy as np
import numpy.random as npr
import sys
import time

from solver import Solver, min_max_arg


def add_parser_args(parser):
    # parser.add_argument('--pop_size', type=int, action=min_max_arg('Population', 5), default=205, help='BRKGA population size')
    # parser.add_argument('--generations', type=int, action=min_max_arg('Generations', 1), default=1000, help='BRKGA number of generations')
    # parser.add_argument('--elite', type=float, action=min_max_arg('Elite percent', 1e-3,1.0), default=0.11829048731284703, help='BRKGA elite percentage')
    # parser.add_argument('--mutants', type=float, action=min_max_arg('Mutant percent', 1e-3,1.0), default=0.12223603372566214, help='BRKGA mutant percentage')
    # parser.add_argument('--bias', type=float, action=min_max_arg('Crossover elite bias', 0.5,1.0), default=0.8637517550580126, help='BRKGA mutant percentage')
    # parser.add_argument('--loop_to_convergence', action='store_true', default=False, help='Ignore the maximum number of generations and only stop when the population has converged')
    # parser.add_argument('--max_gen_no_improve', type=int, default=50, help='Maximum number of generations without improvement before termination')  # New argument
    # parser.add_argument('--pop_size', type=int, action=min_max_arg('Population', 5), default=500, help='BRKGA population size')
    # parser.add_argument('--generations', type=int, action=min_max_arg('Generations', 1), default=1000, help='BRKGA number of generations')
    # parser.add_argument('--elite', type=float, action=min_max_arg('Elite percent', 1e-3,1.0), default=0.1, help='BRKGA elite percentage')
    # parser.add_argument('--mutants', type=float, action=min_max_arg('Mutant percent', 1e-3,1.0), default=0.1, help='BRKGA mutant percentage')
    # parser.add_argument('--bias', type=float, action=min_max_arg('Crossover elite bias', 0.5,1.0), default=0.6, help='BRKGA mutant percentage')
    # parser.add_argument('--loop_to_convergence', action='store_true', default=False, help='Ignore the maximum number of generations and only stop when the population has converged')
    # parser.add_argument('--max_gen_no_improve', type=int, default=50, help='Maximum number of generations without improvement before termination')  # New argument
    parser.add_argument('--pop_size', type=int, action=min_max_arg('Population', 5), default=391, help='BRKGA population size')
    parser.add_argument('--generations', type=int, action=min_max_arg('Generations', 1), default=1000, help='BRKGA number of generations')
    parser.add_argument('--elite', type=float, action=min_max_arg('Elite percent', 1e-3,1.0), default=0.18669311993421434, help='BRKGA elite percentage')
    parser.add_argument('--mutants', type=float, action=min_max_arg('Mutant percent', 1e-3,1.0), default=0.10395168244120889, help='BRKGA mutant percentage')
    parser.add_argument('--bias', type=float, action=min_max_arg('Crossover elite bias', 0.5,1.0), default=0.6058608996127087, help='BRKGA mutant percentage')
    parser.add_argument('--loop_to_convergence', action='store_true', default=False, help='Ignore the maximum number of generations and only stop when the population has converged')
    parser.add_argument('--convergence_eps', type=float, default=1e-4, help='Stop if the average change in objective function over the last convergence_last generations falls below this epsilon')
    parser.add_argument('--convergence_last', type=float, default=5, help='Stop if the average change in objective function over the last n generations falls below convergence_eps')
class brkga(Solver):
    def __init__(self, problem, args):
        super().__init__(problem, args)
        self.zero_obj = False
        if args.elite + args.mutants >= 1.0:
            logger.error(f"Elite percentage plus mutant percentage must be less than 1! ({args.elite} + {args.mutants} = {args.elite + args.mutants}")
            sys.exit(4)

    def terminate(self):
        return super().terminate()

    def solve(self):
        pop = self.warm_population_UPMSR_PS(self.args.pop_size)
        # pop = self.warm_population_IPMR_P(self.args.pop_size)
        # pop = self.random_population(self.args.pop_size)

        fitness = self.problem.batch_evaluate(pop, self.args.threads)

        nelite = max(1, int(self.args.elite * self.args.pop_size))
        nmutants = max(1, int(self.args.mutants * self.args.pop_size))
        nnonelite = self.args.pop_size - nelite
        recomb = max(1, self.args.pop_size - nelite - nmutants)

        best_idx = np.argmin(fitness)
        best = pop[best_idx].copy()
        best_val = fitness[best_idx]
        self.status_new_best(best_val, "initial population")

        use_gens = self.args.generations + 1
        # if self.args.loop_to_convergence:
        #     use_gens = 2147483647
        no_improve_counter = 0   # Track generations without improvement

        if best_val == 0:
            return best_val, best

        for gg in range(1, use_gens):
            sort_idx = np.argsort(fitness)
            nelite_slice = sort_idx[:nelite]
            elite = pop[nelite_slice,:]
            nonelite = pop[sort_idx[nelite:],:]

            relite_idx = npr.choice(nelite, recomb, replace=True)
            rnonelite_idx = npr.choice(nnonelite, recomb, replace=True)

            recomb_elite = elite[relite_idx,:]
            recomb_nonelite = nonelite[rnonelite_idx,:]

            rm = npr.rand(recomb, self.problem.dimensions)
            recombined = np.where(rm <= self.args.bias, recomb_elite, recomb_nonelite)

            # mutants = self.warm_evolution(nmutants, elite)
            mutants = self.random_population(nmutants)

            # don't recompute fitness for elite population
            recomb_mutant = np.vstack((recombined, mutants))
            fitness_rm = self.problem.batch_evaluate(recomb_mutant, self.args.threads)
                            
            fitness = np.hstack((fitness[nelite_slice], fitness_rm))
            best_idx = np.argmin(fitness)
            
            if fitness[best_idx] < best_val:
                best = pop[best_idx].copy()
                best_val = fitness[best_idx]
                self.status_new_best(best_val, msg=f"Generation {gg}")
                no_improve_counter = 0  # Reset counter
            else:
                # if self.args.loop_to_convergence:
                no_improve_counter += 1  # Increment no improvement counter

            if best_val == 0:
                return best_val, best

            pop = np.vstack((elite, recomb_mutant))

            if self.args.loop_to_convergence and no_improve_counter >= self.args.max_gen_no_improve:
                # Terminate if maximum generations without improvement is reached
                logger.info(f"Terminating after {gg} generations due to {no_improve_counter} generations without improvement.")
                break

            if self.terminate():
                break
        
        return best_val, best


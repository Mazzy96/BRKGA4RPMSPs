import numpy as np
from loguru import logger
from problem import Problem
import pandas as pd
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import time

def add_parser_args(parser):
    parser.add_argument('--weight_tardiness', type=float, default=1, help='Weight for tardiness penalty')

def read_instance(inst_path):

    inst = IPMR_P_instance()

    inst.nb_jobs = 0
    inst.nb_machines = 0
    inst.nb_workers = 0
    inst.nb_days = 0
    inst.comp_jm = []  # Compatibility matrix: jobs/machines
    inst.comp_jw = []  # Compatibility matrix: jobs/workers
    inst.comp_mw = []  # Compatibility matrix: machines/workers
    inst.r_j = []  # Release dates
    inst.d_j = []  # Due dates
    inst.q_j = []  # Occupation
    inst.w_j = []  # Weights
    inst.p_j = []  # Processing times
    inst.u_kt = []  # Availability
    inst.nb_prec = 0  # Number of precedences
    inst.mat_prec = []  # Matrix of precedences
    inst.nb_contig = 0  # Number of contiguity
    inst.mat_contig = []  # Matrix of contiguity

    with open(inst_path, 'r') as file:

        # Read number of jobs
        inst.nb_jobs = int(file.readline().strip())

        # Read number of machines
        inst.nb_machines = int(file.readline().strip())

        # Read number of workers
        inst.nb_workers = int(file.readline().strip())

        # Read number of days
        inst.nb_days = int(file.readline().strip())

        # Read compatibility matrix jobs/machines
        for _ in range(inst.nb_jobs):
            inst.comp_jm.append(list(map(int, file.readline().strip().split())))

        # Read compatibility matrix jobs/workers
        for _ in range(inst.nb_jobs):
            inst.comp_jw.append(list(map(int, file.readline().strip().split())))

        # Read compatibility matrix machines/workers
        for _ in range(inst.nb_machines):
            inst.comp_mw.append(list(map(int, file.readline().strip().split())))

        # Read release dates
        inst.r_j = list(map(int, file.readline().strip().split()))

        # Read due dates
        inst.d_j = list(map(int, file.readline().strip().split()))

        # Read occupation
        inst.q_j = list(map(int, file.readline().strip().split()))

        # Read weights
        inst.w_j = list(map(int, file.readline().strip().split()))

        # Read processing times
        inst.p_j = list(map(int, file.readline().strip().split()))

        # Read availability
        for _ in range(inst.nb_workers):
            inst.u_kt.append(list(map(int, file.readline().strip().split())))

        # Read number of precedences
        inst.nb_prec = int(file.readline().strip())

        # Read matrix of precedences
        for _ in range(inst.nb_prec):
            inst.mat_prec.append(list(map(int, file.readline().strip().split())))

        # Read number of contiguity
        inst.nb_contig = int(file.readline().strip())

        # Read matrix of contiguity
        for _ in range(inst.nb_contig):
            inst.mat_contig.append(list(map(int, file.readline().strip().split())))

   
    # logger.info(f"Instance Number: {inst.instance_number}")
    # logger.info(f"Number of Jobs: {inst.number_of_jobs}")
    # logger.info(f"Number of Machines: {inst.number_of_machines}")
    # logger.info(f"Number of Weeks: {inst.number_of_weeks}")
    # logger.info(f"Personnel Available: {inst.personnel_available}")
    # logger.info(f"Max Time Window Length: {inst.max_time_window_length}")
    # logger.info(f"Time Window Density: {inst.time_window_density}")
    # logger.info(f"Weekly Personnel Availability: {inst.weekly_personnel_availability}")
    # logger.info(f"Mean Time Value: {inst.mean_time_value}")
    # logger.info(f"Upper Bound: {inst.upper_bound}")
    # logger.info(f"Lower Bound: {inst.lower_bound}")

    # logger.info("Job Processing Times:")
    # logger.info(inst.job_processing_times)

    # logger.info("Initial Setup Times:")
    # logger.info(inst.initial_setup_times)

    # logger.info("Machine Eligibilities:")
    # for machine, jobs in inst.machine_eligibilities.items():
    #     logger.info(f"Machine {machine + 1}: {jobs}")

    # logger.info("Personnel Times:")
    # logger.info(inst.personnel_times)

    # logger.info("Personnel Assignments:")
    # logger.info(inst.personnel_assignments)

    # logger.info("Unavailability Start Periods:")
    # logger.info(inst.unavailability_start_period)

    # logger.info("Unavailability End Periods:")
    # logger.info(inst.unavailability_end_period)

    # logger.info("Unavailability Start Times:")
    # logger.info(inst.unavailability_start_times)

    # logger.info("Unavailability End Times:")
    # logger.info(inst.unavailability_end_times)

    # logger.info("Release Period:")
    # logger.info(inst.release_period)

    # logger.info("Delivery Period:")
    # logger.info(inst.delivery_period)

    # logger.info("Release Times:")
    # logger.info(inst.release_times)

    # logger.info("Delivery Times:")
    # logger.info(inst.delivery_times)

    # for machine, setup_times in inst.sequence_dependent_setup_times.items():
    #     logger.info(f"Sequence Dependent Setup Times Machine {machine + 1}:")
    #     logger.info(setup_times)

    return inst

class IPMR_P_instance(object):
    def __init__(self):
        self.n_jobs = 0
        self.n_machines = 0
        self.n_workers = 0
        self.n_days = 0
        self.comp_jm = []  # Compatibility matrix: jobs/machines
        self.comp_jw = []  # Compatibility matrix: jobs/workers
        self.comp_mw = []  # Compatibility matrix: machines/workers
        self.ready_times = []  # Release dates
        self.delivery_times = []  # Due dates
        self.occupation_list = []  # Occupation
        self.weight_list = []  # Weights
        self.processing_times = []  # Processing times
        self.daily_worker_availability = []  # Availability
        self.n_prec = 0  # Number of precedences
        self.precedence_matrix = []  # Matrix of precedences
        self.n_contig = 0  # Number of contiguity
        self.contiguity_matrix = []  # Matrix of contiguity

class IPMR_P(Problem):
    def __init__(self, args, inst):
        super().__init__(args, inst.nb_jobs)
        self.inst = inst
        self.args = args  # Store args in the class to access it later
        self.seed = args.seed  # Store the seed value in the class
        # Extract data from instance_data
        self.n_jobs = inst.nb_jobs
        self.n_machines = inst.nb_machines
        self.n_workers = inst.nb_workers
        self.n_days = inst.nb_days
        self.n_prec = inst.nb_prec
        self.n_contig = inst.nb_contig

        print()

        # Ready times (directly from r_j in the file)
        self.ready_times = np.array(inst.r_j).tolist()

        # Delivery times (directly from d_j in the file)
        self.delivery_times = np.array(inst.d_j).tolist()

        # Processing times (directly from p_j in the file)
        self.occupation_list = np.array(inst.q_j)

        # Processing times (directly from p_j in the file)
        self.weight_list = np.array(inst.w_j)

        # Processing times (directly from p_j in the file)
        self.processing_times = np.array(inst.p_j)

        # Daily worker availability (directly from u_kt in the file)
        self.daily_worker_availability = np.array(inst.u_kt)  # This is already a list of lists

        # Initialize eligible_machines and machine_eligibilities
        self.eligible_machines = [[] for _ in range(self.n_jobs)]
        self.machine_eligibilities = [[] for _ in range(self.n_machines)]

        # Initialize worker_job_eligibilities and eligible_workers_jobs
        self.worker_job_eligibilities = [[] for _ in range(self.n_workers)]
        self.eligible_workers_jobs = [[] for _ in range(self.n_jobs)]

        # Initialize machine_worker_eligibilities and eligible_machines_workers
        self.machine_worker_eligibilities = [[] for _ in range(self.n_workers)]
        self.eligible_machines_workers = [[] for _ in range(self.n_machines)]

        # Populate all eligibility matrices in a single loop
        for machine in range(self.n_machines):
            for job in range(self.n_jobs):
                if inst.comp_jm[job][machine] == 1:  # Job-Machine compatibility
                    self.eligible_machines[job].append(machine)  # Add machine to job's eligible machines
                    self.machine_eligibilities[machine].append(job)  # Add job to machine's eligible jobs


            for worker in range(self.n_workers):
                if inst.comp_mw[machine][worker] == 1:  # Machine-Worker compatibility
                    self.eligible_machines_workers[machine].append(worker)  # Add worker to machine's eligible workers
                    self.machine_worker_eligibilities[worker].append(machine)  # Add machine to worker's eligible machines

        for worker in range(self.n_workers):
            for job in range(self.n_jobs):
                if inst.comp_jw[job][worker] == 1:  # Job-Worker compatibility
                    self.worker_job_eligibilities[worker].append(job)  # Add job to worker's eligible jobs
                    self.eligible_workers_jobs[job].append(worker)  # Add worker to job's eligible workers

        # Precedence matrix (directly from mat_prec in the file)
        self.precedence_matrix = inst.mat_prec  # List of lists representing precedence constraints

        # Contiguity matrix (directly from mat_contig in the file)
        self.contiguity_matrix = inst.mat_contig  # List of lists representing contiguity constraints
        self.best_main_result = None
        self.previous_value = float('inf')
        self.best_evaluation = float('inf')
        self.count = 0
        self.run_time = 0

    def batch_evaluate(self, keys, threads=1, zero_obj = False, print_sol=False):
        super().batch_evaluate(keys)
        return np.array([self.evaluate(key, zero_obj, print_sol) for key in keys])
    
    def generate_random_keys_from_order(self, order):
        n = len(order)
        if n == 1:
            return np.array([0.0])  # Avoid division by zero

        sorted_values = np.array([(i) / (n - 1) for i in range(n)])  # Step 1 & 2

        # Step 3: Assign sorted values based on order
        keys = np.zeros(n)
        for i, position in enumerate(order):
            keys[position] = sorted_values[i]
        
        return keys
    
    def move_tardiest_job_backward(self, JS, tardiness, tabu_list):

        sorted_indices = np.argsort(tardiness)[::-1]  # Sort by tardiness (descending)
       
        # Find the most tardy job that is NOT in the tabu list and is movable
        tardiest_idx = None
        for idx in sorted_indices:
            tardiest_job = JS[idx]
            if tardiest_job not in tabu_list and idx > 0:  # Ensure it's movable
                tardiest_idx = idx
                tabu_list.add(tardiest_job)
                break

        if tardiest_idx is None:
            tabu_list = set()
            return JS, tardiness, tabu_list  # No movable tardy jobs

        # Move the tardiest job backward and evaluate different positions
        best_JS = JS.copy()
        best_tardiness = np.sum(tardiness)
        best_individual_tardiness = tardiness

        for pos in range(tardiest_idx, 0, -1):  
            JS[pos], JS[pos - 1] = JS[pos - 1], JS[pos]  # Swap positions

            key = self.generate_random_keys_from_order(JS)
            # Evaluate new tardiness
            solution = self.evaluate_final(key, check_print = False)
            if (solution['job_start_days'][pos - 1] == solution['aligned_ready_times'][pos - 1]):
                break
            new_tardiness = np.sum(solution['individual_weighted_tardiness'])

            # Apply randomness: Accept if better OR with probability p if worse
            if new_tardiness < best_tardiness:
                best_tardiness = new_tardiness
                best_individual_tardiness = solution['individual_weighted_tardiness']
                best_JS = JS.copy()
            
            if np.sum(best_individual_tardiness) == 0:
                break
            
        return best_JS, best_individual_tardiness, tabu_list  # Return the best sequence and moved job
    
    def local_search(self, order):
        # Initialize tabu list and start the process
        key = self.generate_random_keys_from_order(order)
        solution = self.evaluate_final(key, check_print = False)
        ready_jobs, best_individual_tardiness = solution['JS'], solution['individual_weighted_tardiness']
        if np.sum(best_individual_tardiness) == 0:
            return key
        
        tabu_list = set()
        count = 0
        converged_tardiness = np.sum(best_individual_tardiness)
        converged_count = 0
        
        while  time.process_time() < 1800 and count < 2 * self.n_jobs:
            # print("time", time.process_time())
            ready_jobs, best_individual_tardiness, tabu_list = self.move_tardiest_job_backward(ready_jobs, best_individual_tardiness, tabu_list)
            print(f"Iteration {count+1}: Weighted Tardiness = {np.sum(best_individual_tardiness)}")
            
            if np.sum(best_individual_tardiness) < converged_tardiness:
                converged_tardiness = np.sum(best_individual_tardiness)
                converged_count = 0
            else:
                converged_count += 1

            count += 1

            if np.sum(best_individual_tardiness) == 0:
                break
        
        key = self.generate_random_keys_from_order(ready_jobs)

        return key
    
    def evaluate(self, key, zero_obj=False, print_sol=False):
        start_cpu = time.process_time()  # Capture start CPU time
        JS = []
        JM = []
        JW = []
        job_start_days = []
        job_end_days = []
        worker_start_days = []
        worker_end_days = []
        machine_loads = np.zeros(self.n_machines)
        contg_flag = defaultdict(list)

        # Create a new NumPy array with shape (n_workers, n_days * 2)
        worker_availability_hours = np.zeros((self.n_workers, self.n_days * 2))

        # Now, populate worker_availability_hours based on self.daily_worker_availability
        # Assuming self.daily_worker_availability is a 2D array with shape (n_workers, n_days)
        for worker in range(self.n_workers):
            for day in range(self.n_days):
                worker_availability_hours[worker, day] = self.daily_worker_availability[worker, day]  # First half
                worker_availability_hours[worker, day + self.n_days] = self.daily_worker_availability[worker, day]  # Second half

        # worker_availability_hours = self.daily_worker_availability.copy()
        suc_jobs = []
        prec_jobs = []
        contg_jobs = []
        acontg_jobs = []

        # Initialize the set of jobs
        job_order = list(range(self.n_jobs))
        # Sort jobs based on the key
        job_order = np.argsort(key)

        for job in job_order:
            JS, JM, JW, job_start_days, job_end_days, worker_start_days, worker_end_days, machine_loads, worker_availability_hours, suc_jobs, prec_jobs, contg_jobs, acontg_jobs, contg_flag = self.assign_jobs_to_machines(
                job, JS, JM, JW, job_start_days, job_end_days, worker_start_days, worker_end_days, machine_loads, worker_availability_hours, suc_jobs, prec_jobs, contg_jobs, acontg_jobs, contg_flag)
        
        for job, machines in list(contg_flag.items()):
            if machines:  # Check if there's at least one machine assigned
                machine = machines[0]  # Take the first machine in the list
                JS, JM, JW, job_start_days, job_end_days, worker_start_days, worker_end_days, \
                machine_loads, worker_availability_hours, suc_jobs, prec_jobs, \
                contg_jobs, acontg_jobs, contg_flag = self.assign_jobs_to_machines(
                    job, JS, JM, JW, job_start_days, job_end_days, worker_start_days,
                    worker_end_days, machine_loads, worker_availability_hours,
                    suc_jobs, prec_jobs, contg_jobs, acontg_jobs, contg_flag, machine_flag=machine
                )
    
        aligned_delivery_times = np.array([self.delivery_times[job] for job in JS])
        job_end_days = np.array(job_end_days)
        tardiness = np.maximum(0, job_end_days - aligned_delivery_times)
        weights = np.array([self.weight_list[job] for job in JS])
        weighted_tardiness = np.sum(weights * tardiness)

        # Compute final value using weights
        current_value = weighted_tardiness

        # Compare the current value with the previous value
        if current_value < self.previous_value: 
            self.previous_value = current_value
            self.best_main_result = key.copy()  # Save the current metrics as the best metrics

        end_cpu = time.process_time()  # Capture end CPU time
        elapsed_cpu = end_cpu - start_cpu  # Compute elapsed CPU time
        self.run_time += elapsed_cpu

        if print_sol:
            # print('key',self.best_main_result)
            self.evaluate_final(self.best_main_result, check_print=True)
            
        return current_value  # Return the objective function

    def evaluate_final(self, key, check_print):
        start_cpu = time.process_time()  # Capture start CPU time
        JS = []
        JM = []
        JW = []
        job_start_days = []
        job_end_days = []
        worker_start_days = []
        worker_end_days = []
        machine_loads = np.zeros(self.n_machines)

        suc_jobs = []
        prec_jobs = []
        contg_jobs = []
        acontg_jobs = []

        contg_flag = defaultdict(list)

        # Create a new NumPy array with shape (n_workers, n_days * 2)
        worker_availability_hours = np.zeros((self.n_workers, self.n_days * 2))

        # Now, populate worker_availability_hours based on self.daily_worker_availability
        # Assuming self.daily_worker_availability is a 2D array with shape (n_workers, n_days)
        for worker in range(self.n_workers):
            for day in range(self.n_days):
                worker_availability_hours[worker, day] = self.daily_worker_availability[worker, day]  # First half
                worker_availability_hours[worker, day + self.n_days] = self.daily_worker_availability[worker, day]  # Second half

        # Initialize the set of jobs
        job_order = list(range(self.n_jobs))
        # Sort jobs based on the key
        job_order = np.argsort(key)

        for job in job_order:
            JS, JM, JW, job_start_days, job_end_days, worker_start_days, worker_end_days, machine_loads, worker_availability_hours, suc_jobs, prec_jobs, contg_jobs, acontg_jobs, contg_flag = self.assign_jobs_to_machines(
                job, JS, JM, JW, job_start_days, job_end_days, worker_start_days, worker_end_days, machine_loads, worker_availability_hours, suc_jobs, prec_jobs, contg_jobs, acontg_jobs, contg_flag)
        
        for job, machines in list(contg_flag.items()):
            if machines:  # Check if there's at least one machine assigned
                machine = machines[0]  # Take the first machine in the list
                JS, JM, JW, job_start_days, job_end_days, worker_start_days, worker_end_days, \
                machine_loads, worker_availability_hours, suc_jobs, prec_jobs, \
                contg_jobs, acontg_jobs, contg_flag = self.assign_jobs_to_machines(
                    job, JS, JM, JW, job_start_days, job_end_days, worker_start_days,
                    worker_end_days, machine_loads, worker_availability_hours,
                    suc_jobs, prec_jobs, contg_jobs, acontg_jobs, contg_flag, machine_flag=machine
                )
        
        metrics = self.calculate_metrics(JS, JM, JW, job_start_days, job_end_days, machine_loads)

        aligned_eligible_machines = np.array([self.eligible_machines[job] for job in JS])
        
        current_value = metrics['weighted_tardiness']

        main_result = {'best_objective_found':current_value, 'run_time':self.run_time, 'key':key, 'JM': JM, 'JS': JS, 'JW': JW, 
            'job_start_days': job_start_days, 'job_end_days': job_end_days, 
            'worker_availability_hours': worker_availability_hours, 
            'worker_start_days': worker_start_days, 'worker_end_days': worker_end_days, 
            'aligned_eligible_machines': aligned_eligible_machines, **metrics}
        
        if check_print == True:

            key = self.local_search(list(JS))

            main_result = self.evaluate_final(key, check_print=False)

            end_cpu = time.process_time() 

            elapsed_cpu = end_cpu - start_cpu  # Compute elapsed CPU time
            self.run_time += elapsed_cpu

            main_result['run_time'] = self.run_time
  
            self.display_individual(main_result)

            # Example of self.args.instance from argparse
            instance_path = self.args.instance

            # Extract filename without extension
            instance_name = os.path.splitext(os.path.basename(instance_path))[0]

            # Print the extracted instance name
            print(instance_name)  # Output: TEST0-50-2-2-J
            excel_file_path = f"Results\\IPMR-P\\{instance_name}_{self.seed}.xlsx"
            # Now, excel_file_path contains the path with the seed value appended
            # print(excel_file_path)

            # Save the results to an Excel file when print_sol is True
            self.save_results_to_excel(main_result, excel_file_path)
        
        return main_result

    def assign_jobs_to_machines(self, job, JS, JM, JW, job_start_days, job_end_days, worker_start_days, worker_end_days, machine_loads, worker_availability_hours, suc_jobs, prec_jobs, contg_jobs, acontg_jobs, contg_flag, machine_flag=None):      
        check_point = True
        # Step 1: Check precedence constraints
        for prec_job, suc_job in self.precedence_matrix:
            if suc_job == job:  
                if prec_job not in prec_jobs:  # Precedence job hasn't finished yet
                    suc_jobs.append(job)  # Store the job for later
                    check_point = False  # Exit without scheduling

        # Check if the job has a contiguity constraint
        for (prev_job, next_job) in self.contiguity_matrix:
            if job == next_job:  # This job is a successor
                if prev_job not in acontg_jobs:  # Its predecessor is not scheduled
                    contg_jobs.append(job)  # Save it for later
                    check_point = False
                elif prev_job in acontg_jobs:
                    if machine_flag == None:
                        check_point = False

        if check_point == True:
            best_machine = None
            min_time_increase = float('inf')
            min_makespan = float('inf')
            # potential_min_availability_sum = float('inf')

            if machine_flag == None:
                remaining_machines = self.eligible_machines[job]
            else:
                remaining_machines = [machine_flag]

            potential_time_increase = self.processing_times[job]
        
            # Step 1: Machine Selection
            for machine in remaining_machines:
                
                potential_start_day = max(machine_loads[machine], self.ready_times[job])
                
                worker_idx, min_availability_sum, earliest_worker_avail = self.assign_worker_to_machine(job, machine, machine_loads[machine], potential_start_day, worker_availability_hours)

                potential_makespan_increase = max(potential_start_day, earliest_worker_avail) + self.processing_times[job]

                if potential_makespan_increase <= min_makespan:
                    best_machine = machine
                    best_worker = worker_idx
                    min_time_increase = potential_time_increase
                    min_makespan = potential_makespan_increase

            result = [job1 for job1, machines in contg_flag.items() if best_machine in machines]

            if result:  # Ensure the list is not empty to avoid IndexError
                force_job = result[0]
                contg_flag[force_job].remove(best_machine)  # Remove the machine from the list

                    # If the list is empty after removal, delete the job from the dictionary
                if not contg_flag[force_job]:  
                    del contg_flag[force_job]

                self.assign_jobs_to_machines(force_job, JS, JM, JW, job_start_days, job_end_days, worker_start_days, worker_end_days, machine_loads, worker_availability_hours, suc_jobs, prec_jobs, contg_jobs, acontg_jobs, contg_flag, machine_flag = best_machine)
                
                self.assign_jobs_to_machines(job, JS, JM, JW, job_start_days, job_end_days, 
                    worker_start_days, worker_end_days, machine_loads, 
                    worker_availability_hours, suc_jobs, prec_jobs, 
                    contg_jobs, acontg_jobs, contg_flag, machine_flag = None)
                
                return JS, JM, JW, job_start_days, job_end_days, worker_start_days, worker_end_days, machine_loads, worker_availability_hours, suc_jobs, prec_jobs, contg_jobs, acontg_jobs, contg_flag

            else:
                force_job = None  # Or handle the case where no job is found

            JS.append(job)
            JM.append(best_machine) 
            JW.append(best_worker)  

            job_start_day = min_makespan - min_time_increase
            job_end_day = min_makespan

            # Record setup and job times
            job_start_days.append(job_start_day)
            job_end_days.append(job_end_day)

            # Update worker assignment
            worker_start_days.append(job_start_day)  
            worker_end_days.append(job_end_day)

            # Update machine load
            machine_loads[best_machine] = job_end_day

            job_start_day = int(job_start_day)
            job_end_day = int(job_end_day)

            for day in range(job_start_day, job_end_day):
                if self.occupation_list[job] == 8:
                    worker_availability_hours[best_worker, day] = 0
                else:
                    worker_availability_hours[best_worker, day] -= 1

            # Only add to prec_jobs if the job appears as a precedence in the matrix
            if any(job == prec_job for prec_job, _ in self.precedence_matrix):
                prec_jobs.append(job)

            # Step 4: Check if any successor jobs can now be scheduled
            for suc_job in list(suc_jobs):  # Iterate over a copy to modify list
                for prec_job, dependent_job in self.precedence_matrix:
                    if dependent_job == suc_job and prec_job in prec_jobs:
                        if suc_job in suc_jobs:  # Ensure `suc_job` is still present
                            suc_jobs.remove(suc_job)  # Remove from waiting list
                            self.assign_jobs_to_machines(suc_job, JS, JM, JW, job_start_days, job_end_days,worker_start_days, worker_end_days, machine_loads, 
                            worker_availability_hours, suc_jobs, prec_jobs, 
                            contg_jobs, acontg_jobs, contg_flag)
                        
            # If the job is scheduled, check if it was an acontg_jobs
            for (prev_job, next_job) in self.contiguity_matrix:
                if job == prev_job:  # This job is a predecessor
                    acontg_jobs.append(job)  # Mark it as scheduled
                    # Ensure its successor (if waiting) is scheduled on the same machine
                    if next_job in contg_jobs:
                        contg_jobs.remove(next_job)  # Remove from pending jobs
                        self.assign_jobs_to_machines(next_job, JS, JM, JW, job_start_days, job_end_days, worker_start_days, worker_end_days, machine_loads, 
                        worker_availability_hours, suc_jobs, prec_jobs, contg_jobs, acontg_jobs, contg_flag, machine_flag = best_machine)

                    elif next_job not in contg_jobs:  # Its successor is not scheduled
                        contg_flag[next_job].append(best_machine)  # Save it for later
                        
        return JS, JM, JW, job_start_days, job_end_days, worker_start_days, worker_end_days, machine_loads, worker_availability_hours, suc_jobs, prec_jobs, contg_jobs, acontg_jobs, contg_flag

    def assign_worker_to_machine(self, job, machine, machine_makespan, potential_start_day, worker_availability_hours):
        
        min_difference = float('inf')
        best_worker = None
        difference = [0] * self.n_workers

        # Find the worker with the earliest availability time among eligible workers
        best_worker = None
        earliest_avail_time = float('inf')
        min_availability_sum = float('inf')  # Track the minimum availability sum
        
        # Start searching from the job's release time
        potential_start_day = int(potential_start_day)
        job_duration = self.processing_times[job]
        
        for worker in self.eligible_machines_workers[machine]:  # Iterate over eligible workers
            if worker in self.eligible_workers_jobs[job]:  # Ensure worker is also eligible for the machine
                copy_potential_start_day = potential_start_day

                while (copy_potential_start_day + job_duration <= 2 * self.n_days and not np.all(worker_availability_hours[worker, copy_potential_start_day:copy_potential_start_day + job_duration] >= self.occupation_list[job])):
                    copy_potential_start_day += 1


                if self.occupation_list[job] < 8:
                    # Calculate the sum of availability hours in the selected window
                    availability_sum = np.sum(worker_availability_hours[worker, copy_potential_start_day:copy_potential_start_day + job_duration])

                    # Update best worker based on the lowest total availability sum
                    if availability_sum < min_availability_sum or (availability_sum == min_availability_sum and copy_potential_start_day < earliest_avail_time):
                        min_availability_sum = availability_sum
                        best_worker = worker
                        earliest_avail_time = copy_potential_start_day
                else:
                    if copy_potential_start_day <= earliest_avail_time:
                        earliest_avail_time = copy_potential_start_day
                        best_worker = worker

        return best_worker, min_availability_sum, earliest_avail_time

    def calculate_metrics(self, JS, JM, JW, job_start_days, job_end_days, machine_loads):

        # Align ready times and delivery times based on JS
        aligned_ready_times = np.array([self.ready_times[job] for job in JS])
        aligned_delivery_times = np.array([self.delivery_times[job] for job in JS])
        aligned_processing_times = np.array([self.processing_times[job] for job in JS])
        aligned_occupation_times = np.array([self.occupation_list[job] for job in JS])

        job_start_days = np.array(job_start_days)
        job_end_days = np.array(job_end_days)

        # Compute total production time
        total_production_time = np.sum(self.occupation_list * self.processing_times)

        machine_completion_times = machine_loads
        makespan = np.max(job_end_days)

        tardiness = np.maximum(0, job_end_days - aligned_delivery_times)
        weights = np.array([self.weight_list[job] for job in JS])
        individual_weighted_tardiness = weights * tardiness
        total_tardiness = np.sum(tardiness)

        # Compute weighted tardiness
        weighted_tardiness = np.sum(individual_weighted_tardiness)

        earliness = np.maximum(0, aligned_ready_times - job_start_days)
        total_earliness = np.sum(earliness)

        starving_times = np.maximum(0, job_start_days - aligned_ready_times)
        float_times = np.maximum(0, aligned_delivery_times - job_end_days)

        # Calculate machine starving time
        machine_starving_times = np.zeros(self.n_machines)

        # Calculate worker utilization
        worker_utilization = np.zeros(self.n_workers)
        
        return {
            'aligned_processing_times': aligned_processing_times,
            'aligned_ready_times': aligned_ready_times,
            'aligned_delivery_times': aligned_delivery_times,
            'total_production_time': total_production_time,
            'aligned_occupation_times': aligned_occupation_times,
            'machine_completion_times': machine_completion_times,
            'makespan': makespan,
            'tardiness': tardiness,
            'individual_weighted_tardiness': individual_weighted_tardiness,
            'total_tardiness': total_tardiness,
            'weighted_tardiness': weighted_tardiness,
            'earliness': earliness,
            'total_earliness': total_earliness,
            'starving_times': starving_times,
            'floating_times': float_times,
            # 'total_worker_idle_time': total_worker_idle_time,
            'machine_starving_times': machine_starving_times,
            'worker_utilization': worker_utilization,    
        }

    def display_individual(self, individual):
        # print("Eligible Machines:\n", individual['aligned_eligible_machines'])
        print("Run time:\n", individual['run_time'])
        print("Processing Times:\n", individual['aligned_processing_times'])
        print("aligned_occupation_times", individual['aligned_occupation_times'])
        # print("Ready Times:\n", self.ready_times)
        # print("Delivery Times:\n", self.delivery_times, "\n")
        print("JM:", individual['JM'])
        print("JS:", individual['JS'])
        print("JW:", individual['JW'], "\n")
        print("Job Start Days:", individual['job_start_days'])
        print("Job End Days:", individual['job_end_days'], "\n")
        print("Worker Start Days:", individual['worker_start_days'])
        print("Worker End Days:", individual['worker_end_days'], "\n")

        print("Total Production Time:", individual['total_production_time'])

        print("Machine Completion Times:", individual['machine_completion_times'])
        print("Makespan:", individual['makespan'],"\n")
        print("Aligned Delivery Time:", individual['aligned_delivery_times'])
        print("Job Tardiness:", individual['tardiness'])
        print("Total Tardiness:", individual['total_tardiness'])
        print("Total Weighted Tardiness:", individual['weighted_tardiness'])

        print("Aligned Ready Time:", individual['aligned_ready_times'])
        print("Job Earliness:", individual['earliness'])
        print("Total Earliness:", individual['total_earliness'],"\n")
        # Find indices where Job Tardiness is non-zero
        non_zero_indices = np.nonzero(individual['earliness'])[0]
        # Extract corresponding Aligned Delivery Time values
        non_zero_aligned_early_time = individual['aligned_ready_times'][non_zero_indices]
        print("Aligned Delivery Time of Early Jobs:", non_zero_aligned_early_time)

        print("Job Waiting Times:", individual['starving_times'])
        print("Job Floating Times:", individual['floating_times'])
        print("Machine Starvation Times:", individual['machine_starving_times'],"\n")
        print("Worker Utilization:", individual['worker_utilization'])


    def save_results_to_excel(self, results, excel_file_path):
        # Convert the results dictionary to a DataFrame
        results_df = pd.DataFrame([results])

        # Ensure that the Excel file exists; if not, create it
        if not os.path.exists(excel_file_path):
            results_df.to_excel(excel_file_path, sheet_name='Results', index=False)
        else:
            # Append the results to the existing Excel file
            with pd.ExcelWriter(excel_file_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Results', header=False, index=False)
import numpy as np
from loguru import logger
from problem import Problem
import pandas as pd
import os
import time

from multiprocessing import Pool, cpu_count

def add_parser_args(parser):
    parser.add_argument('--weight_time', type=float, default=1, help='Weight for time objective')
    parser.add_argument('--weight_tardiness', type=float, default=1, help='Weight for tardiness penalty')
    parser.add_argument('--weight_transition_limit', type=float, default=100, help='Weight for transition limit penalty')


def read_instance(inst_path):
    with open(inst_path, 'r') as fp:
        lines = [line.strip() for line in fp]

    inst = PMSP_instance()

    # Flags to identify the current section
    current_section = None
  
    job_processing_times = []
    initial_setup_times = []
    machine_eligibilities = {}
    personnel_times = []
    personnel_assignments = []
    release_period = []
    delivery_period = []
    release_times = []
    delivery_times = []
    unavailability_start_period = []
    unavailability_end_period = []
    unavailability_start_times = []
    unavailability_end_times = []
    machine_setup_times = {}

    for line in lines:
        line = line.strip()
        
        if line.startswith("Instance Info"):
            current_section = "Instance Info"
            continue
        elif line.startswith("Job Processing Times:"):
            current_section = "Job Processing Times"
            continue
        elif line.startswith("Initial Setup Times:"):
            current_section = "Initial Setup Times"
            continue
        elif line.startswith("Machine Eligibilities:"):
            current_section = "Machine Eligibilities"
            continue
        elif line.startswith("Personnel Times:"):
            current_section = "Personnel Times"
            continue
        elif line.startswith("Personnel Assignments:"):
            current_section = "Personnel Assignments"
            continue
        elif line.startswith("Worker Start Off Periods:"):
            current_section = "Worker Start Off Periods"
            continue
        elif line.startswith("Worker End Off Periods:"):
            current_section = "Worker End Off Periods"
            continue
        elif line.startswith("Worker Start Off Times:"):
            current_section = "Worker Start Off Times"
            continue
        elif line.startswith("Worker End Off Times:"):
            current_section = "Worker End Off Times"
            continue
        elif line.startswith("Release Period:"):
            current_section = "Release Period"
            continue
        elif line.startswith("Delivery Period:"):
            current_section = "Delivery Period"
            continue
        elif line.startswith("Release Times:"):
            current_section = "Release Times"
            continue
        elif line.startswith("Delivery Times:"):
            current_section = "Delivery Times"
            continue
        elif line.startswith("Sequence Dependent Setup Times:"):
            current_section = "Sequence Dependent Setup Times"
            continue
        elif line.startswith("Machine ") and line.split()[1].endswith(':'):
            machine_number = int(line.split()[1][:-1]) - 1
            current_section = f"Machine {machine_number}"
            if machine_number not in machine_setup_times:
                machine_setup_times[machine_number] = []
            continue

        if current_section == "Instance Info":
            inst.instance_number = int(lines[1].split(":")[1].strip())
            inst.number_of_jobs = int(lines[2].split(":")[1].strip())
            inst.number_of_machines = int(lines[3].split(":")[1].strip())
            inst.number_of_weeks = int(lines[4].split(":")[1].strip())
            inst.personnel_available = int(lines[5].split(":")[1].strip())
            inst.max_time_window_length = int(lines[6].split(":")[1].strip())
            inst.time_window_density = float(lines[7].split(":")[1].strip())
            inst.weekly_personnel_availability = int(lines[8].split(":")[1].strip())
            inst.mean_time_value = float(lines[9].split(":")[1].strip())
            inst.upper_bound = float(lines[10].split(":")[1].strip())
            inst.lower_bound = float(lines[11].split(":")[1].strip())
            inst.machine_eligibility_constraint = float(lines[12].split(":")[1].strip())
            inst.num_of_unavb_workers = int(lines[13].split(":")[1].strip())
        elif current_section == "Job Processing Times":
            job_processing_times.append(list(map(float, line.split()[1:])))
        elif current_section == "Initial Setup Times":
            initial_setup_times.append(list(map(float, line.split()[1:])))
        elif current_section == "Machine Eligibilities":
            parts = list(map(int, line.split()))
            machine_number = parts[0] - 1
            if machine_number not in machine_eligibilities:
                machine_eligibilities[machine_number] = []
            machine_eligibilities[machine_number].extend(parts[1:])
        elif current_section == "Personnel Times":
            personnel_times.append(float(line.split()[1]))
        elif current_section == "Personnel Assignments":
            personnel_assignments.append(list(map(float, line.split()[1:])))
        elif current_section == "Worker Start Off Periods":
            unavailability_start_period.append(int(line.split()[1]))
        elif current_section == "Worker End Off Periods":
            unavailability_end_period.append(int(line.split()[1]))
        elif current_section == "Worker Start Off Times":
            unavailability_start_times.append(list(map(float, line.split()[1:])))
        elif current_section == "Worker End Off Times":
            unavailability_end_times.append(list(map(float, line.split()[1:])))
        elif current_section == "Release Period":
            release_period.append(int(line.split()[1]))
        elif current_section == "Delivery Period":
            delivery_period.append(int(line.split()[1]))
        elif current_section == "Release Times":
            release_times.append(list(map(float, line.split()[1:])))
        elif current_section == "Delivery Times":
            delivery_times.append(list(map(float, line.split()[1:])))
        elif current_section.startswith("Machine "):
            machine_setup_times[machine_number].append(list(map(float, line.split()[1:])))

    # Constants
    minutes_per_week = inst.weekly_personnel_availability

    # Convert lists to numpy arrays
    release_period = np.array(release_period).flatten()
    delivery_period = np.array(delivery_period).flatten()
    release_times = np.array(release_times)
    delivery_times = np.array(delivery_times)
    unavailability_start_period = np.array(unavailability_start_period).flatten()
    unavailability_end_period = np.array(unavailability_end_period).flatten()
    unavailability_start_times = np.array(unavailability_start_times)
    unavailability_end_times = np.array(unavailability_end_times)

    # Calculate release parameters
    release_times1 = []
    for i, period in enumerate(release_period):
        release_times1.append(release_times[i, period - 1])
    release_times1 = np.array(release_times1)
    release_parameters = (release_period - 1) * minutes_per_week + release_times1

    # Calculate delivery parameters
    delivery_times1 = []
    for i, period in enumerate(delivery_period):
        delivery_times1.append(delivery_times[i, period - 1])
    delivery_times1 = np.array(delivery_times1)
    delivery_parameters = (delivery_period - 1) * minutes_per_week + delivery_times1

    # Calculate unavailability start parameters
    unavailability_start_times1 = []
    for i, period in enumerate(unavailability_start_period):
        unavailability_start_times1.append(unavailability_start_times[i, int(period - 1)])
    unavailability_start_times1 = np.array(unavailability_start_times1)
    unavailability_start_parameters = (unavailability_start_period-1) * minutes_per_week + unavailability_start_times1

    # Calculate unavailability end parameters
    unavailability_end_times1 = []
    for i, period in enumerate(unavailability_end_period):
        unavailability_end_times1.append(unavailability_end_times[i, int(period - 1)])
    unavailability_end_times1 = np.array(unavailability_end_times1)
    unavailability_end_parameters = (unavailability_end_period-1) * minutes_per_week + unavailability_end_times1

    # Convert lists to numpy arrays
    inst.job_processing_times = np.array(job_processing_times)
    inst.initial_setup_times = np.array(initial_setup_times)
    inst.machine_eligibilities = machine_eligibilities
    inst.personnel_times = np.array(personnel_times)
    inst.personnel_assignments = np.array(personnel_assignments)
    inst.release_period = release_period
    inst.delivery_period = delivery_period
    inst.release_times = release_parameters
    inst.delivery_times = delivery_parameters
    inst.unavailability_start_period = np.array(unavailability_start_period)
    inst.unavailability_end_period = np.array(unavailability_end_period)
    inst.unavailability_start_times = unavailability_start_parameters
    inst.unavailability_end_times = unavailability_end_parameters
    inst.sequence_dependent_setup_times = {k: np.array(v) for k, v in machine_setup_times.items()}

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

class PMSP_instance(object):
    def __init__(self):
        self.instance_number = 0
        self.number_of_jobs = 0
        self.number_of_machines = 0
        self.number_of_weeks = 0
        self.personnel_available = 0
        self.max_time_window_length = 0
        self.time_window_density = 0.0
        self.weekly_personnel_availability = 0
        self.mean_time_value = 0.0
        self.upper_bound = 0.0
        self.lower_bound = 0.0
        self.machine_eligibility_constraint = 0.0

        self.job_processing_times = None
        self.initial_setup_times = None
        self.machine_eligibilities = None
        self.personnel_times = None
        self.personnel_assignments = None
        self.release_period = None
        self.delivery_period = None
        self.release_times = None
        self.delivery_times = None
        self.unavailability_start_period = None
        self.unavailability_end_period = None
        self.unavailability_start_times = None
        self.unavailability_end_times = None
        self.sequence_dependent_setup_times = {}

class PMSP(Problem):
    def __init__(self, args, inst):
        super().__init__(args, inst.number_of_jobs)
        self.inst = inst
        self.args = args  # Store args in the class to access it later
        self.seed = args.seed  # Store the seed value in the class
        self.n_jobs = int(inst.number_of_jobs)
        self.n_machines = int(inst.number_of_machines)
        self.n_workers = int(inst.personnel_available)
        self.n_weeks = int(inst.number_of_weeks)
        self.weekly_personnel_availability = int(inst.weekly_personnel_availability)
        self.tw_density = inst.time_window_density
        self.m_eligibility = inst.machine_eligibility_constraint
        self.n_unavb_worker = inst.num_of_unavb_workers
        self.setup_times = inst.sequence_dependent_setup_times
        self.initial_setup_times = inst.initial_setup_times
        self.processing_times = inst.job_processing_times
        self.eligible_machines = self.get_eligible_machines()
        self.ready_times = inst.release_times
        self.delivery_times = inst.delivery_times
        self.unavailability_start_times = inst.unavailability_start_times  # 1D array for unavailability start times of workers
        self.unavailability_end_times = inst.unavailability_end_times  # 1D array for unavailability end times of workers
        self.best_main_result = None
        self.previous_value = float('inf')
        self.run_time = 0

    def batch_evaluate(self, keys, threads=1, print_sol=False):
        super().batch_evaluate(keys)
        return np.array([self.evaluate(key, print_sol) for key in keys])
    
    def get_eligible_machines(self): 
        eligible_machines = [[] for _ in range(self.n_jobs)]
        for machine, jobs in self.inst.machine_eligibilities.items():
            for job in jobs:
                eligible_machines[job - 1].append(machine + 1)
        return eligible_machines

    def evaluate(self, key, print_sol=False):
        start_cpu = time.process_time()  # Capture start CPU time
        JS = []
        JM = []
        JW = []
        start_times = []
        end_times = []
        worker_start_times = []
        worker_end_times = []
        setup_times_start = []
        setup_times_end = []
        machine_loads = np.zeros(self.n_machines)
        worker_availability = np.zeros(self.n_workers)
        last_jobs_on_machine = [-1] * self.n_machines  # Initialize last jobs processed on each machine
        last_worker_on_machine = [-1] * self.n_machines
        last_machine_for_worker = [-1] * self.n_workers  # Initialize last machine for each worker
        machine_flags = np.zeros(self.n_machines, dtype=int)
        transitions_per_week = {}
        # Initialize the set of jobs
        job_order = list(range(self.n_jobs))
        
        # Sort jobs based on the key
        job_order = np.argsort(key)

        # Assign jobs to machines
        for job in job_order:
            
            JS, JM, JW, start_times, end_times, worker_start_times, worker_end_times, setup_times_start, setup_times_end, machine_loads, worker_availability, last_jobs_on_machine, last_worker_on_machine, last_machine_for_worker, machine_flags, transitions_per_week = self.assign_jobs_to_machines(
                job, JS, JM, JW, start_times, end_times, worker_start_times, worker_end_times, setup_times_start, setup_times_end, machine_loads, worker_availability, last_jobs_on_machine, last_worker_on_machine, last_machine_for_worker, machine_flags, transitions_per_week)
        
        transitions_per_week, machine_flags, worker_durations, machine_durations = self.calculate_worker_transitions(JS, JM, JW, setup_times_start, end_times)

        total_transitions = int(np.sum(machine_flags)) 

        metrics = self.calculate_metrics(JS, start_times, end_times, setup_times_start, setup_times_end)

        # Compute final value using weights
        current_value = (
              self.args.weight_time * metrics['total_production_time']
            + self.args.weight_tardiness * metrics['total_tardiness']
            + self.args.weight_transition_limit * total_transitions
        )
        # Compare the current value with the previous value
        if current_value < self.previous_value:
            self.previous_value = current_value  # Update the previous value
            self.best_main_result = key.copy()  # Save the current metrics as the best metrics

        end_cpu = time.process_time()  # Capture end CPU time
        elapsed_cpu = end_cpu - start_cpu  # Compute elapsed CPU time
        self.run_time += elapsed_cpu
        if print_sol:
            # print('key',self.best_main_result)

            self.evaluate_final(self.best_main_result)

        return current_value  # Return the objective function

    def evaluate_final(self, key):
        JS = []
        JM = []
        JW = []
        start_times = []
        end_times = []
        worker_start_times = []
        worker_end_times = []
        setup_times_start = []
        setup_times_end = []
        machine_loads = np.zeros(self.n_machines)
        worker_availability = np.zeros(self.n_workers)
        last_jobs_on_machine = [-1] * self.n_machines  # Initialize last jobs processed on each machine
        last_worker_on_machine = [-1] * self.n_machines
        last_machine_for_worker = [-1] * self.n_workers  # Initialize last machine for each worker
        machine_flags = np.zeros(self.n_machines, dtype=int)
        transitions_per_week = {}

        # Initialize the set of jobs
        job_order = list(range(self.n_jobs))

        # Sort jobs based on the key
        job_order = np.argsort(key)

        # Assign jobs to machines
        for job in job_order:
            
            JS, JM, JW, start_times, end_times, worker_start_times, worker_end_times, setup_times_start, setup_times_end, machine_loads, worker_availability, last_jobs_on_machine, last_worker_on_machine, last_machine_for_worker, machine_flags, transitions_per_week = self.assign_jobs_to_machines(
                job, JS, JM, JW, start_times, end_times, worker_start_times, worker_end_times, setup_times_start, setup_times_end, machine_loads, worker_availability, last_jobs_on_machine, last_worker_on_machine, last_machine_for_worker, machine_flags, transitions_per_week)

        machine_flags = np.zeros(self.n_machines, dtype=int)
        
        transitions_per_week, machine_flags, worker_durations, machine_durations = self.calculate_worker_transitions(JS, JM, JW, setup_times_start, end_times)

        total_transitions = int(np.sum(machine_flags))
        
        metrics = self.calculate_metrics(JS, start_times, end_times, setup_times_start, setup_times_end)

        # post_metrics = self.calculate_post_metrics(JS, JM, JW, start_times, end_times, setup_times_start, setup_times_end, machine_loads, metrics['aligned_ready_times'])

        current_value = (
              self.args.weight_tardiness * metrics['total_production_time']
            + self.args.weight_tardiness * metrics['total_tardiness']
            + metrics['total_earliness']
            + self.args.weight_transition_limit * total_transitions
        )
        
        main_result = {'best_objective_found':current_value, 'run_time':self.run_time, 'JM': JM, 'JS': JS, 'JW': JW, 'start_times': start_times, 'end_times': end_times, 'worker_start_times': worker_start_times, 'worker_end_times': worker_end_times,
                'setup_times_start': setup_times_start, 'setup_times_end': setup_times_end, 'total_transitions': total_transitions, 'transitions_per_week': transitions_per_week, 
                'worker_durations': worker_durations, 'machine_durations': machine_durations, **metrics}
        
        self.display_individual(main_result)

        # Directly append the seed to the file path
        excel_file_path = f"Results\\PMSP\\J{self.n_jobs}_M{self.n_machines}_P{self.n_workers}_W{self.n_weeks}_TW{self.tw_density}_ME{self.m_eligibility}_AP{self.n_unavb_worker}_Seed{self.seed}.xlsx"

        # Now, excel_file_path contains the path with the seed value appended
        # print(excel_file_path)

        # Save the results to an Excel file when print_sol is True
        # self.save_results_to_excel(main_result, excel_file_path)

    def assign_jobs_to_machines(self, job, JS, JM, JW, start_times, end_times, worker_start_times, worker_end_times, setup_times_start, setup_times_end, machine_loads, worker_availability, last_jobs_on_machine, last_worker_on_machine, last_machine_for_worker, machine_flags, transitions_per_week):

        # Make a local copy of eligible machines to modify if needed
        remaining_machines = self.eligible_machines[job].copy()

        forced_assignment = False  # Flag to indicate forced assignment
        first_machine = None  # Track the first valid machine assignment
        first_worker = None  # Track the first valid worker assignment
        first_worker_earliest_worker_avail = None


        while remaining_machines:  # Continue as long as there are machines left to assign the job
            # Sort jobs by the number of eligible machines (least flexible first)
            best_machine = None
            min_time_increase = float('inf')
            min_makespan_increase = float('inf')

            # Step 1: Machine Selection
            for machine in remaining_machines:
                machine_idx = machine - 1
                setup_time = self.initial_setup_times[job][machine_idx] if machine_loads[machine_idx] == 0 else self.setup_times[machine_idx][last_jobs_on_machine[machine_idx]][job]
                if (machine_loads[machine_idx] + setup_time) <= self.ready_times[job]:
                    potential_makespan_increase = self.ready_times[job] + self.processing_times[job][machine_idx] # Makespan Minimization
                else:
                    potential_makespan_increase = machine_loads[machine_idx] + self.processing_times[job][machine_idx] + setup_time
                potential_time_increase = self.processing_times[job][machine_idx] + setup_time

                if potential_time_increase == min_time_increase:
                    if potential_makespan_increase <= min_makespan_increase:
                        best_machine = machine
                        assigned_setup_time = setup_time
                        assigned_processing_time = self.processing_times[job][machine_idx]
                        min_time_increase = potential_time_increase
                        min_makespan_increase = potential_makespan_increase

                elif potential_time_increase < min_time_increase:
                    best_machine = machine
                    assigned_setup_time = setup_time
                    assigned_processing_time = self.processing_times[job][machine_idx]
                    min_time_increase = potential_time_increase
                    min_makespan_increase = potential_makespan_increase

            # Track the first machine assigned
            if first_machine is None:
                first_machine = best_machine

            worker_idx, earliest_worker_avail = self.assign_worker_to_machine(last_worker_on_machine[best_machine-1], worker_availability, machine_loads[best_machine - 1], assigned_processing_time, assigned_setup_time)
            best_worker = worker_idx
            best_worker_earliest_worker_avail = earliest_worker_avail
            
            if first_worker is None:
                first_worker = worker_idx
                first_worker_earliest_worker_avail = earliest_worker_avail

            machine_idx = best_machine - 1
            setup_start = max(machine_loads[machine_idx], earliest_worker_avail)

            current_week = int(setup_start // self.weekly_personnel_availability)
            if current_week not in transitions_per_week:
                transitions_per_week[current_week] = [0] * self.n_machines

            # Check if this is the first time the machine is being assigned in the week
            if last_machine_for_worker[worker_idx] is None:
                transitions_per_week[current_week][machine_idx] += 1
            else:
                if last_machine_for_worker[worker_idx] != best_machine:
                    transitions_per_week[current_week][machine_idx] += 1

            # Check if transitions exceed the limit
            if last_machine_for_worker[worker_idx] is not best_machine and transitions_per_week[current_week][machine_idx] > 2:
                # Exceeds allowed transitions, revert changes and restart
                transitions_per_week[current_week][machine_idx] -= 1  # Revert transition count
                # Remove this machine from the list of remaining machines
                remaining_machines.remove(best_machine) 
            else:
                # Valid machine found, break the loop and proceed with final updates
                break

            # If no remaining machines are left, force assignment to the last checked machine
            if not remaining_machines:
                forced_assignment = True
                break  # Break the loop to force assign to the last machine

        # # Step 4.1: Perform final updates after determining the eligible machine
        if forced_assignment:
            # If forced assignment, assign the first machine encountered
            best_machine = first_machine
            best_worker = first_worker
            best_worker_earliest_worker_avail = first_worker_earliest_worker_avail


        # Step 4: Perform final updates after determining the eligible machine  
        # if best_machine is not None or forced_assignment:
        if best_machine is not None:
        # Update main lists only once a valid machine is found or forced assignment is made
            machine_idx = best_machine - 1
            setup_start = max(machine_loads[machine_idx], best_worker_earliest_worker_avail)
            JS.append(job + 1)  # +1 to match 1-based indexing
            JM.append(best_machine) 
            setup_end = setup_start + assigned_setup_time
            start_time = max(setup_end, self.ready_times[job])  # Consider worker availability 
            end_time = start_time + self.processing_times[job][machine_idx]

            # Record setup and job times
            setup_times_start.append(setup_start)
            setup_times_end.append(setup_end)
            start_times.append(start_time)
            end_times.append(end_time)

            # Update worker assignment
            JW.append(best_worker + 1)  # +1 to match 1-based indexing
            worker_start_times.append(setup_start)  
            worker_end_times.append(end_time)
            worker_availability[worker_idx] = end_time

            # Update machine load
            machine_loads[machine_idx] = end_time
            last_jobs_on_machine[machine_idx] = job  # Update the last job processed on this machine
            last_machine_for_worker[worker_idx] = best_machine
            last_worker_on_machine[machine_idx] = best_worker

        if forced_assignment:
            transitions_per_week[current_week][machine_idx] += 1  # Revert transition count
            machine_flags[machine_idx] += 1

        return JS, JM, JW, start_times, end_times, worker_start_times, worker_end_times, setup_times_start, setup_times_end, machine_loads, worker_availability, last_jobs_on_machine, last_worker_on_machine, last_machine_for_worker, machine_flags, transitions_per_week

    def assign_worker_to_machine(self, last_worker_on_machine, worker_availability, machine_makespan, assigned_processing_time, assigned_setup_time):

        min_difference = float('inf')
        best_worker = None
        # candidate_worker = None
        difference = [0] * self.n_workers
        
        if last_worker_on_machine == -1:
    
            for worker_idx, avail_time in enumerate(worker_availability):
                difference [worker_idx] = abs(machine_makespan - avail_time)

                if difference [worker_idx] < min_difference:
                    min_difference = difference [worker_idx]
                    best_worker = worker_idx
   
        else:
            best_worker =  last_worker_on_machine

        if self.unavailability_end_times[best_worker] != 0:
            if worker_availability[best_worker] + assigned_processing_time + assigned_setup_time >= self.unavailability_start_times[best_worker] and worker_availability[best_worker] < self.unavailability_end_times[best_worker]:
                worker_availability[best_worker] = self.unavailability_end_times[best_worker]

        earliest_worker_avail = worker_availability[best_worker]
        return best_worker, earliest_worker_avail

    def calculate_worker_transitions(self, JS, JM, JW, setup_times_start, end_times):
        machine_flags = np.zeros(self.n_machines, dtype=int)
        transitions_per_week = {}
        worker_durations = {worker: [] for worker in range(self.n_workers)}
        machine_durations = {machine: [] for machine in range(1, self.n_machines + 1)}

        last_machine_for_worker = [None] * self.n_workers  # Initialize last machine for each worker
        first_job_start_time_for_worker = [None] * self.n_workers  # Initialize the start time of the first job on the current machine for each worker
        last_job_end_time_for_worker = [None] * self.n_workers  # Initialize the end time of the last job on the current machine for each worker

        for idx in range(len(JS)):
            worker_idx = JW[idx] - 1
            machine_idx = JM[idx] - 1
            setup_start = setup_times_start[idx]
            end_time = end_times[idx]

            current_week = int(setup_start // self.weekly_personnel_availability)
            transitions_per_week.setdefault(current_week, [0] * self.n_machines)

            # Check if the worker is transitioning to a different machine
            if last_machine_for_worker[worker_idx] is not None and last_machine_for_worker[worker_idx] != JM[idx]:
                # Calculate the duration on the previous machine
                duration = last_job_end_time_for_worker[worker_idx] - first_job_start_time_for_worker[worker_idx]
                worker_durations[worker_idx].append((last_machine_for_worker[worker_idx], duration))
                machine_durations[last_machine_for_worker[worker_idx]].append((worker_idx + 1, duration))
                # Reset the start time for the new machine
                first_job_start_time_for_worker[worker_idx] = setup_start
            elif last_machine_for_worker[worker_idx] is None:
                # This is the first job for this worker, initialize the start time
                first_job_start_time_for_worker[worker_idx] = setup_start

            # Update the end time for the current job
            last_job_end_time_for_worker[worker_idx] = end_time

            # Check if this is the first time the machine is being assigned in the week
            if last_machine_for_worker[worker_idx] is None:
                transitions_per_week[current_week][machine_idx] += 1

            if last_machine_for_worker[worker_idx] is not None and last_machine_for_worker[worker_idx] != JM[idx]:
                if transitions_per_week[current_week][machine_idx] >= 2:
                    transitions_per_week[current_week][machine_idx] += 1
                    machine_flags[machine_idx] += 1
                else:
                    transitions_per_week[current_week][machine_idx] += 1

            last_machine_for_worker[worker_idx] = JM[idx]

        # Calculate the duration for the last assignment of each worker
        for worker_idx in range(self.n_workers):
            if last_machine_for_worker[worker_idx] is not None:
                duration = last_job_end_time_for_worker[worker_idx] - first_job_start_time_for_worker[worker_idx]
                worker_durations[worker_idx].append((last_machine_for_worker[worker_idx], duration))
                machine_durations[last_machine_for_worker[worker_idx]].append((worker_idx + 1, duration))

        return transitions_per_week, machine_flags, worker_durations, machine_durations

    def calculate_metrics(self,JS, start_times, end_times, setup_times_start, setup_times_end):
        # Align ready times and delivery times based on JS
        aligned_ready_times = np.array([self.ready_times[job - 1] for job in JS])
        aligned_delivery_times = np.array([self.delivery_times[job - 1] for job in JS])

        start_times = np.array(start_times)
        end_times = np.array(end_times)
        setup_times_start = np.array(setup_times_start)
        setup_times_end = np.array(setup_times_end)

        total_processing_time = np.sum(end_times - start_times)
        total_setup_time = np.sum(setup_times_end - setup_times_start)
        total_production_time = total_processing_time + total_setup_time

        tardiness = np.maximum(0, end_times - aligned_delivery_times)
        total_tardiness = np.sum(tardiness)
        earliness = np.maximum(0, aligned_ready_times - start_times)
        total_earliness = np.sum(earliness)

        # return {
        #     'aligned_ready_times': aligned_ready_times,
        #     'aligned_delivery_times': aligned_delivery_times,
        #     'total_production_time': total_production_time,
        #     'total_processing_time': total_processing_time,
        #     'total_setup_time': total_setup_time,
        #     'tardiness': tardiness,
        #     'total_tardiness': total_tardiness,
        #     'earliness': earliness,
        #     'total_earliness': total_earliness,
        # }

        return {
            'total_production_time': total_production_time,
            'total_tardiness': total_tardiness,
            'total_earliness': total_earliness,
        }

    def calculate_post_metrics(self, JS, JM, JW, start_times, end_times, setup_times_start, setup_times_end, machine_loads, aligned_ready_times):

        aligned_eligible_machines = [(job, self.eligible_machines[job - 1]) for job in JS]

        machine_completion_times = machine_loads
        makespan = np.max(end_times)

        starving_times = np.maximum(0, start_times - aligned_ready_times)

        # Calculate machine starving time
        machine_starving_times = np.zeros(self.n_machines)
        for m in range(self.n_machines):
            machine_processing_time = np.sum([self.processing_times[job - 1][m] for job in range(1, self.n_jobs + 1) if JM[job - 1] == m + 1])
            
            machine_setup_time = 0
            jobs_on_machine = [job for job in range(1, self.n_jobs + 1) if JM[job - 1] == m + 1]
            if len(jobs_on_machine) > 1:
                job_pairs = zip(jobs_on_machine[:-1], jobs_on_machine[1:])
                machine_setup_time = sum(self.setup_times[m][j1 - 1][j2 - 1] for j1, j2 in job_pairs)
        
            machine_starving_times[m] = machine_completion_times[m] - (machine_processing_time + machine_setup_time)

        # Calculate worker utilization
        worker_utilization = np.zeros(self.n_workers)
        for w in range(self.n_workers):
            worker_jobs = [job for job in range(self.n_jobs) if JW[job] == w + 1]
            worker_utilization[w] = np.sum([(end_times[job] - start_times[job]) + (setup_times_end[job] - setup_times_start[job]) for job in worker_jobs])

        total_worker_idle_time = makespan * self.n_workers - np.sum(worker_utilization)

        return {
            'aligned_eligible_machines': aligned_eligible_machines,
            'machine_completion_times': machine_completion_times,
            'makespan': makespan,
            'starving_times': starving_times,
            'total_worker_idle_time': total_worker_idle_time,
            'machine_starving_times': machine_starving_times,
            'worker_utilization': worker_utilization,   
        }

    def display_individual(self, individual):
        # Extract data from individual
        # print("Eligible Machines:")
        # for job, machines in individual['aligned_eligible_machines']:
        #     print(f"Job {job}: {machines}")
        # print("Processing Times:\n", self.processing_times)
        # print("Setup Times:\n", self.setup_times)
        # print("Ready Times:\n", individual['aligned_ready_times'])
        # print("Delivery Times:\n", individual['aligned_delivery_times'], "\n")
        print("Run time:\n", individual['run_time'])
        individual['JS'] = [int(x) for x in individual['JS']]
        print("JM:", individual['JM'])
        print("JS:", individual['JS'])
        print("JW:", individual['JW'])
        # print("Setup Start Times:", individual['setup_times_start'])
        # print("Setup End Times:", individual['setup_times_end'], "\n")
        # print("Job Start Times:", individual['start_times'])
        # print("Job End Times:", individual['end_times'], "\n")
        # print("Worker Start Times:", individual['worker_start_times'])
        # print("Worker End Times:", individual['worker_end_times'], "\n")
        print("Total Production Time:", individual['total_production_time'])
        # print("Total Processing Time:", individual['total_processing_time'])
        # print("Total Setup Time:", individual['total_setup_time'],"\n")
        # print("Machine Completion Times:", individual['machine_completion_times'])
        # print("Makespan:", individual['makespan'],"\n")
        # print("Job Tardiness:", individual['tardiness'])
        print("Total Tardiness:", individual['total_tardiness'])
        # print("Job Earliness:", individual['earliness'])
        print("Total Earliness:", individual['total_earliness'],"\n")
        # print("Job Starvation Times:", individual['starving_times'])
        # print("Machine Starvation Times:", individual['machine_starving_times'],"\n")
        # print("Worker Durations:", individual['worker_durations'])
        # print("Machine Durations:", individual['machine_durations'])

        # for worker_idx, durations in individual['worker_durations'].items():
        #     print(f"Worker {worker_idx + 1}:")
        #     for duration in durations:
        #         print(f"  Machine: {duration[0]}, Total Duration: {duration[1]}")

        # for machine_idx, durations in individual['machine_durations'].items():
        #     print(f"Machine {machine_idx}:")
        #     for duration in durations:
        #         print(f" Worker {duration[0]}: {duration[1]}")
        
        print("Transitions per Week:", individual['transitions_per_week'])
        # print("Worker Transition:", individual['machine_flags'])
        print("Total Transition:", individual['total_transitions'])

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
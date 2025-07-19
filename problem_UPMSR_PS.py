import argparse
import numpy as np
from loguru import logger
from problem import Problem
import pandas as pd
import os
import copy

def add_parser_args(parser):
    parser.add_argument('--weight_time', type=float, default=1, help='Weight for time objective')

def read_instance(inst_path):

    with open(inst_path, 'r') as file:
        lines = file.readlines()

    inst = UPMSR_PS_instance()

    # Parse general information
    inst.n_jobs, inst.n_machines, inst.num_stages = map(int, lines[0].split())
    inst.n_machines_stage = int(lines[1].strip())

    # Initialize variables
    job_processing_times = [[None for _ in range(inst.n_machines)] for _ in range(inst.n_jobs)]
    setup_times = {}
    resources = {}
    resource_availability = {}
    current_section = None
    machine_id = None
    resource_type = None  # To track the current resource type

    # Parse the remaining lines
    job_index = 0
    for line in lines[2:]:
        line = line.strip()

        if line == "SSD":
            current_section = "SSD"
            continue
        elif line == "Resources":
            current_section = "Resources"
            continue
        elif line in ["process", "setup"] and current_section == "Resources":
            resource_type = line
            if resource_type in ["process", "shared_process"]:
                resources[resource_type] = []  # Initialize as a list
            else:  # setup or shared_setup
                resources[resource_type] = {}
            resource_availability[resource_type] = None  # Placeholder for availability
            continue
        elif line.startswith("M") and current_section == "SSD":
            machine_id = int(line[1:])
            setup_times[machine_id] = []
            continue
        elif line.startswith("M") and current_section == "Resources" and resource_type == "setup":
            machine_id = int(line[1:])
            resources[resource_type][machine_id] = []
            continue
        elif line.startswith("R"):
            continue  # Skip resource ID (e.g., R0)
        elif line.isdigit() and current_section == "Resources":
            resource_availability[resource_type] = int(line)
            continue

        # Process job processing times
        if current_section is None:
            parts = list(map(int, line.split()))
            for i in range(0, len(parts), 2):
                machine = parts[i]
                time = parts[i + 1]
                job_processing_times[job_index][machine] = time
            job_index += 1
        elif current_section == "SSD" and machine_id is not None:
            setup_times[machine_id].append(list(map(int, line.split())))
        elif current_section == "Resources" and resource_type == "process":
            parts = list(map(int, line.split()))
            row = [parts[i + 1] for i in range(0, len(parts), 2)]
            resources[resource_type].append(row)
        elif current_section == "Resources" and resource_type == "setup" and machine_id is not None:
            resources[resource_type][machine_id].append(list(map(int, line.split())))

    # Convert lists to numpy arrays
    inst.job_processing_times = np.array(job_processing_times)
    inst.setup_times = {machine: np.array(times) for machine, times in setup_times.items()}
    inst.resources = {
        resource_type: {
            machine: np.array(matrix) if isinstance(matrix, list) else np.array(matrix)
            for machine, matrix in machines.items()
        } if isinstance(machines, dict) else np.array(machines)
        for resource_type, machines in resources.items()
    }
    inst.resource_availability = resource_availability

    # logger.info(f"Instance Number: {inst.instance_number}")

    # Logging the parsed instance data
    # logger.info("Parsed Instance Data:")
    # logger.info(f"Number of Jobs: {inst.n_jobs}")
    # logger.info(f"Number of Machines: {inst.n_machines}")
    # logger.info(f"Number of Stages: {inst.num_stages}")
    # logger.info(f"Number of Machines per Stage: {inst.n_machines_stage}")
    
    # # Log job processing times
    # logger.info("Job Processing Times:")
    # for job_idx, times in enumerate(inst.job_processing_times):
    #     logger.info(f"Job {job_idx + 1}: {times}")

    # # Log initial setup times
    # logger.info("Sequence-Dependent Setup Times:")
    # for machine, setup_matrix in inst.setup_times.items():
    #     logger.info(f"Machine {machine}:")
    #     logger.info(setup_matrix)

    # # Log resource data
    # logger.info("Resources and Availability:")
    # for resource_type, resource_data in inst.resources.items():
    #     logger.info(f"Resource Type: {resource_type}")
    #     if isinstance(resource_data, dict):  # For setup and shared_setup resources
    #         for machine, matrix in resource_data.items():
    #             logger.info(f"  Machine {machine}:")
    #             logger.info(matrix)
    #     else:  # For process and shared_process resources
    #         logger.info(resource_data)

    # # Log resource availability
    # logger.info("Resource Availability:")
    # for resource_type, availability in inst.resource_availability.items():
    #     logger.info(f"  {resource_type}: {availability}")

    return inst

class UPMSR_PS_instance(object):
    def __init__(self):
        """Initialize the pmspr_ps class with default values."""
        self.n_jobs = 0
        self.n_machines = 0
        self.setup_times = {}  # 3D array for job sequence and machine-dependent setup times
        self.processing_times = None  # 2D array for job and machine-dependent processing times
        self.resources = {}  # Dictionary containing all resource data by type
        self.resource_availability = {}  # Dictionary containing resource availability for each type

class UPMSR_PS(Problem):
    def __init__(self, args, inst):
        super().__init__(args, inst.n_jobs)
        self.inst = inst
        self.args = args  # Store args in the class to access it later
        self.seed = args.seed  # Store the seed value in the class
        self.n_jobs = inst.n_jobs
        self.n_machines = inst.n_machines
        self.processing_times = inst.job_processing_times  # 2D array for job and machine-dependent processing times
        self.resources = inst.resources  # Dictionary containing all resource data by type
        self.resource_availability = inst.resource_availability  # Dictionary containing resource availability for each type
        self.best_main_result = None
        self.previous_value = float('inf')
        self.count = 0

        # Initialize setup times with zeros and fill it with the parsed data
        self.setup_times = np.zeros((self.n_jobs, self.n_jobs, self.n_machines))
        for machine in range(self.n_machines):
            if machine in inst.setup_times:
                self.setup_times[:, :, machine] = inst.setup_times[machine]

    def batch_evaluate(self, keys, threads=1, print_sol=False):
        super().batch_evaluate(keys)
        return np.array([self.evaluate(key, print_sol) for key in keys])
    
    def batch_evaluate2(self, keys, threads=1, print_sol=True):
        super().batch_evaluate(keys)
        return np.array([self.evaluate_final(key, print_sol) for key in keys])

    def warm_start(self):
        # Initialize setup_times and process_times
        setup_times = np.zeros((self.n_machines, self.n_jobs, 2))  # [n_machines, n_jobs, (start_time, end_time)]
        process_times = np.zeros((self.n_machines, self.n_jobs, 2))  # Same shape as setup_times

        # Machine and worker load tracking
        machine_loads = np.zeros(self.n_machines)  # Machine workloads
        # Dictionary to store job sequences for each machine
        machine_job_sequence = {machine: [] for machine in range(self.n_machines)}
        # Initialize the set of jobs
        job_order = list(range(self.n_jobs))
        
        threshold_size = 1

        machine_job_sequence, machine_loads = self.constructive_heuristic(job_order, threshold_size)

        # Perform local search to improve the solution
        machine_job_sequence, machine_loads = self.local_search(machine_job_sequence, machine_loads)
    
        setup_times, process_times, machine_loads = self.calculate_timings(machine_job_sequence, setup_times, process_times)

        machine_job_sequence, machine_loads, setup_times, process_times = self.repair_phase(machine_job_sequence, machine_loads, setup_times, process_times)

        main_result = {'machine_job_sequence': machine_job_sequence, 'process_times': process_times,'setup_times': setup_times}

        warm_solutions = self.calculate_initial_sequence(main_result)

        metric = self.calculate_metrics(setup_times, process_times, machine_loads, machine_job_sequence)

        main_result = {'machine_job_sequence': machine_job_sequence, 'process_times': process_times,'setup_times': setup_times, **warm_solutions, **metric}

        return main_result
    
    def evaluate(self, key, print_sol=False):
        # Job, Machine, Worker schedules
        JS = []  # Job start times
        JM = []  # Machine assignments

        # Initialize setup_times and process_times
        setup_times = np.zeros((self.n_machines, self.n_jobs, 2))  # [n_machines, n_jobs, (start_time, end_time)]
        process_times = np.zeros((self.n_machines, self.n_jobs, 2))  # Same shape as setup_times

        # Machine and worker load tracking
        machine_loads = np.zeros(self.n_machines)  # Machine workloads
        last_jobs_on_machine = [-1] * self.n_machines
        # Dictionary to store job sequences for each machine
        machine_job_sequence = {machine: [] for machine in range(self.n_machines)}
        # Initialize the set of jobs
        job_order = list(range(self.n_jobs))
        
        # Sort jobs based on the key
        job_order = np.argsort(key)

        total_resource_requirement = 0

        for job in job_order:
            JS, JM, setup_times, process_times, machine_loads, last_jobs_on_machine, machine_job_sequence, total_resource_requirement = self.assign_jobs_to_machines(
                job, JS, JM, setup_times, process_times, machine_loads, last_jobs_on_machine, machine_job_sequence, total_resource_requirement)        

        setup_times, process_times, machine_loads = self.calculate_timings(machine_job_sequence, setup_times, process_times)

        metric = self.calculate_metrics(setup_times, process_times, machine_loads, machine_job_sequence)

        # Compute final value using weights
        current_value = 1 * metric['makespan'] + 0.25* metric['total_production_time'] + 0.0 * total_resource_requirement

        # Compare the current value with the previous value
        if metric['makespan'] <= self.previous_value:
            self.previous_value = metric['makespan']  # Update the previous value

            # Store the top 50 keys
            if not hasattr(self, "top_keys"):
                self.top_keys = []  # Initialize the list if it doesn't exist


            self.top_keys.append((current_value, key.copy()))  # Store the value and key

            # Keep only the top 50 keys based on their values (smallest first)
            self.top_keys.sort(key=lambda x: x[0])
            if len(self.top_keys) > 2:
                self.top_keys.pop()  # Remove the worst key if more than 50

        if print_sol:
            best_evaluation = float("inf")  # Assuming lower is better

            for _, key_instance in self.top_keys:
                evaluation = self.evaluate_final(key_instance, check_print=False)
                if evaluation < best_evaluation:
                    best_evaluation = evaluation
                    best_key_instance = key_instance

            self.evaluate_final(best_key_instance, check_print=True)

        return current_value  # Return the objective function

    def evaluate_final(self, key, check_print):
        # Job, Machine, Worker schedules
        JS = []  # Job start times
        JM = []  # Machine assignments

        # Initialize setup_times and process_times
        setup_times = np.zeros((self.n_machines, self.n_jobs, 2))  # [n_machines, n_jobs, (start_time, end_time)]
        process_times = np.zeros((self.n_machines, self.n_jobs, 2))  # Same shape as setup_times

        # Machine and worker load tracking
        machine_loads = np.zeros(self.n_machines)  # Machine workloads
        last_jobs_on_machine = [-1] * self.n_machines
        # Dictionary to store job sequences for each machine
        machine_job_sequence = {machine: [] for machine in range(self.n_machines)}
        # Initialize the set of jobs
        job_order = list(range(self.n_jobs))

        total_resource_requirement = 0
        
        # Sort jobs based on the key
        job_order = np.argsort(key)

        for job in job_order:
            JS, JM, setup_times, process_times, machine_loads, last_jobs_on_machine, machine_job_sequence, total_resource_requirement = self.assign_jobs_to_machines(
                job, JS, JM, setup_times, process_times, machine_loads, last_jobs_on_machine, machine_job_sequence, total_resource_requirement)
            
        # Perform local search to improve the solution
        machine_job_sequence, machine_loads = self.local_search(machine_job_sequence, machine_loads)

        setup_times, process_times, machine_loads = self.calculate_timings(machine_job_sequence, setup_times, process_times)

        machine_job_sequence, machine_loads, setup_times, process_times = self.repair_phase(machine_job_sequence, machine_loads, setup_times, process_times)

        metric = self.calculate_metrics(setup_times, process_times, machine_loads, machine_job_sequence)
        
        # Compute final value using weights
        current_value = 1 * metric['makespan'] + 0.25* metric['total_production_time'] + 0.0 * total_resource_requirement

        if check_print == True:
        
            main_result = {'best_objective_found':current_value, 'JM': JM, 'JS': JS, 'machine_job_sequence': machine_job_sequence, 'machine_loads': machine_loads, 'process_times': process_times,
                    'setup_times': setup_times, **metric}

            self.display_individual(main_result)

            # Example of self.args.instance from argparse
            instance_path = self.args.instance

            # Extract filename without extension
            instance_name = os.path.splitext(os.path.basename(instance_path))[0]

            # Print the extracted instance name
            print(instance_name)  # Output: TEST0-50-2-2-J
            excel_file_path = f"Results\\UPMSR_PS\\{instance_name}_{self.seed}.xlsx"

            # Directly append the seed to the file path
            # excel_file_path = f'Results\\J{self.n_jobs}_M{self.n_machines}_RS{self.resource_availability["setup"]}_RP{self.resource_availability["process"]}_Seed{self.seed}.xlsx'

            # Now, excel_file_path contains the path with the seed value appended
            # print(excel_file_path)

            # Save the results to an Excel file when print_sol is True
            self.save_results_to_excel(main_result, excel_file_path)

        return metric['makespan']

    def constructive_heuristic(self, job_order, threshold_size):

        # Machine and worker load tracking
        machine_loads = np.zeros(self.n_machines)  # Machine workloads
        last_jobs_on_machine = [-1] * self.n_machines

        # Dictionary to store job sequences for each machine
        machine_job_sequence = {machine: [] for machine in range(self.n_machines)}
        
        while len(job_order) > 0:
            best_machine = None
            best_job = None
            assigned_setup_time = 0
            assigned_processing_time = 0

            job_order_np = np.array(job_order)

            # Combine setup times and processing times into a single matrix
            combined_matrix = np.full((len(job_order), self.n_machines), float('inf'))

            for job_idx, job in enumerate(job_order_np):
                for machine_idx in range(self.n_machines):
                    # Determine setup time
                    if machine_loads[machine_idx] == 0:  # First-time assignment to the machine
                        setup_time = self.setup_times[job, job, machine_idx]
                        setup_resource = self.resources["setup"][machine_idx][job][job]
                    else:  # Sequence-dependent setup time
                        setup_time = self.setup_times[last_jobs_on_machine[machine_idx], job, machine_idx]
                        setup_resource = self.resources["setup"][machine_idx][last_jobs_on_machine[machine_idx]][job]

                    # Add processing time (2D matrix) to the setup time
                    processing_time = self.processing_times[job, machine_idx]
                    process_resource = self.resources["process"][job][machine_idx]

                    resource_increase = setup_resource + process_resource
                    makespan_increase = setup_time + processing_time + machine_loads[machine_idx]
                    # linear_combination = 0.7 * makespan_increase + 0.3* resource_increase

                    combined_matrix[job_idx, machine_idx] = makespan_increase

            # Sort Cij to create RCL
            sorted_indices = np.argsort(combined_matrix, axis=None)
            
            # Build RCL with top `threshold_size` elements
            RCL = []
            for idx in sorted_indices[:threshold_size]:
                RCL.append(np.unravel_index(idx, combined_matrix.shape))
                
            # Define the probability distribution type (uniform or decreasing)
            distribution_type = 'decreasing'  # or 'uniform'

            if distribution_type == 'uniform':
                # Uniform distribution: equal probability for all elements in RCL
                job_idx, machine_idx = RCL[np.random.randint(len(RCL))]

            elif distribution_type == 'decreasing':
                # Decreasing distribution: probabilities based on position in RCL
                s = len(RCL)
                probabilities = [(s - k + 1) * 2 / (s * (s + 1)) for k in range(1, s + 1)]
                
                # Randomly select an index based on the decreasing probabilities
                selected_index = np.random.choice(range(s), p=probabilities)
                job_idx, machine_idx = RCL[selected_index]
                
            # Find the job-machine pair with the minimum combined time
            # job_idx, machine_idx = np.unravel_index(np.argmin(combined_matrix), combined_matrix.shape)

            best_job = job_order_np[job_idx]
            best_machine = machine_idx
            # makespan = combined_matrix[job_idx, machine_idx]

            # Extract setup and processing times for the selected job and machine
            if machine_loads[best_machine] == 0:
                assigned_setup_time = self.setup_times[best_job, best_job, best_machine]                
            else:
                assigned_setup_time = self.setup_times[last_jobs_on_machine[best_machine], best_job, best_machine]
            
            assigned_processing_time = self.processing_times[best_job, best_machine]
            # Update machine load and last job on the machine
            machine_loads[best_machine] += assigned_setup_time + assigned_processing_time
            last_jobs_on_machine[best_machine] = best_job

            # Add the job to the sequence of the assigned machine
            machine_job_sequence[best_machine].append(best_job)

            # Remove the assigned job from the pending job list
            job_order.pop(job_idx)

        return machine_job_sequence, machine_loads
    
    def assign_jobs_to_machines(self, job, JS, JM, setup_times, process_times, machine_loads, last_jobs_on_machine, machine_job_sequence, total_resource_requirement):  
        # Local variables
        remaining_machines = list(range(0, self.n_machines))  # All machines initially eligible

        best_machine = None
        min_makespan_increase = float('inf')
        min_time_increase = float('inf')
        assigned_setup_time = 0
        assigned_processing_time = 0
        min_resource_increase = float('inf')
        # min_linear_combination = float('inf')

        # Evaluate each machine for the current job
        for machine_idx in remaining_machines:

            # Determine setup time
            if machine_loads[machine_idx] == 0:  # Machine is idle
                setup_time = self.setup_times[job, job, machine_idx]
                setup_resource = self.resources["setup"][machine_idx][job][job]

            else:  # Sequence-dependent setup
                setup_time = self.setup_times[last_jobs_on_machine[machine_idx], job, machine_idx]
                setup_resource = self.resources["setup"][machine_idx][last_jobs_on_machine[machine_idx]][job]

            # Calculate potential makespan and time increase
            current_makespan = machine_loads[machine_idx]
            processing_time = self.processing_times[job, machine_idx]
            process_resource = self.resources["process"][job][machine_idx]
            makespan_increase = current_makespan + setup_time + processing_time
            time_increase = setup_time + processing_time
            resource_increase = setup_resource + process_resource

            # linear_combination = setup_resource*setup_time + process_resource*processing_time

            # Select the best machine based on makespan and tie-break with time increase
            if time_increase <= min_time_increase and makespan_increase <= min_makespan_increase:
            # if (resource_increase <= min_resource_increase and makespan_increase <= min_makespan_increase and time_increase <= min_time_increase):
                best_machine = machine_idx
                min_makespan_increase = makespan_increase
                min_time_increase = time_increase
                assigned_setup_time = setup_time
                assigned_processing_time = processing_time
                min_resource_increase = resource_increase
                # min_linear_combination = linear_combination

        
        setup_start_time = machine_loads[best_machine]
        setup_end_time = setup_start_time + assigned_setup_time

        # Calculate job start and end times
        job_start_time = setup_end_time
        job_end_time = job_start_time + assigned_processing_time

        # Perform final updates
        JS.append(job)
        JM.append(best_machine)
        # Add the job to the sequence of the assigned machine
        machine_job_sequence[best_machine].append(job)
        setup_times[best_machine, job, 0] = setup_start_time
        setup_times[best_machine, job, 1] = setup_end_time
        process_times[best_machine, job, 0] = job_start_time
        process_times[best_machine, job, 1] = job_end_time

        machine_loads[best_machine] = job_end_time
        last_jobs_on_machine[best_machine] = job

        total_resource_requirement += min_resource_increase

        return JS, JM, setup_times, process_times, machine_loads, last_jobs_on_machine, machine_job_sequence, total_resource_requirement

    def local_search(self, machine_job_sequence, machine_loads):

        best_solution = max(machine_loads)  # Initial best makespan
        no_improvement_iterations = 0

        while no_improvement_iterations < 50:
            # Find the machine with the maximum load (makespan machine)
            makespan_machine = np.argmax(machine_loads)

            # Get all jobs assigned to the makespan machine
            jobs_on_machine = machine_job_sequence[makespan_machine]

            # Remove a random job from the makespan machine
            if not jobs_on_machine:
                break  # No jobs to remove

            removed_job_idx = np.random.randint(len(jobs_on_machine))
            
            removed_job = jobs_on_machine.pop(removed_job_idx)

            # Update the machine load after removing the job
            machine_loads = self.update_machine_loads(
                machine_loads, makespan_machine, jobs_on_machine, removed_job, removed_job_idx=removed_job_idx)

            # Test reinsertion into all machines
            improvement_flag = False
            for machine_idx in range(self.n_machines):
                for position in range(len(machine_job_sequence[machine_idx]) + 1):
                    # Use a temporary copy of machine loads for simulation
                    temp_machine_loads = machine_loads.copy()

                    # Temporarily insert the job
                    jobs_on_machine = machine_job_sequence[machine_idx]
                    jobs_on_machine.insert(position, removed_job)

                    # Simulate updating the machine load
                    temp_machine_loads = self.update_machine_loads(
                        temp_machine_loads, machine_idx, jobs_on_machine, removed_job, inserted_job_idx=position)

                    # Compute the new makespan
                    new_makespan = max(temp_machine_loads)

                    # Check for improvement
                    if new_makespan < best_solution:
                        best_solution = new_makespan
                        improvement_flag = True
                        best_position = position
                        best_machine = machine_idx

                    # Remove the job from the temporary position
                    jobs_on_machine.pop(position)

            # If improvement was found, make the changes permanent
            if improvement_flag:
                machine_job_sequence[best_machine].insert(best_position, removed_job)
                machine_loads = self.update_machine_loads(
                    machine_loads, best_machine, machine_job_sequence[best_machine], removed_job, inserted_job_idx=best_position)
                no_improvement_iterations = 0  # Reset counter if improvement is found
            else:
            # Reinsert the job into its original position
                machine_job_sequence[makespan_machine].insert(removed_job_idx, removed_job)
                machine_loads = self.update_machine_loads(
                    machine_loads, makespan_machine, machine_job_sequence[makespan_machine], removed_job, inserted_job_idx=removed_job_idx
                )
                no_improvement_iterations += 1

        return machine_job_sequence, machine_loads
    
    def update_machine_loads(self, machine_loads, machine_idx, job_list, removed_job, removed_job_idx=None, inserted_job_idx=None):
        
        # Handle job removal
        if removed_job_idx is not None:
            # removed_job = job_list[removed_job_idx]
            prev_job = job_list[removed_job_idx - 1] if removed_job_idx > 0 else None 
            next_job = job_list[removed_job_idx] if removed_job_idx < len(job_list) else None
            machine_loads[machine_idx] -= self.processing_times[removed_job, machine_idx]

            # If the removed job is the first job, subtract its initial setup time
            if prev_job is None:
                machine_loads[machine_idx] -= self.setup_times[removed_job, removed_job, machine_idx]
                if next_job is not None:
                    machine_loads[machine_idx] -= self.setup_times[removed_job, next_job, machine_idx]
                    machine_loads[machine_idx] += self.setup_times[next_job, next_job, machine_idx]
            else:
                machine_loads[machine_idx] -= self.setup_times[prev_job, removed_job, machine_idx]
                if next_job is not None:
                    machine_loads[machine_idx] -= self.setup_times[removed_job, next_job, machine_idx]
                    machine_loads[machine_idx] += self.setup_times[prev_job, next_job, machine_idx]

        # Handle job reinsertion
        if inserted_job_idx is not None:
            inserted_job = job_list[inserted_job_idx]
            prev_job = job_list[inserted_job_idx - 1] if inserted_job_idx > 0 else None
            next_job = job_list[inserted_job_idx + 1] if inserted_job_idx < len(job_list)-1 else None
            machine_loads[machine_idx] += self.processing_times[inserted_job, machine_idx]

            # If the inserted job is the first job, add its initial setup time
            if prev_job is None:
                machine_loads[machine_idx] += self.setup_times[inserted_job, inserted_job, machine_idx]
                if next_job is not None:
                    machine_loads[machine_idx] -= self.setup_times[next_job, next_job, machine_idx]
                    machine_loads[machine_idx] += self.setup_times[inserted_job, next_job, machine_idx]
            else:
                machine_loads[machine_idx] += self.setup_times[prev_job, inserted_job, machine_idx]
                if next_job is not None:
                    machine_loads[machine_idx] += self.setup_times[inserted_job, next_job, machine_idx]
                    machine_loads[machine_idx] -= self.setup_times[prev_job, next_job, machine_idx]

        return machine_loads

    def calculate_timings(self, machine_job_sequence, setup_times, process_times):

        # Initialize timing and load tracking
        machine_loads = np.zeros(self.n_machines)  # Machine workloads

        # Iterate over each machine
        for machine_idx, job_sequence in machine_job_sequence.items():
            last_job = -1  # Track the last job on this machine

            for job_idx, job in enumerate(job_sequence):
                # Determine setup time
                if last_job == -1:  # First job on the machine
                    setup_time = self.setup_times[job, job, machine_idx]
                else:  # Sequence-dependent setup time
                    setup_time = self.setup_times[last_job, job, machine_idx]

                # Processing time
                processing_time = self.processing_times[job, machine_idx]

                # Calculate timings
                setup_start_time = machine_loads[machine_idx]
                setup_end_time = setup_start_time + setup_time
                job_start_time = setup_end_time
                job_end_time = job_start_time + processing_time

                # Update setup and process times using numpy
                setup_times[machine_idx, job, 0] = setup_start_time
                setup_times[machine_idx, job, 1] = setup_end_time
                process_times[machine_idx, job, 0] = job_start_time
                process_times[machine_idx, job, 1] = job_end_time

                # Update machine load and last job
                machine_loads[machine_idx] = job_end_time
                last_job = job

        

        return setup_times, process_times, machine_loads

    def repair_phase(self, machine_job_sequence, machine_loads, setup_times, process_times):
        makespan = max(machine_loads)  # Current makespan
        t = 0  # Start at time 0

        while t <= makespan:
            check_point = True
            parallel_jobs = []
            # Step 1: Handle setup resource violations
            while check_point:
                setup_resource_requirements = 0

                # Identify jobs starting their setup at time t
                if not parallel_jobs:
                    for machine_idx, job_sequence in machine_job_sequence.items():
                        for job_idx, job in enumerate(job_sequence):
                            setup_start, setup_end = setup_times[machine_idx, job]

                            if setup_start <= t <= setup_end:  # Ensure t is within the setup interval
                                parallel_jobs.append((machine_idx, job_idx, job))

                                # Calculate setup resource requirements
                                resource_type = "setup"
                                last_job = job_sequence[job_idx - 1] if job_idx > 0 else -1

                                if last_job == -1:  # Machine is idle
                                    setup_resource_requirements += self.resources[resource_type][machine_idx][job][job]
                                else:  # Sequence-dependent setup
                                    setup_resource_requirements += self.resources[resource_type][machine_idx][last_job][job]
                else:
                    for machine_idx, job_idx, job in parallel_jobs:
                        setup_start, setup_end = setup_times[machine_idx, job]

                            # Calculate setup resource requirements
                        resource_type = "setup"
                        job_sequence = machine_job_sequence [machine_idx]
                        last_job = job_sequence[job_idx - 1] if job_idx > 0 else -1

                        if last_job == -1:  # Machine is idle
                            setup_resource_requirements += self.resources[resource_type][machine_idx][job][job]
                        else:  # Sequence-dependent setup
                            setup_resource_requirements += self.resources[resource_type][machine_idx][last_job][job]

                # Check if setup resources are within availability
                if setup_resource_requirements <= self.resource_availability["setup"]:
                    check_point = False

                else:
                    # Simulate adjustments for each job in parallel_jobs
                    min_makespan = float('inf')
                    best_job_entry = None

                    for job_entry in parallel_jobs:
                        machine_idx, job_idx, job = job_entry

                        # Make temporary copies of times and machine loads
                        temp_setup_times = copy.deepcopy(setup_times)
                        temp_process_times = copy.deepcopy(process_times)
                        temp_machine_loads = copy.deepcopy(machine_loads)

                        temp_t = t
                        resource_violation = True
                        max_setup_start = float('inf')      
                        parallel_jobs = sorted(parallel_jobs, key=lambda job_entry: setup_times[job_entry[0], job_entry[2]][1])

                        # Use parallel_jobs2 instead of machine_job_sequence
                        for machine_idx2, job_idx2, job2 in parallel_jobs:
                             # Fetch setup times for this job
                            setup_start, setup_end = setup_times[machine_idx2, job2]

                            if job2 !=job and resource_violation:
                                max_setup_start = setup_end

                                current_setup_start = setup_times[machine_idx, machine_job_sequence[machine_idx][job_idx]][0]
                                temp_t = current_setup_start

                                while current_setup_start <= max_setup_start:
                                    # Increment time for this job and its successors
                                    for idx in range(job_idx, len(machine_job_sequence[machine_idx])):
                                        temp_setup_start, temp_setup_end = temp_setup_times[machine_idx, machine_job_sequence[machine_idx][idx]]
                                        temp_setup_times[machine_idx, machine_job_sequence[machine_idx][idx]] += 1

                                        temp_process_start, temp_process_end = temp_process_times[machine_idx, machine_job_sequence[machine_idx][idx]]
                                        temp_process_times[machine_idx, machine_job_sequence[machine_idx][idx]] += 1
                                    # Update machine load by incrementing by one unit of time
                                    current_setup_start += 1
                                    temp_machine_loads[machine_idx] += 1
                                    temp_t += 1

                                    # Recalculate resource requirements at temp_t
                                    temp_setup_resource_requirements = 0

                                    # Use parallel_jobs2 instead of machine_job_sequence
                                for machine_idx2, job_idx2, job2 in parallel_jobs:
                                    # Fetch setup times for this job
                                    setup_start, setup_end = temp_setup_times[machine_idx2, job2]

                                    # if setup_end <= max_setup_start:
                                    #     max_setup_start = setup_end

                                    if setup_start <= temp_t <= setup_end:  # Ensure t is within the setup interval
                                        # Calculate setup resource requirements
                                        resource_type = "setup"
                                        last_job2 = machine_job_sequence[machine_idx2][job_idx2 - 1] if job_idx2 > 0 else -1

                                        if last_job2 == -1:  # Machine is idle
                                            temp_setup_resource_requirements += self.resources[resource_type][machine_idx2][job2][job2]
                                        else:  # Sequence-dependent setup
                                            temp_setup_resource_requirements += self.resources[resource_type][machine_idx2][last_job2][job2]
                                # Check if violation is resolved
                                resource_violation = temp_setup_resource_requirements > self.resource_availability["setup"]

                                # Calculate the new makespan after resolving violation
                                temp_makespan = max(temp_machine_loads)

                                # Update the best job if this adjustment results in a lower makespan
                                if temp_makespan < min_makespan and resource_violation == False:
                                    min_makespan = temp_makespan
                                    best_job_entry = job_entry
                                    best_setup_times = copy.deepcopy(temp_setup_times)
                                    best_process_times = copy.deepcopy(temp_process_times)
                                    best_machine_loads = copy.deepcopy(temp_machine_loads)

                    # Apply the adjustment for the job with the lowest makespan increase
                    if best_job_entry:
                        final_machine_idx, final_job_idx, final_job = best_job_entry
                        setup_times = copy.deepcopy(best_setup_times)
                        process_times = copy.deepcopy(best_process_times)
                        machine_loads = copy.deepcopy(best_machine_loads)
                        parallel_jobs.remove((final_machine_idx, final_job_idx, final_job))
                        check_point = True

            check_point = True

            # Step 2: Handle processing resource violations
            parallel_jobs = []
            while check_point:              
                process_resource_requirements = 0

                # Identify jobs being processed at time t
                if not parallel_jobs:
                    for machine_idx, job_sequence in machine_job_sequence.items():
                        for job_idx, job in enumerate(job_sequence):
                            process_start, process_end = process_times[machine_idx, job]

                            if process_start <= t <= process_end:  # Ensure t is within the process interval
                                parallel_jobs.append((machine_idx, job_idx, job))

                                # Calculate process resource requirements
                                resource_type = "process"
                                process_resource_requirements += self.resources[resource_type][job][machine_idx]
                else:
                    for machine_idx, job_idx, job in parallel_jobs:
                        process_start, process_end = process_times[machine_idx, job]

                        if process_start <= t <= process_end:  # Ensure t is within the process interval
                            # Calculate process resource requirements
                            resource_type = "process"
                            process_resource_requirements += self.resources[resource_type][job][machine_idx]

                # Check if process resources are within availability
                if process_resource_requirements <= self.resource_availability["process"]:
                    check_point = False
                else:
                    # Simulate adjustments for each job in parallel_jobs
                    min_makespan = float('inf')

                    # Use parallel_jobs2 instead of machine_job_sequence
                    for job_entry in parallel_jobs:
                        machine_idx, job_idx, job = job_entry



                        # Make temporary copies of times and machine loads
                        temp_setup_times = copy.deepcopy(setup_times)
                        temp_process_times = copy.deepcopy(process_times)
                        temp_machine_loads = copy.deepcopy(machine_loads)

                        temp_t = t
                        resource_violation = True
                        max_process_start = float('inf')

                        parallel_jobs = sorted(parallel_jobs, key=lambda job_entry: process_times[job_entry[0], job_entry[2]][1])
                        
                        for machine_idx2, job_idx2, job2 in parallel_jobs:
                            # Fetch setup times for this job
                            process_start, process_end = process_times[machine_idx2, job2]

                            if job2 != job and resource_violation:
                                max_process_start = process_end
                                
                                current_process_start = process_times[machine_idx, machine_job_sequence[machine_idx][job_idx]][0]
                                temp_t = current_process_start

                                while current_process_start <= max_process_start:
                                    # Increment time for this job and its successors
                                    for idx in range(job_idx, len(machine_job_sequence[machine_idx])):
                                        temp_process_start, temp_process_end = temp_process_times[machine_idx, machine_job_sequence[machine_idx][idx]]
                                        temp_process_times[machine_idx, machine_job_sequence[machine_idx][idx]] = np.array([temp_process_start + 1, temp_process_end + 1])

                                        if idx > job_idx:
                                            temp_setup_start, temp_setup_end = temp_setup_times[machine_idx, machine_job_sequence[machine_idx][idx]]
                                            temp_setup_times[machine_idx, machine_job_sequence[machine_idx][idx]] = np.array([temp_setup_start + 1, temp_setup_end + 1])
                                    # Update machine load by incrementing by one unit of time
                                    
                                    current_process_start += 1
                                    temp_machine_loads[machine_idx] += 1
                                    temp_t += 1
                                
                                    # Recalculate resource requirements at temp_t
                                    temp_process_resource_requirements = 0
                                    # Use parallel_jobs2 instead of machine_job_sequence
                                    for machine_idx2, job_idx2, job2 in parallel_jobs:
                                        # Fetch setup times for this job
                                        process_start, process_end = temp_process_times[machine_idx2, job2]
                                        
                                        if process_start <= temp_t <= process_end:  # Ensure t is within the setup interval
                                            # Calculate setup resource requirements
                                            resource_type = "process"
                                            temp_process_resource_requirements += self.resources[resource_type][job2][machine_idx2]
                                    
                                    # Check if violation is resolved
                                    resource_violation = temp_process_resource_requirements > self.resource_availability["process"]
                                
                                # Calculate the new makespan after resolving violation
                                temp_makespan = max(temp_machine_loads)
                                    
                                # Update the best job if this adjustment results in a lower makespan
                                if temp_makespan < min_makespan and resource_violation == False:
                                    min_makespan = temp_makespan
                                    best_job_entry = job_entry
                                    best_setup_times = copy.deepcopy(temp_setup_times)
                                    best_process_times = copy.deepcopy(temp_process_times)
                                    best_machine_loads = copy.deepcopy(temp_machine_loads)

                    # Apply the adjustment for the job with the lowest makespan increase
                    if best_job_entry:
                        final_machine_idx, final_job_idx, final_job = best_job_entry
                        setup_times = copy.deepcopy(best_setup_times)
                        process_times = copy.deepcopy(best_process_times)
                        machine_loads = copy.deepcopy(best_machine_loads)
                        parallel_jobs.remove((final_machine_idx, final_job_idx, final_job))
                        check_point = True
                     
                      
            t += 1
            makespan = max(machine_loads)
        
        return machine_job_sequence, machine_loads, setup_times, process_times
    
    def calculate_metrics(self, setup_times, process_times, machine_loads, machine_job_sequence):
        # Calculate total processing time and total setup time
        total_processing_time = 0
        total_setup_time = 0

        for machine_idx, job_sequence in machine_job_sequence.items():
            for job_idx, job in enumerate(job_sequence):
                setup_start, setup_end = setup_times[machine_idx, job]
                process_start, process_end = process_times[machine_idx, job]

                total_setup_time += setup_end - setup_start
                total_processing_time += process_end - process_start

        total_production_time = total_processing_time + total_setup_time

        # Machine completion times are the machine loads
        machine_completion_times = machine_loads
        makespan = np.max(machine_loads)


        return {
            'total_production_time': total_production_time,
            'total_processing_time': total_processing_time,
            'total_setup_time': total_setup_time,
            'machine_completion_times': machine_completion_times,
            'makespan': makespan,
        }

    def display_individual(self, individual):
        # print("Machine Job Sequence:", individual['machine_job_sequence'], "\n")
        # print("Setup Times:")
        # for machine_idx, job_sequence in enumerate(individual['machine_job_sequence'].values()):
        #     print(f"  Machine {machine_idx}:")
        #     for job_idx, job in enumerate(job_sequence):
        #         setup_start, setup_end = individual['setup_times'][machine_idx, job]
        #         print(f"    Job {job}: Start = {setup_start}, End = {setup_end}")
        # print("\n")

        # print("Process Times:")
        # for machine_idx, job_sequence in enumerate(individual['machine_job_sequence'].values()):
        #     print(f"  Machine {machine_idx}:")
        #     for job_idx, job in enumerate(job_sequence):
        #         process_start, process_end = individual['process_times'][machine_idx, job]
        #         print(f"    Job {job}: Start = {process_start}, End = {process_end}")
        # print("\n")

        # # Initialize dictionaries to store times
        # job_setup_start = {}
        # job_setup_end = {}
        # job_process_start = {}
        # job_process_end = {}

        # # Extract setup and process times
        # for machine_idx, job_sequence in enumerate(individual['machine_job_sequence'].values()):
        #     for job_idx, job in enumerate(job_sequence):
        #         setup_start, setup_end = individual['setup_times'][machine_idx, job]
        #         process_start, process_end = individual['process_times'][machine_idx, job]

        #         # Store values in job-specific lists
        #         if job not in job_setup_start:
        #             job_setup_start[job] = []
        #             job_setup_end[job] = []
        #             job_process_start[job] = []
        #             job_process_end[job] = []

        #         job_setup_start[job].append(setup_start)
        #         job_setup_end[job].append(setup_end)
        #         job_process_start[job].append(process_start)
        #         job_process_end[job].append(process_end)

        # # Sort by job number
        # sorted_jobs = sorted(job_setup_start.keys())

        # Convert to lists
        # setup_start = [job_setup_start[job][0] for job in sorted_jobs]
        # setup_end = [job_setup_end[job][0] for job in sorted_jobs]
        # process_start = [job_process_start[job][0] for job in sorted_jobs]
        # process_end = [job_process_end[job][0] for job in sorted_jobs]

        # # Convert to numpy arrays
        # setup_start = np.array([job_setup_start[job][0] for job in sorted_jobs])
        # setup_end = np.array([job_setup_end[job][0] for job in sorted_jobs])
        # process_start = np.array([job_process_start[job][0] for job in sorted_jobs])
        # process_end = np.array([job_process_end[job][0] for job in sorted_jobs])

        # # Print results
        # print("Setup Start Times:", setup_start)
        # print("Setup End Times:", setup_end)
        # print("Process Start Times:", process_start)
        # print("Process End Times:", process_end)

        print("JS:", individual['JS'])
        print("JM:", individual['JM'])

        print("Total Production Time:", individual['total_production_time'])
        print("Total Processing Time:", individual['total_processing_time'])
        print("Total Setup Time:", individual['total_setup_time'], "\n")

        # print("Machine Completion Times:", individual['machine_completion_times'])
        print("Machine Loads:", individual['machine_loads'], "\n")
        print("Makespan:", individual['makespan'], "\n")
    
    def calculate_initial_sequence(self, individual):
        # Initialize dictionaries to store times
        job_setup_start = {}
        job_setup_end = {}
        job_process_start = {}
        job_process_end = {}

        # Extract setup and process times
        for machine_idx, job_sequence in enumerate(individual['machine_job_sequence'].values()):
            for job_idx, job in enumerate(job_sequence):
                setup_start, setup_end = individual['setup_times'][machine_idx, job]
                process_start, process_end = individual['process_times'][machine_idx, job]

                # Store values in job-specific lists
                if job not in job_setup_start:
                    job_setup_start[job] = []
                    job_setup_end[job] = []
                    job_process_start[job] = []
                    job_process_end[job] = []

                job_setup_start[job].append(setup_start)
                job_setup_end[job].append(setup_end)
                job_process_start[job].append(process_start)
                job_process_end[job].append(process_end)

        # Sort by job number
        sorted_jobs = sorted(job_setup_start.keys())

        # Convert to lists
        setup_start = [job_setup_start[job][0] for job in sorted_jobs]
        setup_end = [job_setup_end[job][0] for job in sorted_jobs]
        process_start = [job_process_start[job][0] for job in sorted_jobs]
        process_end = [job_process_end[job][0] for job in sorted_jobs]

        return {'setup_start':setup_start, 'setup_end':setup_end, 'process_start':process_start, 'process_end':process_end}

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
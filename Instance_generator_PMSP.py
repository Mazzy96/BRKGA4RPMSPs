# Imports
import os
import itertools
import numpy as np
import pandas as pd
import math
# Also requires xlsxwriter to be installed

np.random.seed(0)

# Generator Input Parameters
job_numbers = [30,60,90,120,180] #360
machine_numbers = [2,4,8,10]
scheduling_weeks = [1, 3, 6] # up to
personnel_capacity = [1, 3, 6, 9] # up to
TW_density = [0,0.5,1] # Percentage of jobs with time windows
# worker_unavailability_numbers = [0,1,3,6] # Generate worker unavailability times
worker_unavailability_numbers = [0] # Generate worker unavailability times
machine_eligibility_constraint = [0.6,0.8,1] # percent of jobs that can only be processed on a given machine
weekly_personnel_availability = 2250  # Weekly personnel availability value

# Functions
def generateJobProcessingTimes(min_processing_time, max_processing_time, mean_value, n_machines, n_jobs, machine_eligibility):
    # job_proc_times = np.random.uniform(low=min_processing_time, high=max_processing_time, size=(n_machines, n_jobs)) # Continuous Uniform distribution
    job_proc_times = np.random.randint(low=min_processing_time, high=max_processing_time, size=(n_machines, n_jobs)) # Discrete Uniform distribution

    job_initial_setup_times = generateInitialSetupTimes(n_machines, n_jobs, mean_value)

    machine_eligibilities = generateMachineEligibilities(n_machines, n_jobs, machine_eligibility)

    # set processing times to zero on non-eligible machines
    for i, machine_eligibility in enumerate(machine_eligibilities):
        for job in range(n_jobs):
            if job+1 not in machine_eligibility:
                job_proc_times[i, job] = 0
                job_initial_setup_times[i, job] = 0
    
    return job_proc_times, job_initial_setup_times, machine_eligibilities

def generateMachineEligibilities(n_machines, n_jobs, eligibility_constraint):
    # Calculate the number of machines that can process each job based on the eligibility constraint
    n_eligible_machines_per_job = max(1, int(n_machines * eligibility_constraint))
    # n_eligible_job_per_machine = max(1,n_jobs * eligibility_constraint)

    # Create a list for each machine to store the jobs that can be processed on it
    machine_eligibilities = [[] for _ in range(n_machines)]

    # Create a list of jobs
    jobs = list(range(1, n_jobs + 1))
    np.random.shuffle(jobs)

    # Assign jobs to a subset of machines
    for job in jobs:
        # Randomly select machines that can process this job
        eligible_machines = np.random.choice(n_machines, n_eligible_machines_per_job, replace=False)
        for machine in eligible_machines:
            machine_eligibilities[machine].append(job)
    return machine_eligibilities

def generateInitialSetupTimes(n_machines, n_jobs, mean):
    # create an array of initial setup times for each job
    initial_setup_times = np.full((n_machines, n_jobs), int(mean/2))

    return initial_setup_times

def generateSequenceDependantSetupTimes(n_machines, n_jobs, lower_bound, upper_bound):
    setup_time_lb, setup_time_ub = int(lower_bound/2), int(upper_bound/2)
    # Sequence Dependant Setup Times

    # Based on coordinate method in paper to ensure triangle inequality is met
    # Equation from paper:
    # l + ((u-l)/100)[|x1a -x2a|+|y1a-y2a|]      (l+((u-l)/100)[|x1b-x2b|+|y1b-y2b|])

    plane_coordinates = np.full([n_machines, n_jobs, 2, 2], 0)
    seq_dep_setup_times = np.full([n_machines, n_jobs, n_jobs], 0)
    for i in range(n_machines):
        np.fill_diagonal(seq_dep_setup_times[i], 0, wrap=False)
        for j in range(n_jobs):
            # plane_coordinates[i,j,:] = np.random.uniform(low=0.0, high=50.0, size=[2,2]) # 2x2 matrix of coordinates based on paper
            plane_coordinates[i,j,:] = np.random.randint(low=0.0, high=50.0, size=[2,2]) # 2x2 matrix of coordinates based on paper

        for j in range(n_jobs):
            for k in range(n_jobs):
                if j!=k:
                    if j>k:
                        x1a = plane_coordinates[i, j, 0, 0]
                        x2a = plane_coordinates[i, k, 0, 0]
                        y1a = plane_coordinates[i, j, 0, 1]
                        y2a = plane_coordinates[i, k, 0, 1]
                    elif k>j:
                        x1a = plane_coordinates[i, j, 1, 0] # x1b
                        x2a = plane_coordinates[i, k, 1, 0] # x2b
                        y1a = plane_coordinates[i, j, 1, 1] # y1b
                        y2a = plane_coordinates[i, k, 1, 1] # y2b

                    seq_dep_setup_times[i, j, k] = setup_time_lb + ((setup_time_ub-setup_time_lb)/100)*(np.abs(x1a -x2a)+np.abs(y1a-y2a))
                    
    return seq_dep_setup_times

def generatePersonnelTimes(n_staff, n_weeks, week_minutes):
    people_times = np.full((n_staff, n_weeks), week_minutes)

    # Generate Personnel Assignments
    personnel_assignments = [[] for i in range(n_staff)]
    for i in range(n_staff):
        personnel_assignments[i].append(2*i+1)
        personnel_assignments[i].append(2*i+2)

    return people_times, personnel_assignments

def generateTimeWindows(n_jobs, tw_density, n_weeks, week_minutes):
    day_minutes = week_minutes/5

    # Generate random indexes for windows
    n_tw = int(tw_density * n_jobs) # number of jobs with time windows
    tw_indexes = np.random.choice(n_jobs, size=n_tw, replace=False) # indexes of jobs with time windows

    # Generate random time window lengths based on number of days rather than weeks
    tw_lengths = np.random.randint(low=0, high=5*n_weeks, size=n_tw) # lengths of time windows (days)

    # Generate start time based on days rather than weeks
    tw_starts = np.random.randint(low=0, high=5*n_weeks-tw_lengths, size=n_tw) # start times of time windows (days)
    tw_ends = tw_starts + tw_lengths

    # Create empty arrays for time windows
    release_times = np.full((n_jobs, n_weeks), 0)
    delivery_times = np.full((n_jobs, n_weeks), 0)

    release_periods = np.full(n_jobs, 1) # initialize all to week 1
    delivery_periods = np.full(n_jobs, n_weeks) # initialize all to last week

    # Convert tw_starts to release times
    for i, val in enumerate(tw_indexes):
        release_day = tw_starts[i] % 5 # day of week
        release_week = tw_starts[i] // 5
        release_times[val, release_week] = release_day * day_minutes # start of day, no need to account for index
        release_periods[val] = release_week+1 # plus one to account for index

    # Convert tw_lengths to delivery times
    for i, val in enumerate(tw_indexes):
        delivery_day = (tw_ends[i]) % 5 # day of week
        delivery_week = (tw_ends[i]) // 5
        for j in range(delivery_week):
            delivery_times[val, j] = day_minutes*5
        delivery_times[val, delivery_week] = (delivery_day+1) * day_minutes # plus one for end of day
        delivery_periods[val] = delivery_week+1 # plus one to account for index
    
    # Set all other jobs to have a delivery time of 5 days
    for i in range(n_jobs):
        if i not in tw_indexes:
            for j in range(n_weeks):
                delivery_times[i, j] = day_minutes*5

    return release_times, delivery_times, release_periods, delivery_periods

def generateWorkerUnavailability(n_workers, n_weeks, worker_unavailability_numbers, week_minutes):
    day_minutes = week_minutes / 5

    # Generate random indexes for unavailability
    unavailable_indexes = np.random.choice(n_workers, size=worker_unavailability_numbers, replace=False)  # indices of unavailable workers

    # Generate random time window lengths based on number of days rather than weeks
    unavailable_lengths = np.random.randint(low=0, high=2*n_weeks, size=worker_unavailability_numbers) # lengths of time windows (days) / length of unavailability is maximum 5

    # Generate start time based on days rather than weeks
    unavailable_starts = np.random.randint(low=0, high=5*n_weeks-unavailable_lengths, size=worker_unavailability_numbers) # start times of time windows (days)
    unavailable_ends = unavailable_starts + unavailable_lengths

    # Create empty arrays for unavailability periods and times
    start_off_periods = np.full(n_workers, 0) # initialize all to 0
    end_off_periods = np.full(n_workers, 0) # initialize all to 0

    start_off_times = np.full((n_workers, n_weeks), 0)
    end_off_times = np.full((n_workers, n_weeks), 0)

    # Convert unavailable_starts to start-off times
    for i, val in enumerate(unavailable_indexes):
        start_off_day = unavailable_starts[i] % 5 # day of week
        start_off_week = unavailable_starts[i] // 5
        start_off_times[val, start_off_week] = start_off_day * day_minutes # start of day, no need to account for index
        start_off_periods[val] = start_off_week+1 # plus one to account for index

    # Convert unavailable_lengths to end-off times
    for i, val in enumerate(unavailable_indexes):
        end_off_day = (unavailable_ends[i]) % 5 # day of week
        end_off_week = (unavailable_ends[i]) // 5
        for j in range(end_off_week):
            end_off_times[val, j] = 0
        end_off_times[val, end_off_week] = (end_off_day+1) * day_minutes # plus one for end of day
        end_off_periods[val] = end_off_week+1 # plus one to account for index

    # Set all other workers to have end-off times at the end of each week
    for i in range(n_workers):
        if i not in unavailable_indexes:
            for j in range(n_weeks):
                end_off_times[i, j] = 0  # end of each day
    
    # Calculate the sum of unavailable lengths
    total_unavailable_length = np.sum(unavailable_lengths * day_minutes)

    return start_off_times, end_off_times, start_off_periods, end_off_periods, total_unavailable_length

def writeExcel(index, n_machines, n_weeks, job_proc_times, seq_dep_setup_times, release_times, delivery_times, release_periods, delivery_periods, people_times, machine_eligibilities, job_initial_setup_times, tw_density, worker_unavailability_numbers, weekly_personnel_availability, mean_value, upper_bound, lower_bound, machine_eligibility, personnel_assignments, worker_start_off_times, worker_end_off_times, worker_start_off_periods, worker_end_off_periods, total_unavailable_length):
    # Job processing times
    df_job_proc_times = pd.DataFrame(job_proc_times.T)
    df_job_proc_times.columns = [f'Machine {i}' for i in range(n_machines)]
    df_job_proc_times.index.name = "Job"

    # Sequence Dependant Setup Times
    dfs_seq_dep_setup_times = {}
    for i in range(n_machines):
        dfs_seq_dep_setup_times[i] = pd.DataFrame(seq_dep_setup_times[i,:,:])

    # Time Windows
    df_periods = pd.DataFrame()
    df_periods['Release Periods'] = release_periods
    df_periods['Delivery Periods'] = delivery_periods
    df_periods.index.name = 'Job'

    df_release_times = pd.DataFrame(release_times)
    df_release_times.columns = [f'Week {i+1}' for i in range(n_weeks)]
    df_release_times.index.name = 'Job'

    df_delivery_times = pd.DataFrame(delivery_times)
    df_delivery_times.columns = [f'Week {i+1}' for i in range(n_weeks)]
    df_delivery_times.index.name = 'Job'


    # Worker Unavailability
    df_worker_unavailability_periods = pd.DataFrame()
    df_worker_unavailability_periods['Start Off Periods'] = worker_start_off_periods
    df_worker_unavailability_periods['End Off Periods'] = worker_end_off_periods
    df_worker_unavailability_periods.index.name = 'Worker'

    df_worker_start_off_times = pd.DataFrame(worker_start_off_times)
    df_worker_start_off_times.columns = [f'Week {i+1}' for i in range(n_weeks)]
    df_worker_start_off_times.index.name = 'Worker'

    df_worker_end_off_times = pd.DataFrame(worker_end_off_times)
    df_worker_end_off_times.columns = [f'Week {i+1}' for i in range(n_weeks)]
    df_worker_end_off_times.index.name = 'Worker'

    # Personnel Times
    df_people_times = pd.DataFrame(people_times)
    df_people_times.columns = [f'Week {i+1}' for i in range(n_weeks)]
    df_people_times.index.name = 'Personnel'

    # Initial Setup Times
    df_initial_setup_times = pd.DataFrame(job_initial_setup_times.T)
    df_initial_setup_times.columns = [f'Machine {i}' for i in range(n_machines)]
    df_initial_setup_times.index.name = "Job"


    # Check for data dir
    if not os.path.exists('data'):
        os.makedirs('data')

    # Write to excel
    writer = pd.ExcelWriter(f'./data/Scheduling_Instance_{index}.xlsx', engine='xlsxwriter')
    # writer = pd.ExcelWriter(f'./data/J{n_jobs}_M{n_machines}_P{n_staff}_W{n_weeks}_TW{tw_density}_ME{machine_eligibility}_AP{worker_unavailability_numbers}.xlsx', engine='xlsxwriter')

    # workbook = writer.book

    # Write instance info sheet
    worksheet = writer.book.add_worksheet('Instance Info')
    worksheet.write(0, 0, 'Instance Info')
    worksheet.write(1, 0, 'Instance Number')
    worksheet.write(1, 1, index)
    worksheet.write(2, 0, 'Number of Jobs')
    worksheet.write(2, 1, n_jobs)
    worksheet.write(3, 0, 'Number of Machines')
    worksheet.write(3, 1, n_machines)
    worksheet.write(4, 0, 'Number of Weeks')
    worksheet.write(4, 1, n_weeks)
    worksheet.write(5, 0, 'Personnel Available')
    worksheet.write(5, 1, n_staff)
    worksheet.write(6, 0, 'Max Time Window Length (Days)')
    worksheet.write(6, 1, n_weeks*5)
    worksheet.write(7, 0, 'Time Window Density')
    worksheet.write(7, 1, tw_density)
    worksheet.write(8, 0, 'Weekly Personnel Availability')
    worksheet.write(8, 1, weekly_personnel_availability)
    worksheet.write(9, 0, 'Mean Time Value')
    worksheet.write(9, 1, mean_value)
    worksheet.write(10, 0, 'Upper Bound')
    worksheet.write(10, 1, upper_bound)
    worksheet.write(11, 0, 'Lower Bound')
    worksheet.write(11, 1, lower_bound)
    worksheet.write(12, 0, 'Machine Eligibility Constraint')
    worksheet.write(12, 1, machine_eligibility)
    worksheet.write(13, 0, 'Number of Unavailable Workers:')
    worksheet.write(13, 1, worker_unavailability_numbers)
    worksheet.write(14, 0, 'Total Unavailable Length:')
    worksheet.write(14, 1, total_unavailable_length)

    # Write Sets Sheet
    worksheet = writer.book.add_worksheet('Sets')
    worksheet.write(0, 0, 'NumberOfOrders')
    worksheet.write(1, 0, n_jobs+1)

    worksheet.write(3, 0, 'EndOrder')
    worksheet.write(4, 0, n_jobs+1)

    worksheet.write(6, 0, 'NumberOfProductionLines')
    worksheet.write(7, 0, n_machines)

    worksheet.write(9, 0, 'NumberOfCrewsWithReassignment')
    worksheet.write(10, 0, n_staff*2)

    worksheet.write(12, 0, 'NumberOfCrewsWithoutReassignment')
    worksheet.write(13, 0, n_staff)

    worksheet.write(15, 0, 'NumberOfWeeks')
    worksheet.write(16, 0, n_weeks)

    # Machine Eligibilities
    worksheet.write(0, 2, 'Machine Eligibilities')
    for i in range(n_machines):
        worksheet.write(i+1, 2, f'Machine {i+1}')
        # worksheet.write(i+1, 3, machine_eligibilities[i])
    # worksheet.write(1, 3, f'data') # TODO: fix this to write the actual data
    formatted_machine_eligibilities = [f'{{{", ".join(map(str, machine_eligibilities[i]))}}}' for i in range(len(machine_eligibilities))]
    formatted_machine_eligibilities = str(dict(zip([n+1 for n in range(len(formatted_machine_eligibilities))],formatted_machine_eligibilities)))
    formatted_machine_eligibilities = formatted_machine_eligibilities.replace("'",'')
    formatted_machine_eligibilities = f"data {formatted_machine_eligibilities};"
    worksheet.write(1, 3, formatted_machine_eligibilities)


    # Personnel Assignments
    worksheet.write(0, 5, 'Position Of Line')
    for i in range(n_machines):
        worksheet.write(i+1, 5, f'Machine {i+1}')
        # worksheet.write(i+1, 6, str(personnel_assignments[i]))
    # worksheet.write(n_staff + 1, 5, 'data') # TODO: fix this to write the actual data
    formatted_Position_Line = [f'{{{", ".join(map(str, personnel_assignments[i]))}}}' for i in range(len(personnel_assignments))]
    formatted_Position_Line = str(dict(zip([n+1 for n in range(len(formatted_Position_Line))],formatted_Position_Line)))
    formatted_Position_Line = formatted_Position_Line.replace("'",'')
    formatted_Position_Line = f"data {formatted_Position_Line};"
    worksheet.write(1, 6, formatted_Position_Line)

    # Write OrderInfo Sheet
    worksheet = writer.book.add_worksheet('OrderInfo')
    # Order Processing Times
    worksheet.write(0, 0, 'OrdersProcessingTimes')
    worksheet.write(1, 0, 'Orders')
    worksheet.write(1, 1, 'Machines')
    for i in range(n_machines):
        worksheet.write(2, i+1, i+1)
    for i in range(n_jobs):
        worksheet.write(i+3, 0, i+1)
        for j in range(n_machines):
            worksheet.write(i+3, j+1, job_proc_times[j, i])

    # Order Initial Setup Times
    worksheet.write(0, 10, 'Time_Begin')
    worksheet.write(1, 10, 'Orders')
    worksheet.write(1, 11, 'Machines')
    for i in range(n_machines):
        worksheet.write(2, i+11, i+1)
    for i in range(n_jobs):
        worksheet.write(i+3, 10, i+1)
        for j in range(n_machines):
            worksheet.write(i+3, j+11, job_initial_setup_times[j, i])

    # Release Weeks
    worksheet.write(0, 20, 'ReleaseWeeks')
    worksheet.write(1, 20, 'Orders')
    worksheet.write(1, 21, 'ReleaseWeek')
    for i in range(n_jobs):
        worksheet.write(i+2, 20, i+1)
        worksheet.write(i+2, 21, release_periods[i])

    # Delivery Weeks
    worksheet.write(0, 23, 'DeliveryWeeks')
    worksheet.write(1, 23, 'Orders')
    worksheet.write(1, 24, 'Deadline')
    for i in range(n_jobs):
        worksheet.write(i+2, 23, i+1)
        worksheet.write(i+2, 24, delivery_periods[i])

    # Release Times
    worksheet.write(0, 26, 'ReleaseTimes')
    worksheet.write(1, 26, 'Orders')
    worksheet.write(1, 27, 'Weeks')
    for i in range(n_weeks):
        worksheet.write(2, i+27, i+1)
    for i in range(n_jobs):
        worksheet.write(i+3, 26, i+1)
        for j in range(n_weeks):
            worksheet.write(i+3, j+27, release_times[i, j])

    # Delivery Times
    worksheet.write(0, 34, 'DeliveryTimes')
    worksheet.write(1, 34, 'Orders')
    worksheet.write(1, 35, 'Weeks')
    for i in range(n_weeks):
        worksheet.write(2, i+35, i+1)
    for i in range(n_jobs):
        worksheet.write(i+3, 34, i+1)
        for j in range(n_weeks):
            worksheet.write(i+3, j+35, delivery_times[i, j])
            
    # Write CleanTimes Sheet
    worksheet = writer.book.add_worksheet('CleanTimes')
    worksheet.write(0, 0, 'Machine')
    worksheet.write(0, 1, 'Order')
    for i in range(n_jobs):
        worksheet.write(0, i+2, i+1)
    for i in range(n_machines):
        for j in range(n_jobs):
            worksheet.write(i*n_jobs+j+1, 0, i+1)
            worksheet.write(i*n_jobs+j+1, 1, j+1)
            for j2 in range(n_jobs):
                # if j != j2:
                worksheet.write(i*n_jobs+j+1, j2+2, seq_dep_setup_times[i, j, j2])

    writer.close()

def writeTxt(index, n_machines, n_weeks, job_proc_times, seq_dep_setup_times, release_times, delivery_times, release_periods, delivery_periods, people_times, machine_eligibilities, job_initial_setup_times, tw_density, worker_unavailability_numbers, weekly_personnel_availability, mean_value, upper_bound, lower_bound, machine_eligibility, personnel_assignments, worker_start_off_times, worker_end_off_times, worker_start_off_periods, worker_end_off_periods, total_unavailable_length):

    # Create a directory for data if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Prepare file path
    file_path = f'./data/J{n_jobs}_M{n_machines}_P{n_staff}_W{n_weeks}_TW{tw_density}_ME{machine_eligibility}_AP{worker_unavailability_numbers}.txt'

    with open(file_path, 'w') as file:
        # Write instance info
        file.write('Instance Info\n')
        file.write(f'Instance Number: {index}\n')
        file.write(f'Number of Jobs: {n_jobs}\n')
        file.write(f'Number of Machines: {n_machines}\n')
        file.write(f'Number of Weeks: {n_weeks}\n')
        file.write(f'Personnel Available: {n_staff}\n')
        file.write(f'Max Time Window Length (Days): {n_weeks*5}\n')
        file.write(f'Time Window Density: {tw_density}\n')
        file.write(f'Weekly Personnel Availability: {weekly_personnel_availability}\n')
        file.write(f'Mean Time Value: {mean_value}\n')
        file.write(f'Upper Bound: {upper_bound}\n')
        file.write(f'Lower Bound: {lower_bound}\n')
        file.write(f'Machine Eligibility Constraint: {machine_eligibility}\n')
        file.write(f'Number of Unavailable Workers: {worker_unavailability_numbers}\n')
        file.write(f'Total Unavailable Length: {total_unavailable_length}\n')

        # Write job processing times
        file.write('Job Processing Times:\n')
        for i in range(n_jobs):
            formatted_times = ' '.join(f'{time:.2f}' for time in job_proc_times[:, i])
            file.write(f'{i+1} {formatted_times}\n')

        file.write('Initial Setup Times:\n')
        for i in range(n_jobs):
            formatted_times = ' '.join(f'{time:.2f}' for time in job_initial_setup_times[:, i])
            file.write(f'{i+1} {formatted_times}\n')
            
        # Write machine eligibilities
        file.write('Machine Eligibilities:\n')
        for i in range(n_machines):
            formatted_eligibilities = ' '.join(f'{int(eligibility)}' for eligibility in machine_eligibilities[i])
            file.write(f'{i+1} {formatted_eligibilities}\n')

        # Write personnel times
        file.write('Personnel Times:\n')
        for i in range(n_staff):
            formatted_times = ' '.join(f'{time:.2f}' for time in people_times[i])
            file.write(f'{i+1} {formatted_times}\n')

        # Write personnel assignments
        file.write('Personnel Assignments:\n')
        for i in range(n_staff):
            formatted_assignments = ' '.join(f'{assignment:.2f}' for assignment in personnel_assignments[i])
            file.write(f'{i+1} {formatted_assignments}\n')

        # Write worker unavailability periods and times
        file.write('Worker Start Off Periods:\n')
        for i in range(n_staff):
            file.write(f'{i+1} {int(worker_start_off_periods[i])}\n')

        file.write('Worker End Off Periods:\n')
        for i in range(n_staff):
            file.write(f'{i+1} {int(worker_end_off_periods[i])}\n')

        file.write('Worker Start Off Times:\n')
        for i in range(n_staff):
            formatted_times = ' '.join(f'{time:.2f}' for time in worker_start_off_times[i])
            file.write(f'{i+1} {formatted_times}\n')

        file.write('Worker End Off Times:\n')
        for i in range(n_staff):
            formatted_times = ' '.join(f'{time:.2f}' for time in worker_end_off_times[i])
            file.write(f'{i+1} {formatted_times}\n')

        # Write time windows
        file.write('Release Period:\n')
        for i in range(n_jobs):
            file.write(f'{i+1} {int(release_periods[i])}\n')

        file.write('Delivery Period:\n')
        for i in range(n_jobs):
            file.write(f'{i+1} {int(delivery_periods[i])}\n')

        file.write('Release Times:\n')
        for i in range(n_jobs):
        # Format each float to 2 decimal places
            formatted_times = ' '.join(f'{time:.2f}' for time in release_times[i])
            file.write(f'{i+1} {formatted_times}\n')
        
        file.write('Delivery Times:\n')
        for i in range(n_jobs):
            formatted_times_1 = ' '.join(f'{time:.2f}' for time in delivery_times[i])
            file.write(f'{i+1} {formatted_times_1}\n')

        file.write('Sequence Dependent Setup Times:\n')
        for i in range(n_machines):
            file.write(f'Machine {i+1}:\n')
            for j in range(n_jobs):
                formatted_times = ' '.join(f'{time:.2f}' for time in seq_dep_setup_times[i, j, :])
                file.write(f'{j+1} {formatted_times}\n')

# Main
# Check Valid Combinations

# Initialize a list to store valid combinations
valid_combinations = []

# Collect instance details
instance_records = []

counter = 0

# Iterate through all combinations
for instance_number, (num_machines,num_jobs, num_weeks, personnel_available, tw_percent, worker_unavailability_number, machine_eligibility
) in enumerate(itertools.product(machine_numbers,job_numbers, scheduling_weeks, personnel_capacity, TW_density, worker_unavailability_numbers, machine_eligibility_constraint), start=1):
    
    # Rule 1: The ratio of (number of jobs) / (number of machines * number of weeks) is not less than 15.
    if num_jobs >= 15 * num_machines * num_weeks:
        if num_jobs <= 60 * num_machines * num_weeks:
            # Rule 2: The number of personnel available is less than the number of machines.
            if personnel_available < num_machines:
                # Rule 3: The ratio of machines to personnel is less than or equal to 3.
                if num_machines / personnel_available <= 3:
                    # Rule 4: Worker unavailability density should not exceed the number of available workers
                    if worker_unavailability_number <= personnel_available:
                        # Create a dictionary to store the parameters, instance number, and relevant values
                        instance_data = {
                            'Instance Number': instance_number,
                            'Number of Jobs': num_jobs,
                            'Number of Machines': num_machines,
                            'Number of Weeks': num_weeks,
                            'Personnel Available': personnel_available,
                            'Time Window Density': tw_percent,
                            'Worker Unavailability Number': worker_unavailability_number,
                            'Weekly Personnel Availability': weekly_personnel_availability,
                            'Machine Eligibility Constraint' : machine_eligibility
                        }
                        valid_combinations.append(instance_data)
                        

                        instance_details = f"J{num_jobs}_M{num_machines}_P{personnel_available}_W{num_weeks}_TW{tw_percent}_ME{machine_eligibility}_AP{worker_unavailability_number}.txt"
                        excel_file_name = f"Scheduling_Instance_{counter}.xlsx"
                        counter += 1
                        # Collect the instance details
                        instance_records.append({"Instance Details": instance_details, "Excel File Name": excel_file_name})

# Convert the list of dictionaries to a DataFrame
instance_records_df = pd.DataFrame(instance_records)

# Write the DataFrame to the Excel file
excel_file_path = "./data/instance_records.xlsx"
instance_records_df.to_excel(excel_file_path, index=False)

print('Instance record file is generated!')

# Create a DataFrame from the list of valid combinations
df = pd.DataFrame(valid_combinations)

for index, instance in df.iterrows():

    n_jobs = int(instance['Number of Jobs'])
    n_machines = int(instance['Number of Machines'])
    week_minutes = weekly_personnel_availability
    n_staff = int(instance['Personnel Available'])
    n_weeks = int(instance['Number of Weeks'])
    tw_density = instance['Time Window Density']
    worker_unavailability_numbers = int(instance['Worker Unavailability Number'])
    machine_eligibility = instance['Machine Eligibility Constraint']

    # Generate worker unavailability periods
    worker_start_off_times, worker_end_off_times, worker_start_off_periods, worker_end_off_periods, total_unavailable_length = generateWorkerUnavailability(n_staff, n_weeks, worker_unavailability_numbers, week_minutes)

    # Calculate the mean, upper, and lower bounds for this instance to calculate the setup and processing times
    jobs_per_machine =  math.ceil(n_jobs / n_machines)
    total_available_time = (n_staff * n_weeks * week_minutes -total_unavailable_length) * 0.8 #0.2 of total time is remained unutilized to avoid infeasibility
    mean_value = int (total_available_time / (n_machines*(jobs_per_machine * 1.5))) #1.5= 1 for processing times and 0.5 for setup times / initial setup time is also included
    upper_bound = int((4/3)*mean_value)
    lower_bound = int((2/3)*mean_value)

    # Generate personnel times
    people_times, personnel_assignments = generatePersonnelTimes(n_staff, n_weeks, week_minutes)

    # Generate job processing times
    job_proc_times, job_initial_setup_times, machine_eligibilities = generateJobProcessingTimes(lower_bound, upper_bound, mean_value, n_machines, n_jobs, machine_eligibility)

    # Generate sequence dependant setup times
    seq_dep_setup_times = generateSequenceDependantSetupTimes(n_machines, n_jobs, lower_bound, upper_bound)

    # Generate time windows
    release_times, delivery_times, release_periods, delivery_periods = generateTimeWindows(n_jobs, tw_density, n_weeks, week_minutes)

    # Write to excel
    writeExcel(index, n_machines, n_weeks, job_proc_times, seq_dep_setup_times, release_times, delivery_times, release_periods, delivery_periods, people_times, machine_eligibilities, job_initial_setup_times, tw_density, worker_unavailability_numbers, weekly_personnel_availability, mean_value, upper_bound, lower_bound, machine_eligibility, personnel_assignments, worker_start_off_times, worker_end_off_times, worker_start_off_periods, worker_end_off_periods, total_unavailable_length)
    
    writeTxt(index, n_machines, n_weeks, job_proc_times, seq_dep_setup_times, release_times, delivery_times, release_periods, delivery_periods, people_times, machine_eligibilities, job_initial_setup_times, tw_density, worker_unavailability_numbers, weekly_personnel_availability, mean_value, upper_bound, lower_bound, machine_eligibility, personnel_assignments, worker_start_off_times, worker_end_off_times, worker_start_off_periods, worker_end_off_periods, total_unavailable_length)

    print(f'Instance {index+1} created')
print('Instance generation is complete!')


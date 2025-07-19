import subprocess
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor

def run_command(seed, instance_name, max_cpu):
    command = f"python grka.py PMSP Instances\\PMSP\\{instance_name} --solver brkga --max_cpu {max_cpu} --threads 1 --seed {seed}"
    # command = f"python grka.py UPMSR-PS Instances\\UPMSR-PS\\{instance_name} --solver brkga --max_cpu {max_cpu} --threads 1 --seed {seed}"
    # command = f"python grka.py IPMR-P Instances\\IPMR-P\\{instance_name} --solver brkga --max_cpu {max_cpu} --threads 1 --seed {seed}"

    print(f"Running command: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout
    
    print(f"Command output for seed {seed}:\n{output}")

def process_instance(row):
    instance_name = row["Instance Details"]
    
    # Extract the number of jobs from the instance name
    match = re.search(r'J(\d+)_', instance_name)
    # match = re.search(r'-(\d+)-', instance_name)
    # match = re.search(r'^(\d+)x', instance_name)
    
    if match:
        num_jobs = int(match.group(1))
        max_cpu = int(num_jobs * 2)  # Adjust logic as needed
    else:
        max_cpu = 240  # Fallback value
    
    print(f"Processing instance: {instance_name}")
    
    # Use ThreadPoolExecutor to run commands in parallel
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(run_command, seed, instance_name, max_cpu)
            for seed in range(1, 7)
        ]
        for future in futures:
            future.result()  # Wait for all to complete

# Load your DataFrame (assuming df is already defined)
excel_file_path = "./instance_records_PMSP.xlsx"

df = pd.read_excel(excel_file_path)  # Replace with your actual DataFrame loading method

# Drop rows where "Instance Details" is NaN
df = df.dropna(subset=["Instance Details"])

# Iterate over each instance and process it
for _, row in df.iterrows():
    process_instance(row)

print("All commands executed and results saved to Excel.")



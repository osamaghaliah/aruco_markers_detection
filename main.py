import os
import subprocess

def run_command(command):
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"Error running command: {command}")
        print(process.stderr)
    return process.stdout, process.stderr

def extract_average_processing_time(log_file_path):
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
        if lines:
            last_line = lines[-1]
            if "Frame Processing Time Average" in last_line:
                average_value = last_line.split('=')[-1].strip().split()[0]
                return average_value
    except Exception as e:
        print(f"Error reading log file {log_file_path}: {e}")
    return "Average processing time not found"

def main():
    # Define the input directories and their respective commands
    input_dirs = [f'input{i}' for i in range(1, 7)]
    commands = [
        f"python ArUcoDetector.py --video inputs/{d}/{d}.mp4 --output_dir outputs/output{i} --drone_log inputs/{d}/{d}.csv"
        if d not in ["input5", "input6"] else
        f"python ArUcoDetector.py --video inputs/{d}/{d}.mp4 --output_dir outputs/output{i}"
        for i, d in enumerate(input_dirs, start=1)
    ]

    # Path to the summary CSV file
    summary_file_path = os.path.join('outputs', 'performance.csv')

    # Ensure the outputs directory exists
    os.makedirs('outputs', exist_ok=True)

    # Run each command and collect the results
    with open(summary_file_path, 'w') as summary_file:
        summary_file.write("INPUT#,Frame Processing Time Average\n")

        for i, command in enumerate(commands):
            print(f"Running command for {input_dirs[i]}...")
            run_command(command)

            output_dir = f'outputs/output{i + 1}'  # Adjusted to correctly reflect the output directory
            log_file_path = os.path.join(output_dir, 'processing_times.log')
            print(f"Checking log file at: {log_file_path}")
            if os.path.exists(log_file_path):
                avg_time = extract_average_processing_time(log_file_path)
                summary_file.write(f"{input_dirs[i]},{avg_time}\n")
                print(f"{input_dirs[i]}: Frame Processing Time Average = {avg_time} ms")
            else:
                summary_file.write(f"{input_dirs[i]},log file not found\n")
                print(f"{input_dirs[i]}: log file not found")

    print(f"\nSummary of processing times has been saved to {summary_file_path}")

if __name__ == '__main__':
    main()

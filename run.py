import subprocess

def run_script(script_name):
    try:
        subprocess.run(['python3', script_name], check=True)
        print(f"{script_name} completed successfully.")
    except subprocess.CalledProcessError:
        print(f"Error: {script_name} failed.")
        exit(1)

if __name__ == "__main__":
    # Run kmeans.py
    print("Running kmeans.py...")
    run_script("kmeans.py")

    # Run evaluate_model.py
    print("Running evaluate_model.py...")
    run_script("evaluate_model.py")

    print("Script completed successfully.")

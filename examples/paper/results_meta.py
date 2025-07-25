#!/usr/bin/env python3
"""
Simple meta-runner that keeps calling run_results.py until it completes.
The script monitors the progress file - when it's gone, the work is done.

Usage:
    python run_until_complete.py
"""

import subprocess
import os
import time
import sys
from datetime import datetime

# Configuration
RESULTS_SCRIPT = "results_run.py"
PROGRESS_FILE = "data/prestress_progress.json"
MAX_ATTEMPTS = 100
WAIT_TIME = 5
LOG_FILE = "run_until_complete.log"

def log_message(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(LOG_FILE, 'a') as f:
        f.write(full_message + '\n')

def run_script():
    """Run the results script and return exit code"""
    log_message(f"Starting {RESULTS_SCRIPT}")
    
    process = subprocess.Popen(
        [sys.executable, RESULTS_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Print output in real-time
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
    
    process.wait()
    return process.returncode

def main():
    log_message("="*60)
    log_message("Meta-runner started")
    log_message(f"Will run {RESULTS_SCRIPT} until completion")
    log_message("="*60)
    
    attempt = 0
    
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        
        # Check if work is done
        if not os.path.exists(PROGRESS_FILE) and attempt > 1:
            log_message("Progress file gone - work completed!")
            log_message(f"Total attempts: {attempt - 1}")
            break
        
        log_message(f"\nAttempt {attempt}/{MAX_ATTEMPTS}")
        
        try:
            exit_code = run_script()
            log_message(f"Script exited with code {exit_code}")
            
            # Give it a moment
            time.sleep(2)
            
            # Check if done
            if not os.path.exists(PROGRESS_FILE):
                log_message("SUCCESS: All work completed!")
                break
                
        except KeyboardInterrupt:
            log_message("Interrupted by user")
            sys.exit(1)
            
        except Exception as e:
            log_message(f"Error: {e}")
        
        # Wait before retry
        if os.path.exists(PROGRESS_FILE):
            log_message(f"Waiting {WAIT_TIME}s before retry...")
            time.sleep(WAIT_TIME)
    
    # Final status
    if os.path.exists(PROGRESS_FILE):
        log_message(f"FAILED: Max attempts ({MAX_ATTEMPTS}) reached")
        sys.exit(1)
    else:
        log_message("All done!")
        sys.exit(0)

if __name__ == "__main__":
    main()
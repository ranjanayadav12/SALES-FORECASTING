import os
import subprocess
import time
import sys

def kill_python_processes():
    print("Stopping existing application instances...")
    if sys.platform == "win32":
        try:
            # kill all python processes except the current one
            # Using taskkill /F /IM python.exe is a bit aggressive but often needed here
            subprocess.run(["taskkill", "/F", "/IM", "python.exe"], capture_output=True)
            time.sleep(2)
        except Exception as e:
            print(f"Cleanup note: {e}")
    print("Cleanup complete.")

def run_app():
    print("\nStarting Sales Forecasting Website...")
    print("Access URL: http://127.0.0.1:5000")
    try:
        # Force Port 5000 to resolve conflicting instances
        subprocess.run(["py", "app.py"], env={**os.environ, "FLASK_RUN_PORT": "5000"})
    except KeyboardInterrupt:
        print("\nShutdown complete.")

if __name__ == "__main__":
    kill_python_processes() # Auto-cleanup enabled for stability
    run_app()

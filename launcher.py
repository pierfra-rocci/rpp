import os
import subprocess
import sys
import time
import webbrowser

def main():
    # Get the directory where the executable is located
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the actual app executable
    app_path = os.path.join(app_dir, "rapas_app.exe")
    
    # Start the Streamlit process
    process = subprocess.Popen([app_path, "--server.address=127.0.0.1", "--server.headless=false"])
    
    # Wait a bit for server to start
    time.sleep(5)
    
    # Open browser
    webbrowser.open("http://localhost:8501")
    
    # Wait for the process to finish
    process.wait()

if __name__ == "__main__":
    main()
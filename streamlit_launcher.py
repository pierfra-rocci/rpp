import subprocess
import time

def main():
    # Get path to Streamlit executable and the main app script
    streamlit_path = r"C:\Users\pierf\astronomy\.astronomy\Scripts\streamlit.exe"
    app_script = r"C:\Users\pierf\photometria\pfr_app.py"
    
    # Command to run the app
    cmd = [streamlit_path, "run", app_script, "--server.port=8502"]
    
    print("Starting Streamlit server...")
    
    # Use Popen to start streamlit in the background
    process = subprocess.Popen(cmd)
    
    # Wait for server to start
    time.sleep(2)
    
    # Open browser
    print("Opening browser...")
    
    try:
        # Keep the script running until user presses Ctrl+C
        print("App is running. Press Ctrl+C to stop.")
        process.wait()
    except KeyboardInterrupt:
        print("Stopping app...")
        process.terminate()
    
if __name__ == "__main__":
    main()
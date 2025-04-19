import os
import subprocess

file_path = "C:\\Users\\pierf\\Desktop\\tmp0qbaomrp.fit"

if os.path.exists(file_path):
    # Use subprocess to run the command with powershell
    command = [
        "powershell.exe",
        "-ExecutionPolicy", "Bypass",
        "-File", "run_siril_script.ps1", 
        "-filepath",
        f"{file_path}"
    ]
    subprocess.run(command, check=True)
else:
    print(f"File {file_path} does not exist.")
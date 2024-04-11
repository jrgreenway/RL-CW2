import subprocess
import sys
import os


        
def create_venv(venv_name):
    subprocess.check_call([sys.executable, "-m", "venv", venv_name])

def install(package, venv_name):
    subprocess.check_call([os.path.join(venv_name, 'Scripts', 'python'), "-m", "pip", "install", package])

packages = ["gymnasium[atari]", "gymnasium[accept-rom-license]"]
venv_name = ".venv"

if __name__ == "__main__":
    if os.path.exists(venv_name):
        raise Exception(f"Error: {venv_name} already exists. Please delete before running this script")
    create_venv(venv_name)
    for package in packages:
        install(package, venv_name)
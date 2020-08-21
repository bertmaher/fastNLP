import subprocess
import sys

def pip_install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

def setup_install():
    subprocess.check_call([sys.executable, 'setup.py', 'install'])

if __name__ == '__main__':
    pip_install_requirements()
    setup_install()

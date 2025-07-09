import os
import subprocess


def main():
    path = os.path.join(os.path.dirname(__file__), "gui.py")
    subprocess.run(["streamlit", "run", path])

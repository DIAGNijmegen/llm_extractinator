import os
import subprocess


def main():
    path = os.path.join(os.path.dirname(__file__), "schema_builder.py")
    subprocess.run(["streamlit", "run", path])

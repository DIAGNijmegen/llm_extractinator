import os
import subprocess


def main():
    path = os.path.join(os.path.dirname(__file__), "schema_builder.py")
    env = os.environ.copy()
    # Match the GUI launcher so the standalone builder saves parsers to the same
    # <project>/tasks/parsers the Studio reads from.
    env.setdefault("EXTRACTINATOR_BASE_DIR", os.getcwd())
    subprocess.run(["streamlit", "run", path], env=env)

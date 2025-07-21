import argparse
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the LLM Extractinator GUI.")
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the Streamlit app on (default: 8501)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = os.path.join(os.path.dirname(__file__), "gui.py")
    subprocess.run(["streamlit", "run", path, "--server.port", str(args.port)])


if __name__ == "__main__":
    main()

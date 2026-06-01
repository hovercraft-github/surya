import subprocess
import os
import argparse


def streamlit_app_cli(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501, help="Port to run streamlit on")
    parsed = parser.parse_args(args)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ocr_app_path = os.path.join(cur_dir, "streamlit_app.py")
    cmd = ["streamlit", "run", ocr_app_path, "--server.fileWatcherType", "none", "--server.headless", "true", "--server.port", str(parsed.port)]
    env = {**os.environ, "IN_STREAMLIT": "true"}
    subprocess.run(cmd, env=env)
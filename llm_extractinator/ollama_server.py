# import os
# import subprocess
# import time
# from pathlib import Path

# import ollama


# class OllamaServerManager:
#     def __init__(self, model_name, log_dir, log_filename="ollama_server.log"):
#         """
#         Initialize the server manager with the given model name.
#         """
#         self.model_name = model_name
#         self.log_dir = log_dir
#         self.log_file_path = self.log_dir / log_filename
#         self.serve_process = None

#         # Ensure the output directory exists
#         os.makedirs(self.log_dir, exist_ok=True)

#     def pull_model(self):
#         """
#         Pull the specified model using the `ollama pull` command.
#         """
#         try:
#             print(f"Pulling model: {self.model_name}...")
#             ollama.pull(self.model_name)
#             print(f"Model {self.model_name} pulled successfully.")
#             time.sleep(5)
#         except Exception as e:
#             print(f"Error pulling model {self.model_name}: {e}")

#     def start_server(self):
#         """
#         Start the server for the specified model using the `ollama serve` command.
#         """
#         log_file_handle = open(self.log_file_path, "w")

#         try:
#             serve_command = f"ollama serve"
#             print(f"Starting server...")
#             self.serve_process = subprocess.Popen(
#                 serve_command,
#                 shell=True,
#                 stdout=log_file_handle,
#                 stderr=subprocess.STDOUT,
#             )
#             print("Ollama server is running...")
#             time.sleep(5)
#         except Exception as e:
#             print(f"Error starting Ollama server: {e}")
#             log_file_handle.close()

#     def stop_server(self):
#         """
#         Stop the server if it is running.
#         """
#         if self.serve_process:
#             print("Terminating Ollama server...")
#             self.serve_process.terminate()
#             self.serve_process.wait()  # Ensure the process has been terminated
#             print("Ollama server terminated.")
#             self.serve_process = None

#     def __enter__(self):
#         """
#         Context manager entry point.
#         """
#         # Pull the model and start the server
#         self.start_server()
#         self.pull_model()
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         """
#         Context manager exit point.
#         Stops the server if the script exits or crashes.
#         """
#         # Stop the server if the script exits or crashes
#         self.stop_server()

import subprocess
import time


class OllamaServerManager:
    def __init__(self, host="localhost", port=28900):
        """
        Initializes the OllamaServerManager.
        :param host: Host address to bind the server.
        :param port: Port to run the server.
        """
        self.host = host
        self.port = port
        self.process = None

    def start(self):
        """
        Starts the Ollama server process.
        """
        if self.process is not None:
            raise RuntimeError("Ollama server is already running.")

        command = ["ollama", "serve"]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait briefly to ensure the server has time to start
        time.sleep(2)
        print("Ollama server started.")

    def stop(self, model_name):
        """
        Stops the Ollama server process manually.
        """
        command = ["ollama", "stop", model_name]

        try:
            subprocess.run(command, check=True, text=True)
            print(f"Model '{model_name}' stopped successfully.")
        except subprocess.CalledProcessError:
            print(f"Failed to stop model '{model_name}'.")

    def pull_model(self, model_name):
        """
        Pulls a specified model for offline use.
        :param model_name: Name of the model to pull.
        """
        command = ["ollama", "pull", model_name]
        try:
            subprocess.run(command, check=True, text=True)
            print(f"Model '{model_name}' pulled successfully.")
        except subprocess.CalledProcessError:
            print(f"Failed to pull model '{model_name}'.")

    def __enter__(self):
        """Starts the server when entering the context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stops the server automatically when exiting the context."""
        if self.process is None:
            print("No Ollama server is running.")
            return

        self.process.terminate()
        self.process.wait()
        self.process = None
        print("Ollama server stopped.")

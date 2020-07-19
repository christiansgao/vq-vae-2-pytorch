import os

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

def get_abs_path(local_path):
    return os.path.join(get_project_root(), local_path)

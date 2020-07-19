import os

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

def get_abs_path(local_path):
    return os.path.join(get_project_root(), local_path)


class Paths:
    TRAINING = get_abs_path('dataset/testSample')
    #CHECKPOINT = get_abs_path('checkpoint/vqvae_011.pt')
    CHECKPOINT = get_abs_path('vqvae_560.pt')

    EVAL_OUTPUT = get_abs_path('sample/output.png')
from src.util import get_abs_path

class Paths:
    TRAINING = get_abs_path('dataset/testSample')
    #CHECKPOINT = get_abs_path('checkpoint/vqvae_011.pt')
    CHECKPOINT = get_abs_path('vqvae_560.pt')

    EVAL_OUTPUT = get_abs_path('sample/output.png')
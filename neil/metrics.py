from fastai.metrics import accuracy, BrierScore, F1Score, RocAucBinary

def load_metrics():
    return [accuracy, F1Score(), RocAucBinary(), BrierScore()]

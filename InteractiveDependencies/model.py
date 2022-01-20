from modules.models.Linear import LogReg
from modules.models.Wraps import TorchClassifierWrap


def load_model():
    return TorchClassifierWrap(LogReg(), 100, 300, 500, verbose=False)

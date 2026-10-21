import torch


def count_detectors(configs):
    from subshowers.detector import from_config as detector_from_config
    detector = detector_from_config(configs)
    return len(detector)

class Model(torch.nn.Module):
    def __init__(self, configs):
        self.configs = configs
        self.n_detectors = count_detectors(configs)
    
    @classmethod
    def _load_from_checkpoint(cls, checkpoint):
        pass

    def write(self, path):  # include both weights and configs
        pass


class MultiplicityEncoder(torch.nn.Module):
    def __init__(self, configs):
        self.configs = configs


class MultiplicityPredictor(torch.nn.Module):
    def __init__(self, configs):
        self.configs = configs


class DepositEncoder(torch.nn.Module):
    def __init__(self, configs):
        self.configs = configs


class DepositPDGPredictor(torch.nn.Module):
    def __init__(self, configs):
        self.configs = configs


class DepositEnergyPredictor(torch.nn.Module):
    def __init__(self, configs):
        self.configs = configs

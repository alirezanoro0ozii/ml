import torch


class Batch():
    def __init__(self, seq_ids, tax_ids):
        self.seq_ids = seq_ids
        self.taxes = tax_ids
           
    def gpu(self, device):      
        for name, var in self.__dict__.items():
            if isinstance(var, torch.Tensor) or isinstance(var, torch.LongTensor):
                setattr(self, name, var.to(device))
            else:
                setattr(self, name, {k: v.to(device) for k, v in var.items()})

class GPTBatch():
    def __init__(self, seq_ids, tax_ids):
        self.input_ids = seq_ids
        self.output_ids = tax_ids
           
    def gpu(self, device):      
        for name, var in self.__dict__.items():
            if isinstance(var, torch.LongTensor):
                setattr(self, name, var.to(device))
            else:
                setattr(self, name, {k: v.to(device) for k, v in var.items()})
                

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, best_model, patience, verbose=False):
        self.best_model = best_model
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_qwk_max = np.Inf
        self.force_cancel = False

    def __call__(self, val_qwk, model):
        score = val_qwk

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_qwk, model)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_qwk, model)
            self.counter = 0

    def save_checkpoint(self, val_qwk, model):
        if self.verbose:
            print(f"Validation qwk increased ({self.val_qwk_max:.3f} --> {val_qwk:.3f}). Saving model ...")

        torch.save(model.state_dict(), self.best_model)
        self.val_qwk_max = val_qwk
import os
import sys
import unittest
import random
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import copy
from cs236781.train_results import FitResult

def rotate_2d(X, deg=0):
    """
    Rotates each 2d sample in X of shape (N, 2) by deg degrees.
    """
    a = np.deg2rad(deg)
    return X @ np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]]).T

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from hw2.mlp import MLP
from hw2.classifier import BinaryClassifier

from hw2.training import ClassifierTrainer
from hw2.answers import part3_arch_hp, part3_optim_hp
test = unittest.TestCase()

def create_dls():
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plt.rcParams.update({'font.size': 12})

    np.random.seed(seed)

    N = 10_000
    N_train = int(N * .8)

    # Create data from two different distributions for the training/validation
    X1, y1 = make_moons(n_samples=N_train//2, noise=0.2)
    X1 = rotate_2d(X1, deg=10)
    X2, y2 = make_moons(n_samples=N_train//2, noise=0.25)
    X2 = rotate_2d(X2, deg=50)

    # Test data comes from a similar but noisier distribution
    X3, y3 = make_moons(n_samples=(N-N_train), noise=0.3)
    X3 = rotate_2d(X3, deg=40)

    X, y = np.vstack([X1, X2, X3]), np.hstack([y1, y2, y3])

    # Train and validation data is from mixture distribution
    X_train, X_valid, y_train, y_valid = train_test_split(X[:N_train, :], y[:N_train], test_size=1/3, shuffle=False)

    # Test data is only from the second distribution
    X_test, y_test = X[N_train:, :], y[N_train:]

    batch_size = 32
    
    dl_train, dl_valid, dl_test = [
        DataLoader(
            dataset=TensorDataset(
                torch.from_numpy(X_).to(torch.float32),
                torch.from_numpy(y_)
            ),
            shuffle=True,
            num_workers=0,
            batch_size=batch_size
        )
        for X_, y_ in [(X_train, y_train), (X_valid, y_valid), (X_test, y_test)]
    ]
    return dl_train, dl_valid, dl_test

def test(hp_arch, hp_optim, dl_train, dl_valid):
    model = BinaryClassifier(
        model=MLP(
            in_dim=2,
            dims=[*[hp_arch['hidden_dims'],]*hp_arch['n_layers'], 2],
            nonlins=[*[hp_arch['activation'],]*hp_arch['n_layers'], hp_arch['out_activation']]
        ),
        threshold=0.5,
    )
    print ()
    loss_fn = torch.nn.CrossEntropyLoss()
    if "loss_fn" in hp_optim:
        hp_optim.pop("loss_fn")
    print (f"hp optim is {hp_optim}")
    optimizer = torch.optim.SGD(params=model.parameters(), **hp_optim)
    trainer = ClassifierTrainer(model, loss_fn, optimizer)

    fit_result = trainer.fit(dl_train, dl_valid, num_epochs=int(sys.argv[2]), print_every=10)
    return fit_result

def alter_param(v, scale, func):
    dv = random.normalvariate(0, scale)
    return func(v, dv)

def mult(v, dv):
    return v * (2 ** dv)
def addint(v, dv):
    return int(v + 0.5 + dv)
def activation(act, dv):
    if (act == 'sigmoid' and dv < 1) or (act != 'sigmoid' and dv > 1):
        return 'sigmoid'
    return 'relu'

ALTERATION_HP = {
    "loss_fn": (lambda x, _:x, 1),
    "n_layers": (addint, 0.4),
    "hidden_dims": (addint, 30),
    "activation": (activation, 1.3),
    "out_activation": (activation, 1.3),
    "lr": (mult, 1),
    "weight_decay": (mult, 1),
    "momentum": (mult, 0.01)
}

def alter_params(best_hp_arch, best_hp_optim, i_frac):
    altered = ({}, {})
    for i, d in enumerate([best_hp_arch, best_hp_optim]):
        for key, value in d.items():
            func, scale = ALTERATION_HP[key]
            altered[i][key] = alter_param(value, scale * i_frac, func)
    return altered

@contextmanager
def mute():
    old_out = sys.stdout
    with open(os.devnull, 'w') as null_file:
        sys.stdout = null_file
        yield
    sys.stdout = old_out 


def main():
    print(f"started at {datetime.now()}")
    start = datetime.now()
    best_hp_arch = part3_arch_hp()
    best_hp_optim = part3_optim_hp()
    best_result = 0
    dl_train, dl_valid, dl_test = create_dls()
    
    iters = int(sys.argv[1])
    for i in range(iters):
        i_frac = 2.0001 - 2 ** ((i / iters) ** 2)
        hp_arch, hp_optim = alter_params(best_hp_arch, best_hp_optim, i_frac)
        with mute():
            result = test(hp_arch, hp_optim, dl_train, dl_valid).train_acc[-1]
        if result > best_result:
            best_result = result
            best_hp_arch = copy.copy(hp_arch)
            best_hp_optim = copy.copy(hp_optim)
            print()
            print(f"Found better results, accuracy of train at last epoch: {best_result}")
            print(f"hp_arch:  {hp_arch}")
            print(f"hp_optim: {hp_optim}")
    print(f"finished at {datetime.now()}")
    delta = datetime.now() - start
    print(f"time took is {delta} for {int(sys.argv[1]) * int(sys.argv[2])} epochs")

if __name__ == '__main__':
    main()
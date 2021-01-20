import math
import sys
from copy import deepcopy
import pickle
import random

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine
import datetime

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube
import scipy.stats as sps

import traceback


class Attack:
    def __init__(
            self,
            f,
            dim,
            max_evals,
            verbose=True,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=50,
            device=None,
            dtype="float64",

    ):
        self.f = f
        self.dim = dim

        # Settings
        self.max_evals = max_evals
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cpu") if device is None else device

    def init(self, X_pool, n_init, batch_size, iteration=10):
        self.X_pool = X_pool
        self.hypers = {}
        self.n_init = n_init
        self.batch_size = batch_size
        self._optimize(epoch=iteration)

    def create_candidates(self, X, fX, X_pool, n_training_steps, hypers, sample_number):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        assert X.min() >= 0.0 and X.max() <= 1.0

        mu, sigma = fX.median(), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        try:
            with torch.enable_grad(), gpytorch.settings.max_cholesky_size(2000):
                gp = train_gp(train_x=X, train_y=fX, use_ard=True, num_steps=n_training_steps, hypers=hypers)
                hypers = gp.state_dict()
        except Exception as e:
            with torch.enable_grad(), gpytorch.settings.max_cholesky_size(2000):
                gp = train_gp(train_x=X, train_y=fX, use_ard=True, num_steps=20, hypers={})
                hypers = gp.state_dict()

        # Create candidate points
        X_list = [i for i in range(X_pool.shape[0])]
        if (X_pool.shape[0] > 8000):
            X_list = random.sample(X_list, 8000)
        X_list = torch.tensor(X_list, dtype=torch.long, device=self.device)
        X_cand = X_pool[X_list]
        self.X_list = X_list

        gp = gp.to(dtype=self.dtype, device=self.device)

        if sample_number == 0:
            return hypers
        else:
            # We use Lanczos for sampling if we have enough data
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(2000):
                try:
                    y_cand = gp.likelihood(gp(X_cand))
                except Exception as e:
                    with torch.enable_grad(), gpytorch.settings.max_cholesky_size(2000):
                        gp = train_gp(train_x=X, train_y=fX, use_ard=True, num_steps=20, hypers={})
                        hypers = gp.state_dict()
                    gp = gp.to(dtype=self.dtype, device=self.device)
                    y_cand = gp.likelihood(gp(X_cand))
                # EI
                func_m = y_cand.mean.cpu().numpy()
                func_v = y_cand.variance.cpu().numpy()
                ei_values = torch.min(fX).cpu().numpy()
                func_s = np.sqrt(func_v)
                u = (ei_values - func_m) / func_s
                ncdf = sps.norm.cdf(u)
                npdf = sps.norm.pdf(u)
                ei = func_s * (u * ncdf + npdf)
                if ei.ndim == 1:
                    ei = ei[:, np.newaxis]
                ei = torch.from_numpy(ei)
            return X_cand, -ei, hypers

    def get_likelihood(self, X, fX, X_pool, n_training_steps, hypers, sample_number):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        mu, sigma = fX.median(), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on

        # We use CG + Lanczos for training if we have enough data
        with torch.enable_grad(), gpytorch.settings.max_cholesky_size(2000):
            gp = train_gp(train_x=X, train_y=fX, use_ard=True, num_steps=n_training_steps, hypers=hypers)
            hypers = gp.state_dict()

        X_list = [i for i in range(X_pool.shape[0])]
        if (X_pool.shape[0] > 8000):
            X_list = random.sample(X_list, 8000)
        X_list = torch.tensor(X_list, dtype=torch.long, device=self.device)
        X_cand = X_pool[X_list]
        self.X_list = X_list

        # Figure out what device we are running on
        # We may have to move the GP to a new device
        gp = gp.to(dtype=self.dtype, device=self.device)

        if sample_number == 0:
            return hypers
        else:
            # We use Lanczos for sampling if we have enough data
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(2000):
                y_cand = gp.likelihood(gp(X_cand)).mean  # get the mean of the function
            y_cand = mu + sigma * y_cand

            return X_cand, y_cand, hypers

    def select_candidates(self, X_cand, y_cand, delete_pool=True, get_loss=True):
        """Select candidates."""
        indices = y_cand.argmin(dim=0)
        X_next = X_cand[indices]

        # here change the X_cand to X_pool
        X_list = self.X_list
        X_cand = self.X_pool
        if delete_pool:
            reserve_index = torch.ones(len(X_cand), device=self.device, dtype=torch.bool)
            reserve_index[X_list[indices]] = False  # convert to origin index
            X_cand = X_cand[reserve_index]
        if get_loss:
            fX_next = self.f.get_loss(X_next)
            return X_next, fX_next, X_cand
        else:
            return X_next, X_cand

    def _optimize(self, epoch):
        """Run the full optimization process."""
        X_init = latin_hypercube(self.n_init, self.dim)
        self.X = torch.zeros((0, self.dim), device=self.device)
        self.fX = torch.zeros((0), device=self.device)

        for i in X_init:
            i = torch.from_numpy(i).to(self.device).type(self.X_pool.dtype)
            tmp = torch.norm(self.X_pool - i, dim=1)
            index = torch.argmin(tmp)
            self.X = torch.cat((self.X, self.X_pool[index, :].unsqueeze(0).clone()), dim=0)
            self.X_pool = torch.cat((self.X_pool[:index], self.X_pool[index + 1:]), dim=0)

        self.fX = self.f.get_loss(self.X)
        for i in range(epoch - 1):
            X_cand, y_cand, self.hypers = self.create_candidates(self.X, self.fX, self.X_pool, self.n_training_steps,
                                                                 hypers=self.hypers, sample_number=self.batch_size)
            X_next, fX_next, self.X_pool = self.select_candidates(X_cand, y_cand)
            self.X = torch.cat((self.X, X_next), dim=0)
            self.fX = torch.cat((self.fX, fX_next), dim=0)
            self.n_training_steps = 10




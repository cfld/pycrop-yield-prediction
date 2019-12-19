import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
from tqdm import tqdm
from datetime import datetime

from .loss import l1_l2_loss


class ModelBase:
    """
    Base class for all models
    """
    def __init__(self, model, model_weight, model_bias, model_type, savedir, use_gp=True,
                 sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.savedir = savedir / model_type
        self.savedir.mkdir(parents=True, exist_ok=True)

        print(f'Using {device.type}')
        if device.type != 'cpu':
            model = model.cuda()
        self.model = model
        self.model_type = model_type
        self.model_weight = model_weight
        self.model_bias = model_bias

        self.device = device

        # for reproducability
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)


    def run(self, path_to_data, times='all', pred_years=None, num_runs=2, train_steps=25000,
            batch_size=32, starter_learning_rate=1e-3, weight_decay=0, l1_weight=0, patience=10):



        with np.load(path_to_data) as data:
            images      = data['output_image']
            locations   = data['output_locations']
            yields      = data['output_yield']
            years       = data['output_year']
            indices     = data['output_index']


        years_list, run_numbers, rmse_list, me_list, times_list = [], [], [], [], []

        if pred_years is None:
            pred_years = range(2009, 2016)
        elif type(pred_years) is int:
            pred_years = [pred_years]
        if times == 'all':
            times = [32]
        else:
            times = range(10, 31, 4)


        for pred_year in pred_years:
            for run_number in range(1, num_runs+1):
                for time in times:
                    print(f'Training to predict on {pred_year}, Run number {run_number}')

                    results = self._run_1_year(images, yields,
                                               years, locations,
                                               indices, pred_year,
                                               time, run_number,
                                               train_steps, batch_size,
                                               starter_learning_rate,
                                               weight_decay, l1_weight,
                                               patience)




    def _run_1_year(self, images, yields, years, locations, indices, predict_year, time, run_number,
                    train_steps, batch_size, starter_learning_rate, weight_decay, l1_weight, patience):


        train_data, test_data = self.prepare_arrays(images, yields, locations, indices, years, predict_year, time)



    def prepare_arrays(self, images, yields, locations, indices, years, predict_year, time):

        train_idx   = np.nonzero(years < predict_year)[0]
        test_idx    = np.nonzero(years == predict_year)[0]

        train_images, test_images = self._normalize(images[train_idx], images[test_idx])


    @staticmethod
    def _normalize(train_images, val_images):
        """
        Find the mean values of the bands in the train images. Use these values
        to normalize both the training and validation images.

        A little awkward, since transpositions are necessary to make array broadcasting work
        """
        mean = np.mean(train_images, axis=(0, 2, 3))

        train_images = (train_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)
        val_images = (val_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)

        return train_images, val_images





        return train_data, test_data



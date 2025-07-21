import numpy as np
import torch
from torch.utils.data import Dataset
from utils import sample_mask


class CustomTimeSeriesDataset(Dataset):
    def __init__(self, raw_feature_data, raw_weather_data,
                 in_len, out_len, period, period1,
                 scaler,missing_ratio,missing_pattern, start_offset=0):

        self.in_len = in_len
        self.out_len = out_len
        self.period = period
        self.period1 = period1
        self.scaler = scaler
        self.start_offset = start_offset
        self.train_rng = np.random.default_rng(44)
        self.raw_feature_data = raw_feature_data
        self.raw_weather_data = raw_weather_data
        self.gt_mask = (~np.isnan(raw_feature_data)).astype(np.float32)       
        self.indicating_mask = sample_mask(
            shape=raw_feature_data.shape,
            p=0,
            p_noise=missing_ratio,
            min_seq=0,
            max_seq=0,
            rng=self.train_rng,
        )
        self.scaled_feature_data = self.scaler.transform(self.raw_feature_data)


        self.scaled_feature_data_nan_as_zero = np.nan_to_num(self.scaled_feature_data, nan=0.0)
        
        self.final_scaled_feature_data_nan_as_zero = self.scaled_feature_data_nan_as_zero * (1 - self.indicating_mask) 
        self.mask = self.gt_mask * (1 - self.indicating_mask)      
        self.num_samples = self.raw_feature_data.shape[1] - (self.in_len + self.out_len) + 1
        if self.num_samples <= 0:
            raise ValueError(f"Not enough data to create samples. "
                             f"Data time steps: {self.raw_feature_data.shape[1]}, "
                             f"Required: {self.in_len + self.out_len}")
        print(
        "Train: original missing ratio = {:.4f}, artificial missing ratio = {:.4f}, artificial missing pattern: {}, overall missing ratio = {:.4f}".format(
            1 - np.sum(self.gt_mask) / self.gt_mask.size,
            np.sum(self.indicating_mask) / self.indicating_mask.size,
            missing_pattern,
            1 - np.sum(self.mask) / self.mask.size,
        )
        
    )


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        x_start = idx
        x_end = idx + self.in_len
        x_observed_input = self.scaled_feature_data_nan_as_zero[:, x_start:x_end, :]


        y_start = idx + self.in_len
        y_end = idx + self.in_len + self.out_len
        y_scaled = self.scaled_feature_data[:, y_start:y_end, :] 


        y_original = self.raw_feature_data[:, y_start:y_end, :]


        w = self.raw_weather_data[:, x_start:x_end] 



        abs_ti_start = self.start_offset + x_start
        abs_ti_end = self.start_offset + x_end
        ti_indices = np.arange(abs_ti_start, abs_ti_end)
        ti_add = (ti_indices // self.period % self.period1) / self.period1
        ti = (ti_indices % self.period) / (self.period - 1 if self.period > 1 else 1.0) 

        abs_to_start = self.start_offset + y_start
        abs_to_end = self.start_offset + y_end
        to_indices = np.arange(abs_to_start, abs_to_end)
        to_add = (to_indices // self.period % self.period1) / self.period1
        to = (to_indices % self.period) / (self.period - 1 if self.period > 1 else 1.0) + to_add

        x_input_mask = self.mask[:, x_start:x_end, :]

        x_original_mask = self.gt_mask[:, x_start:x_end, :]
        y_original_mask = self.gt_mask[:, y_start:y_end, :]

        x_indicating_mask = self.indicating_mask[:, x_start:x_end, :]

        x_tensor = torch.FloatTensor(x_observed_input).permute(2, 0, 1)
        y_scaled_tensor = torch.FloatTensor(y_scaled).permute(2, 0, 1)
        y_original_tensor = torch.FloatTensor(y_original).permute(2, 0, 1)
        w_tensor = torch.LongTensor(w) 
        ti_tensor = torch.FloatTensor(ti)
        to_tensor = torch.FloatTensor(to)
        x_input_mask_tensor = torch.LongTensor(x_input_mask).permute(2, 0, 1)
        x_original_mask_tensor = torch.LongTensor(x_original_mask).permute(2, 0, 1)
        y_original_mask_tensor = torch.LongTensor(y_original_mask).permute(2, 0, 1)
        x_indicating_mask_tensor= torch.LongTensor(x_indicating_mask).permute(2, 0, 1)
        return x_tensor, y_scaled_tensor, y_original_tensor, w_tensor, ti_tensor, to_tensor ,x_input_mask_tensor,x_original_mask_tensor,x_indicating_mask_tensor,y_original_mask_tensor


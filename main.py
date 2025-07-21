import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import copy
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader,TensorDataset
import math
from collections import OrderedDict
from dataset import CustomTimeSeriesDataset
from utils import load_data, StandardScaler ,get_lap_pos_enc ,test_error
from model import DST_DAMP

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='China',help='data type')
parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], help='hidden layer dimension', type=float)
parser.add_argument('--in_channels',type=int,default=2,help='input variable')
parser.add_argument("--hidden_channels", nargs="+", default=32, help='hidden layer dimension', type=int)
parser.add_argument('--out_channels',type=int,default=2,help='output variable')
parser.add_argument('--d_model',type=int,default=32,help='size')
parser.add_argument('--dropout',type=float,default=0,help='dropout rate')
parser.add_argument('--heads',type=int,default=8,help='number of attention heads')
parser.add_argument('--support_len',type=int,default=3,help='number of spatial adjacency matrix')
parser.add_argument('--decay', type=float, default=1e-5, help='decay rate of learning rate ')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate ')
parser.add_argument('--in_len',type=int,default=36,help='input time series length')      # a relatively long sequence can handle missing data
parser.add_argument('--out_len',type=int,default=12,help='output time series length')
parser.add_argument('--batch',type=int,default=32,help='training batch size')
parser.add_argument('--episode',type=int,default=50,help='training episodes')
parser.add_argument('--period',type=int,default=36,help='periodic for temporal embedding')
parser.add_argument('--period1',type=int,default=7,help='the input sequence is longer than one day, we use this periodicity to allocate a unique index to each time point')

args = parser.parse_args()
def main():
    device = torch.device(args.device)
    adj, training_data, val_data, test_data, training_w, val_w, test_w = load_data(args.data)
    supports = [torch.tensor(i).to(device) for i in adj]
    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(), training_data[~np.isnan(training_data)].std())

    print(f"Training data shape: {training_data.shape}, Training weather shape: {training_w.shape}")
    print(f"Validation data shape: {val_data.shape}, Validation weather shape: {val_w.shape}")
    print(f"Test data shape: {test_data.shape}, Test weather shape: {test_w.shape}")

    train_dataset = CustomTimeSeriesDataset(
        raw_feature_data=training_data,
        raw_weather_data=training_w,
        in_len=args.in_len,
        out_len=args.out_len,
        period=args.period,
        period1=args.period1,
        scaler=scaler,
        missing_ratio=0.4,
        missing_pattern ="point",
        start_offset=0
    )
    val_start_offset = training_data.shape[1]
    val_dataset = CustomTimeSeriesDataset(
        raw_feature_data=val_data,
        raw_weather_data=val_w,
        in_len=args.in_len,
        out_len=args.out_len,
        period=args.period,
        period1=args.period1,
        scaler=scaler,
        missing_ratio=0.4,
        missing_pattern ="point",
        start_offset=val_start_offset
    )
    test_start_offset = training_data.shape[1] + val_data.shape[1]
    test_dataset = CustomTimeSeriesDataset(
        raw_feature_data=test_data,
        raw_weather_data=test_w,
        in_len=args.in_len,
        out_len=args.out_len,
        period=args.period,
        period1=args.period1,
        scaler=scaler,
        missing_ratio=0.4,
        missing_pattern ="point",
        start_offset=test_start_offset
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0 # Adjust based on your system and preference
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0
    )

    adj_mx = np.load('cdata/dist_mx.npy')
    lap_pos_enc=get_lap_pos_enc(adj_mx,args.hidden_channels)
    lap_pos_enc=torch.tensor(lap_pos_enc).to(device)

    config = {
            'device': args.device,
            'data': args.data,
            'in_channels': args.in_channels,
            'hidden_channels': args.hidden_channels,
            'out_channels': args.out_channels,
            'dropout': args.dropout,
            'd_model':args.d_model,
            'heads': args.heads,
            'support_len': args.support_len,
            'lr': args.lr,
            'decay': args.decay,
            'in_len': args.in_len,
            'out_len': args.out_len,
            'batch_size': args.batch,
            'num_epochs': args.episode,
            'period': args.period,
            'node_nums':training_data.shape[0],
            'lap_pos_enc':lap_pos_enc,
            'hidden':args.hidden_channels,
            'num_timesteps': 50,
            'beta_schedule': 'quad',
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'num_pretrain_epochs':10

        }
    diffusion_model = DST_DAMP(config)

    best_val_loss = float('inf')
    for ep in range(1, config['num_pretrain_epochs']+1):
        # Training phase
        train_losses = []
        for i, (x_batch, y_scaled_batch, y_original_batch, w_batch, ti_batch, to_batch, x_input_mask, x_original_mask, x_indicating_mask,y_original_mask) in enumerate(train_loader):
            trainx = x_batch.to(device)
            trainti = ti_batch.to(device)
            trainto = to_batch.to(device)
            trainw = w_batch.to(device)
            train_input_mask = x_input_mask.to(device)
            train_original_mask = x_original_mask.to(device)
            train_indicating_mask = x_indicating_mask.to(device)
            train_y_original_mask=y_original_mask.to(device)
            y_scaled_batch[np.isnan(y_scaled_batch)] = 0
            trainy = y_scaled_batch.to(device)
            train_loss = diffusion_model.pretrain_step((trainx, trainti, trainto, trainw, trainy, supports, train_input_mask, train_original_mask, train_indicating_mask,train_y_original_mask))
            train_losses.append(train_loss)
        avg_train_loss = np.mean(train_losses)

        # Validation phase
        val_losses = []
        for i, (x_batch, y_scaled_batch, y_original_batch, w_batch, ti_batch, to_batch, x_input_mask, x_original_mask, x_indicating_mask,y_original_mask) in enumerate(val_loader):
            valx = x_batch.to(device)
            valti = ti_batch.to(device)
            valto = to_batch.to(device)
            valw = w_batch.to(device)
            val_input_mask = x_input_mask.to(device)
            val_original_mask = x_original_mask.to(device)
            val_indicating_mask = x_indicating_mask.to(device)
            val_y_original_mask = y_original_mask.to(device)
            y_scaled_batch[np.isnan(y_scaled_batch)] = 0
            valy = y_scaled_batch.to(device)
            val_loss = diffusion_model.val_step((valx, valti, valto, valw, valy, supports, val_input_mask, val_original_mask, val_indicating_mask,val_y_original_mask))
            val_losses.append(val_loss)

        avg_val_loss = np.mean(val_losses)

        # Check if current validation loss is the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model here (you'll need to implement the saving logic)
            # Example: torch.save(diffusion_model.state_dict(), 'best_model.pth')
            torch.save(diffusion_model.model.state_dict(), "best.pth")
        # Print training and validation metrics
        print(f"Epoch {ep}/{config['num_pretrain_epochs']}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")

    pretrained_weights = torch.load("best.pth", map_location=device)

    diffusion_model.model.load_state_dict(pretrained_weights)

    MAE_list = []
    best_val_loss = float('inf')
    for ep in range(1, config['num_epochs']):
        # Training phase
        diffusion_model.model.train()
        train_losses = []
        for i, (x_batch, y_scaled_batch, y_original_batch, w_batch, ti_batch, to_batch, x_input_mask, x_original_mask, x_indicating_mask,y_original_mask) in enumerate(train_loader):
            trainx = x_batch.to(device)
            trainti = ti_batch.to(device)
            trainto = to_batch.to(device)
            trainw = w_batch.to(device)

            train_y_original_mask = y_original_mask.to(device)
            y_scaled_batch[np.isnan(y_scaled_batch)] = 0
            trainy = y_scaled_batch.to(device)



            loss = diffusion_model.train_step((trainx, trainti, trainto, trainw, trainy, supports,train_y_original_mask))
            train_losses.append(loss)
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {ep}/{args.episode}: Train Loss: {avg_train_loss:.4f}")
        outputs = []
        truths = []
        diffusion_model.model.eval()
        for i, (x_batch, y_scaled_batch, y_original_batch, w_batch, ti_batch, to_batch, x_input_mask, x_original_mask, x_indicating_mask,y_original_mask) in enumerate(val_loader):
            valx=x_batch.to(device)
            valti=ti_batch.to(device)
            valto=to_batch.to(device)
            valw=w_batch.to(device)


            output=diffusion_model.test_step((valx, valti, valto, valw,supports))
            output = output.permute(0, 2, 3, 1)
            y_original_batch=y_original_batch.permute(0, 2, 3, 1)
            output = output.detach().cpu().numpy()
            y_original_batch=y_original_batch.detach().cpu().numpy()
            output = scaler.inverse_transform(output)
            outputs.append(output)
            truths.append(y_original_batch)

        yhat = np.concatenate(outputs)
        y = np.concatenate(truths)

        amae = []
        ar2 = []
        armse = []

        for i in range(12):
            metrics = test_error(yhat[:,:,i,:],y[:,:,i,:])
            amae.append(metrics[0])
            ar2.append(metrics[2])
            armse.append(metrics[1])
        MAE_list.append(np.mean(amae))
        if np.mean(amae) == min(MAE_list):
            best_model = copy.deepcopy(diffusion_model.model.state_dict())
        log = 'On average over all horizons, Test MAE: {:.4f}, Test R2: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(ar2),np.mean(armse)))
    diffusion_model.model.load_state_dict(best_model)

    outputs = []
    truths=[]

    for i, (x_batch, y_scaled_batch, y_original_batch, w_batch, ti_batch, to_batch, x_input_mask, x_original_mask, x_indicating_mask,y_original_mask) in enumerate(test_loader):
        testx=x_batch.to(device)
        testti=ti_batch.to(device)
        testto=to_batch.to(device)
        testw=w_batch.to(device)
        output=diffusion_model.test_step((testx, testti, testto, testw,supports))



        output = output.permute(0, 2, 3, 1)
        y_original_batch=y_original_batch.permute(0, 2, 3, 1)
        output = output.detach().cpu().numpy()
        y_original_batch=y_original_batch.detach().cpu().numpy()
        output = scaler.inverse_transform(output)
        outputs.append(output)
        truths.append(y_original_batch)
    yhat = np.concatenate(outputs)
    y = np.concatenate(truths)

    log = '3 step ahead arrival delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:,:,2,0],y[:,:,2,0])
    print(log.format(MAE, R2, RMSE))

    log = '6 step ahead arrival delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:,:,5,0],y[:,:,5,0])
    print(log.format(MAE, R2, RMSE))

    log = '12 step ahead arrival delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:,:,11,0],y[:,:,11,0])
    print(log.format(MAE, R2, RMSE))

    log = '3 step ahead departure delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:,:,2,1],y[:,:,2,1])
    print(log.format(MAE, R2, RMSE))

    log = '6 step ahead departure delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:,:,5,1],y[:,:,5,1])
    print(log.format(MAE, R2, RMSE))

    log = '12 step ahead departure delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:,:,11,1],y[:,:,11,1])
    print(log.format(MAE, R2, RMSE))
if __name__ == "__main__":   
    main() 
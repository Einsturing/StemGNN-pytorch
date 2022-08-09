import numpy as np
import pandas as pd

from models.base_model import Model
from data_loader.forecast_dataloader import ForecastDataset
import torch.utils.data as torch_data
import torch
import torch.nn as nn
import time
import os
from datetime import datetime
from utils.math_utils import evaluate
from .scatterFig import display


def print_param(params):
    for name, parameter in params:
        if not parameter.requires_grad:
            continue
        print(name)
        print(parameter)


def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


def inference(model, dataloader, device, node_cnt, train_window, predict_window, blocks, Ks, Kt):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.unsqueeze(3).to(device)
            target = target.unsqueeze(3).to(device)
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], predict_window, node_cnt, 1], dtype=np.float)
            while step < predict_window:
                forecast_result = model(inputs)
                len_model_output = forecast_result.size()[1]
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :train_window - len_model_output, :] = inputs[:, len_model_output:train_window, :].clone()
                inputs[:, train_window - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(predict_window - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(predict_window - step, len_model_output), :].detach().cpu().numpy()
                step += min(predict_window - step, len_model_output)
            forecast_set.append(forecast_steps)
            target_set.append(target.detach().cpu().numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)


def validate(model, dataloader, device, node_cnt, train_window, predict_window, scaler, blocks, Ks, Kt):
    start = datetime.now()
    forecast_norm, target_norm = inference(model, dataloader, device, node_cnt, train_window, predict_window, blocks, Ks, Kt)
    forecast = []
    target = []
    for i in range(len(forecast_norm)):
        forecast.append(scaler.inverse_transform(np.squeeze(forecast_norm[i], 2)))
        target.append(scaler.inverse_transform(np.squeeze(target_norm[i], 2)))
    forecast, target = np.array(forecast), np.array(target)
    score = evaluate(target, forecast)
    end = datetime.now()
    # print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    return dict(mae=score[1], mape=score[0], rmse=score[2])


def train(train_data, test_data, args, result_file, scaler, blocks, Ks, Kt):
    time_step, node_cnt = train_data.shape[0], train_data.shape[1]
    model = Model(node_cnt, 2, args.train_window, args.predict_window, blocks, Ks, Kt)
    model.to(args.device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-4)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
    train_set = ForecastDataset(train_data, train_window=args.train_window, predict_window=args.predict_window)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                         num_workers=0)
    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")
    performance_metrics = {}
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.unsqueeze(3).to(args.device)
            target = target.unsqueeze(3).to(args.device)
            model.zero_grad()
            forecast = model(inputs)
            loss = forecast_loss(forecast, target)
            cnt += 1
            loss.backward()
            my_optim.step()

            # print_param(model.named_parameters())

            loss_total += float(loss)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt))
        # if (epoch+1) % args.exponential_decay_step == 0:
        #     my_lr_scheduler.step()
        save_model(model, result_file)


def test(test_data, args, result_train_file, scaler, blocks, Ks, Kt):
    model = load_model(result_train_file)
    node_cnt = test_data.shape[1]
    test_set = ForecastDataset(test_data, train_window=args.train_window, predict_window=args.predict_window)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                        num_workers=0)
    performance_metrics = validate(model, test_loader, args.device, node_cnt, args.train_window, args.predict_window,
                                   scaler, blocks, Ks, Kt)
    mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
    print('Performance on test set: MAPE: {:7.9%} | MAE: {:7.9f} | RMSE: {:7.9f}'.format(mape, mae, rmse))

import os
import torch
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from models.handler import train, test

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='uk190624_64')
parser.add_argument('--train_window', type=int, default=10)
parser.add_argument('--predict_window', type=int, default=6)
parser.add_argument('--train_length', type=float, default=8)
parser.add_argument('--test_length', type=float, default=2)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--exponential_decay_step', type=int, default=10)

args = parser.parse_args()
print(f'Training configs: {args}')
data_file = os.path.join('dataset', args.dataset + '.csv')
result_file = os.path.join('output', args.dataset)
if not os.path.exists(result_file):
    os.makedirs(result_file)

data = pd.read_csv(data_file).iloc[:, 2:].T.values
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

train_ratio = args.train_length / (args.train_length + args.test_length)
test_ratio = 1 - train_ratio
train_data = data[:int(train_ratio * len(data))]
test_data = data[int(train_ratio * len(data)) - args.train_window:]
blocks = [[1, 32, 64], [64, 32, 128]]
# blocks = [[1, 64, 128], [128, 64, 256]]
# TODO:微调该参数 +2

# torch.cuda.manual_seed(int.seed)
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            train(train_data, test_data, args, result_file, scaler, blocks, args.ks, args.kt)
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        test(test_data, args, result_file, scaler, blocks, args.ks, args.kt)
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')

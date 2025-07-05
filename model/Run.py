import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)
import copy
import torch
import torch.nn as nn
from model.Model import Traffic_model as Network_Predict
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch, MSE_torch, huber_loss
from lib.Params_pretrain import parse_args
from lib.Params_predictor import get_predictor_params
from lib.data_process import define_dataloder

# *************************************************************************#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_nodes_dict = {'PEMS04': 307, 'PEMS08': 170, 'PEMS07': 883, 'CD_DIDI': 524, 'SZ_DIDI': 627,
                  'METR_LA': 207, 'PEMS_BAY': 325, 'PEMS07M': 228, 'NYC_TAXI': 263, 'CHI_TAXI': 77, 'NYC_BIKE-3': 540,
                  'CAD3': 480, 'CAD4-1': 621, 'CAD4-2': 610, 'CAD4-3': 593, 'CAD4-4': 528, 'CAD5': 211,
                  'CAD7-1': 666, 'CAD7-2': 634, 'CAD7-3': 559, 'CAD8-1': 510, 'CAD8-2': 512, 'CAD12-1': 453, 'CAD12-2': 500,
                  'TrafficZZ': 676, 'TrafficCD': 728, 'TrafficHZ': 672, 'TrafficJN': 576, 'TrafficSH': 896,
                  }
client_no = {'PEMS04': 1, 'PEMS08': 2, 'PEMS07': 3, 'CD_DIDI': 4, 'SZ_DIDI': 5,
                  'METR_LA': 6, 'PEMS_BAY': 7, 'PEMS07M': 8, 'NYC_TAXI': 9, 'CHI_TAXI': 10, 'NYC_BIKE-3': 11,
                  'CAD3': 12, 'CAD4-1': 13, 'CAD4-2': 14, 'CAD4-3': 15, 'CAD4-4': 16, 'CAD5': 17,
                  'CAD7-1': 18, 'CAD7-2': 19, 'CAD7-3': 20, 'CAD8-1': 21, 'CAD8-2': 22, 'CAD12-1': 23, 'CAD12-2': 24,
                  'TrafficZZ': 25, 'TrafficCD': 26, 'TrafficHZ': 27, 'TrafficJN': 28, 'TrafficSH': 29,
                  }

args = parse_args(device)
args.num_nodes_dict = num_nodes_dict
args.client_no=client_no
args_predictor = get_predictor_params(args)

attr_list = []
if args.mode !='pretrain':
    for arg in vars(args):
        attr_list.append(arg)
    for attr in attr_list:
        if hasattr(args, attr) and hasattr(args_predictor, attr):
            setattr(args, attr, getattr(args_predictor, attr))
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('==========')
    for arg in vars(args_predictor):
        print(arg, ':', getattr(args_predictor, arg))
init_seed(args.seed, args.seed_mode)

print('mode: ', args.mode, '  model: ', args.model, '  dataset: ', args.dataset_use, '  load_pretrain_path: ', args.load_pretrain_path, '  save_pretrain_path: ', args.save_pretrain_path)

def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
log_dir = os.path.join(parent_dir,'model_weights', args.model)
Mkdir(log_dir)
args.log_dir = log_dir

#load dataset
train_dataloader, val_dataloader, test_dataloader, scaler_dict = define_dataloder(args=args)

#init model
clients = []
global_model = Network_Predict(args, args_predictor)
print("torch.cuda.device_count()",torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    global_model = nn.DataParallel(global_model)
global_model = global_model.to(args.device)
if args.xavier:
    for p in global_model.parameters():
        if p.requires_grad==True:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
clients.append(global_model)

for i in range(len(num_nodes_dict)):
    client = copy.deepcopy(global_model)
    if torch.cuda.device_count() > 1:
        client = nn.DataParallel(client)
    client = client.to(args.device)
    clients.append(client)

#init loss function, optimizer
def scaler_mae_loss(mask_value):
    def loss(preds, labels, scaler, mask=None):
        if scaler and args.real_value == False:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if args.mode == 'pretrain' and mask is not None:
            preds = preds * mask
            labels = labels * mask
        mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

def scaler_huber_loss(mask_value):
    def loss(preds, labels, scaler, mask=None):
        if scaler and args.real_value == False:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if args.mode == 'pretrain' and mask is not None:
            preds = preds * mask
            labels = labels * mask
        mae = huber_loss(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

if args.loss_func == 'mask_mae':
    loss = scaler_mae_loss(mask_value=args.mape_thresh)
    print('============================scaler_mae_loss')
elif args.loss_func == 'mask_huber':
    if args.mode != 'pretrain':
        loss = scaler_huber_loss(mask_value=args.mape_thresh)
        print('============================scaler_huber_loss')
    else:
        loss = scaler_mae_loss(mask_value=args.mape_thresh)
        print('============================scaler_mae_loss')
    # print(args.model, Mode)
elif args.loss_func == 'mae':
    loss = torch.nn.L1Loss()
elif args.loss_func == 'mse':
    loss = torch.nn.MSELoss()
else:
    raise ValueError
optimizer = torch.optim.Adam(params=global_model.parameters(), lr=args.lr_init, eps=1.0e-8, weight_decay=0, amsgrad=False)

#learning rate decay
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
else:
    scheduler = None


#start training
trainer = Trainer(clients, loss, optimizer, train_dataloader, val_dataloader, test_dataloader, scaler_dict, args, scheduler=scheduler)
if args.mode == 'pretrain' or args.mode == 'ori':
    print_model_parameters(global_model, only_num=False)
    trainer.multi_train()
elif args.mode == 'eval':
    path = log_dir + '/' + args.load_pretrain_path
    if torch.cuda.device_count() > 1:
        global_model.load_state_dict(torch.load(path))
    else:
        model_weights = {k.replace('module.', ''): v for k, v in torch.load(path).items()}
        global_model.load_state_dict(model_weights)
    print("Load saved model")
    for param in global_model.parameters():
        param.requires_grad = False
    for param in global_model.predictor.linear.parameters():
        param.requires_grad = True
    print_model_parameters(global_model, only_num=False)
    trainer.multi_train()
elif args.mode == 'test':
    print_model_parameters(global_model, only_num=False)
    print("Load saved model")
    trainer.test(global_model, trainer.args, scaler_dict, test_dataloader, trainer.logger, path=log_dir + '/' + args.load_pretrain_path)
else:
    raise ValueError

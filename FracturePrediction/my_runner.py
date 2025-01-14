
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       readaption of predrnn runner
@Date     :2023/10/13 22:03:51
@Author      :chenbaolin
@version      :0.1
'''
import os
import shutil
import argparse
import numpy as np
import math
#from core.data_provider import datasets_factory
from DataFactory import DataTypeEnum,CustomDataLoader,CustomDataset
from core.models.mymodel_factory import Model
from core.utils import preprocess
import core.mytrainer as trainer

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cpu:0')
parser.add_argument('--start_ite', type=int, default=0)#再次训练时的起始迭代步，保持连续
# data
parser.add_argument('--dataset_root', type=str, default='F:/FTPRoot/rock-ai-srm/DataSet/')
parser.add_argument('--dataset_name', type=str, default='PBAll')
parser.add_argument('--save_dir', type=str, default='checkpoints/')
parser.add_argument('--gen_frm_dir', type=str, default='results/')
parser.add_argument('--input_length', type=int, default=4)
parser.add_argument('--total_length', type=int, default=8)
parser.add_argument('--train_data_ratio', type=float, default=0.9)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--px', type=int, default=1)
parser.add_argument('--img_channel', type=int, default=1)
parser.add_argument('--data_type', type=str, default=DataTypeEnum.Lcap2slice)
parser.add_argument('--img_norm', type=int, default=0)
parser.add_argument('--rand_or_fix', type=int, default=1)
parser.add_argument('--rand_flag', type=int, default=1)
# model
parser.add_argument('--model_name', type=str, default='predrnn_v3')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)
parser.add_argument('--decouple_beta', type=float, default=0.1)
parser.add_argument('--attention_hidden_dims', type=int, default=4)
parser.add_argument('--loss_mode', type=str, default='mse')
# reverse scheduled sampling
parser.add_argument('--reverse_scheduled_sampling', type=int, default=0)
parser.add_argument('--r_sampling_step_1', type=float, default=25000)
parser.add_argument('--r_sampling_step_2', type=int, default=50000)
parser.add_argument('--r_exp_alpha', type=int, default=5000)
# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--max_iterations', type=int, default=10)
parser.add_argument('--display_interval', type=int, default=1)
parser.add_argument('--test_interval', type=int, default=3)
parser.add_argument('--snapshot_interval', type=int, default=1)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

# visualization of memory decoupling
parser.add_argument('--visual', type=int, default=0)
parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

# action-based predrnn
parser.add_argument('--injection_action', type=str, default='concat')
parser.add_argument('--conv_on_input', type=int, default=1, help='conv on input')
parser.add_argument('--res_on_conv', type=int, default=1, help='res on conv')
parser.add_argument('--num_action_ch', type=int, default=4, help='num action ch')

# motion-aware
parser.add_argument('--tau', type=int, default=8, help='tau')
parser.add_argument('--cell_mode', type=str, default='normal', help='cell_mode')
parser.add_argument('--model_mode', type=str, default='recall', help='model_mode')
parser.add_argument('--sr_size', type=int, default=4, help='sr_size')
parser.add_argument('--img_height', type=int, default=128, help='img_height')

args = parser.parse_args()
print(args)
 # load data
DataRoot=args.dataset_root
DataSetName=args.dataset_name
trainRatio=args.train_data_ratio
dataType=args.data_type
randOrFix=args.rand_or_fix==1
px=args.px #fracture line width by px

DS=CustomDataset(dataSetName=DataSetName,dataRoot=DataRoot,dataType=dataType,randOrFix=randOrFix,sequence_length=args.total_length,input_length=args.input_length,
                 useClips=True,withChannel=True,ChannelLast=True,px=px,size=args.img_width,randFlag=args.rand_flag)
cDL=CustomDataLoader(DS,train_batch_size=args.batch_size,test_batch_size=args.batch_size,train_ratio=trainRatio,shuffle=True)

def reserve_schedule_sampling_exp(batch_size,itr):
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (batch_size, args.input_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))

    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.total_length - 2):
            if j < args.input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - 2,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return real_input_flag


def schedule_sampling(eta,batch_size, itr):
    zeros = np.zeros((batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


def train_wrapper(model,writer):
    if args.pretrained_model:
        model.load(args.pretrained_model)
   
    train_dataLoader,test_dataLoader=cDL.train_dataloader,cDL.test_dataloader
    eta = args.sampling_start_value
    itr=args.start_ite
    while itr<args.max_iterations:
        if itr==args.max_iterations:
            model.save(itr)#保留最后一个
            break
        batch=next(iter(train_dataLoader))
        itr+=1
        sequences,durations=batch['sequences'],batch['durations']
        if args.img_norm:
            sequences=sequences/255
        # patch trick!!!
        ims=preprocess.reshape_patch(sequences,args.patch_size)
        current_batch_size=ims.shape[0]
        if args.reverse_scheduled_sampling == 1:
            real_input_flag = reserve_schedule_sampling_exp(current_batch_size,itr)
        else:
            eta, real_input_flag = schedule_sampling(eta,current_batch_size, itr)
        #这个predrnn需要带入间距
        if args.model_name in ["predrnn_v3","predrnn_sa_v2"]:
            trainer.train_with_interval(model, ims,durations,real_input_flag, args, itr,writer)
        else:
            trainer.train(model, ims, real_input_flag, args, itr,writer)

        if itr % args.snapshot_interval == 0:
            model.save(itr)
        if itr % args.test_interval == 0:
            trainer.test(model,test_dataLoader,args,itr,writer)
            
        

def test_wrapper(model,writer):
    model.load(args.pretrained_model)
    trainer.test(model, cDL.test_dataloader, args,args.start_ite, writer)#这里只是测试，统计测试集的情况

save_path=f"{args.save_dir}/{args.model_name}_{args.dataset_name}_{args.data_type}_{args.rand_or_fix}_{args.img_width}_{args.px}_{args.total_length}"
# if os.path.exists(save_path):
#     shutil.rmtree(save_path)
os.makedirs(save_path,exist_ok=True)
gen_path=f"{args.gen_frm_dir}/{args.model_name}_{args.dataset_name}_{args.data_type}_{args.rand_or_fix}_{args.img_width}_{args.px}_{args.total_length}"
# if os.path.exists(gen_path):
#     shutil.rmtree(gen_path)
os.makedirs(gen_path,exist_ok=True)

print('Initializing models')

model = Model(args)
from core.utils.tensorboard import TensorBoard

if args.is_training:
    writer=TensorBoard(f'./train_log/{args.model_name}_{args.dataset_name}_{args.data_type}_{args.rand_or_fix}_{args.rand_flag}_{args.img_width}_{args.px}_{args.total_length}_{args.loss_mode}')
    train_wrapper(model,writer)
else:
    writer=TensorBoard(f'./test_log/{args.model_name}_{args.dataset_name}_{args.data_type}_{args.rand_or_fix}_{args.rand_flag}_{args.img_width}_{args.px}_{args.total_length}_{args.loss_mode}')
    test_wrapper(model,writer)

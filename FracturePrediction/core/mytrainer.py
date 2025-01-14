import os.path
import datetime
import cv2
import numpy as np
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess, metrics
import lpips
import torch
def record(name,time,itr,loss):
    with open(name+'.csv',"a+") as f:
        f.write("{0},{1},{2}\n".format(time,itr,loss))
    

loss_fn_alex = lpips.LPIPS(net='alex')

def train(model, ims, real_input_flag, configs, itr,writer):
    cost = model.train(ims, real_input_flag)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        cost += model.train(ims_rev, real_input_flag)
        cost = cost / 2
    current_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    gen_path=f"{configs.gen_frm_dir}/{configs.model_name}_{configs.dataset_name}_{configs.data_type}_{configs.rand_or_fix}_{configs.img_width}_{configs.px}_{configs.total_length}"
    record(gen_path,current_time,itr,cost)
    writer.add_scalar("train/loss",float(cost),itr)
    if itr % configs.display_interval == 0:
        print(current_time, 'itr: ' + str(itr))
        print('training loss: ' + str(cost))

#注意数据传入的是与第一帧的距离
def train_with_interval(model,ims,distance,real_input_flag, configs, itr,writer):
    cost = model.trainV3(ims,distance, real_input_flag)
    current_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    gen_path=f"{configs.gen_frm_dir}/{configs.model_name}_{configs.dataset_name}_{configs.data_type}_{configs.rand_or_fix}_{configs.img_width}_{configs.px}_{configs.total_length}"
    record(gen_path,current_time,itr,cost)
    writer.add_scalar("train/loss",float(cost),itr)
    if itr % configs.display_interval == 0:
        print(current_time, 'itr: ' + str(itr))
        print('training loss: ' + str(cost))

def test(model, test_dataLoader, configs, itr,writer=None):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    gen_path=f"{configs.gen_frm_dir}/{configs.model_name}_{configs.dataset_name}_{configs.data_type}_{configs.rand_or_fix}_{configs.img_width}_{configs.px}_{configs.total_length}"
    res_path = os.path.join(gen_path, str(itr))
    os.makedirs(res_path,exist_ok=True)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    lp = []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    # reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length
    current_total_batch_size=0
    for batch in test_dataLoader:
        batch_id = batch_id + 1
        test_ims = batch["sequences"]
        test_distance=batch["durations"]
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        test_ims = test_ims[:, :, :, :, :configs.img_channel]
        if configs.img_norm:
                test_ims=test_ims/255
        current_batch_size=test_ims.shape[0]
        current_total_batch_size+=current_batch_size
        #mask的batch size与输入保持一致，避免最后一个batch size变化
        real_input_flag = np.zeros(
            (current_batch_size,
            configs.total_length - mask_input - 1,
            configs.img_width // configs.patch_size,
            configs.img_width // configs.patch_size,
            configs.patch_size ** 2 * configs.img_channel))

        if configs.reverse_scheduled_sampling == 1:
            real_input_flag[:, :configs.input_length - 1, :, :] = 1.0
        #添加一个测试
        if configs.model_name in ["predrnn_v3","predrnn_sa_v2","predrnn_v32"]:
            img_gen = model.testV3(test_dat,test_distance,real_input_flag)
        else:
            img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length 
        img_out = img_gen[:, -output_length:]

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            # cal lpips
            img_x = np.zeros([current_batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 1]
                img_x[:, 2, :, :] = x[:, :, :, 2]
            else:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 0]
                img_x[:, 2, :, :] = x[:, :, :, 0]
            img_x = torch.FloatTensor(img_x)
            img_gx = np.zeros([current_batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 1]
                img_gx[:, 2, :, :] = gx[:, :, :, 2]
            else:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 0]
                img_gx[:, 2, :, :] = gx[:, :, :, 0]
            img_gx = torch.FloatTensor(img_gx)
            lp_loss = loss_fn_alex(img_x, img_gx)
            lp[i] += torch.mean(lp_loss).item()

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)

            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            for b in range(current_batch_size):
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True,channel_axis=2)
                ssim[i] += score

        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
                if writer:
                    writer.add_image(f"val/{itr}_{i+1}_gt",file_name,itr)
            for i in range(output_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
                if writer:
                    writer.add_image(f"val/{itr}_{i+ 1 + configs.input_length}_pd",file_name,itr)
    #计算全部的，而不是每个批次的
    avg_mse = avg_mse / (current_total_batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        vi=img_mse[i] / (current_total_batch_size)
        print(vi)
        writer.add_scalar(f"test/mse_frame{i}",vi,itr)

    ssim = np.asarray(ssim, dtype=np.float32) / (current_total_batch_size)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])
        writer.add_scalar(f"test/ssim_frame{i}",ssim[i],itr)

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])
        writer.add_scalar(f"test/psnr_frame{i}",psnr[i],itr)

    lp = np.asarray(lp, dtype=np.float32) / batch_id
    print('lpips per frame: ' + str(np.mean(lp)))
    for i in range(configs.total_length - configs.input_length):
        print(lp[i])
        writer.add_scalar(f"test/lpips_frame{i}",lp[i],itr)

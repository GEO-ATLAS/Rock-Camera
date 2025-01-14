import numpy as np
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess, metrics
import lpips
import torch
import os
import datetime
import cv2
loss_fn_alex = lpips.LPIPS(net='alex')

def test(model,test_dataloader,configs,itr,writer=None):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    res_path = os.path.join(configs.gen_frm_dir+"/"+configs.model_name+"_"+configs.dataset_name+"_"+configs.data_type, str(itr))
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

    current_total_batch_size=0
    for batch in test_dataloader:
        batch_id = batch_id + 1
        test_ims = batch["sequences"]
        test_ims=test_ims.permute(0,4,1,2,3).to(configs.device)
        test_distance=batch["durations"]
        current_batch_size=test_ims.shape[0]
        current_total_batch_size+=current_batch_size
        if configs.img_norm:
                test_ims=test_ims/255
        torch.cuda.empty_cache()
        img_gen = model.test(test_ims)
        output_length = configs.total_length - configs.input_length 
        img_out = img_gen[:,:,-output_length:]
        # MSE per frame
        # convert to channel last
        test_ims=test_ims.permute(0,2,3,4,1).cpu().detach().numpy()
        img_out=img_out.permute(0,2,3,4,1).cpu().detach().numpy()
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
            os.makedirs(path,exist_ok=True)
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

    avg_mse = avg_mse / (current_total_batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (current_total_batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (current_total_batch_size)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])

    lp = np.asarray(lp, dtype=np.float32) / batch_id
    print('lpips per frame: ' + str(np.mean(lp)))
    for i in range(configs.total_length - configs.input_length):
        print(lp[i])
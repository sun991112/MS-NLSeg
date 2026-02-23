import sys
import numpy as np
import torch
from dataloader import post_process_mask
import numpy as np
from model.loss import FixedLoss
from scipy.spatial import KDTree
import nibabel as nib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def compute_iter_metrics(pred, target, loss):
    metrics = {}
    # 初始化累积变量
    TP, FP, FN, TN = 0, 0, 0, 0

    # 遍历所有batch的预测结果和真实标签


    # 展平张量
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    # 计算True Positive, False Positive, True Negative, False Negative
    TP += (pred * target).sum().item()
    FP += (pred * (1 - target)).sum().item()
    FN += ((1 - pred) * target).sum().item()
    TN += ((1 - pred) * (1 - target)).sum().item()

    # 计算指标
    dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    for k,lo in enumerate(loss):
        metrics[f'loss"{str(k)}'] = lo.item()


    # 定义一个计算两个坐标集合之间的95% directed Hausdorff距离的辅助函数

    # 返回结果
    metrics['dice']=dice
    metrics['iou']=iou
    metrics['precision']=precision
    metrics['recall']=recall
    metrics['f1']=f1
    return metrics


def _pair_predict(model, model_name, device, x1, x2):
    """统一的两时相前向：优先用 test_forward(x1,x2)；否则通道拼接后走 forward(inp)"""
    x1 = x1.to(device).float()
    x2 = x2.to(device).float()
    #print(x1.device,next(model.parameters()).device)
    with torch.no_grad():
        # 1) 优先走成对接口
        if hasattr(model, "test_forward"):
            inp = torch.cat(
                [x1[:, 0:1, :, :], x2[:, 0:1, :, :],
                 x1[:, 0:1, :, :] - x2[:, 0:1, :, :]],
                dim=1  # channel 维
            )
            pred = model.test_forward(inp)
        elif hasattr(model, "forward") and model.forward.__code__.co_argcount >= 3:
            inp = torch.cat(
                [x1[:, 0:1, :, :], x2[:, 0:1, :, :],
                 x1[:, 0:1, :, :] - x2[:, 0:1, :, :]],
                dim=1  # channel 维
            )
            image_data_tensor=inp
            if model_name == "DAPSAM":

                pred = model(inp, False, 512)['masks']
            # 一些模型 forward(x1, x2)
            elif model_name == 'TriD':
                pred = model(inp, phase='test')
            elif model_name == 'ASDG':
                predict1, predict2, aug_img1_, aug_img2_, aug_img1, aug_img2 = model.foward1(image_data_tensor)
                new_predict1, new_predict2, miloss = model.foward2(image_data_tensor, aug_img1_, aug_img1, aug_img2)
                pred = new_predict1

            else:

                pred = model(x1, x2)
        else:
            # 2) 兜底：通道拼接 [B,3,H,W] + [B,3,H,W] -> [B,6,H,W]
            inp = torch.cat(
                [x1[:, 0:1, :, :], x2[:, 0:1, :, :],
                 x1[:, 0:1, :, :] - x2[:, 0:1, :, :]],
                dim=1  # channel 维
            )
            image_data_tensor=inp
            if model_name == 'DeSAM':
                box=torch.from_numpy(np.array([[0, 0,512, 512]]*image_data_tensor.shape[0])).float().to(device)
                with torch.no_grad():
                    image_embeddings=model.image_encoder(image_data_tensor)
                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=None,
                        boxes=box,
                        masks=None,
                    )
                mask_pred, _ = model.mask_decoder(
                    image_embeddings=image_embeddings,  # (B, 256, 64, 64)
                    image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                    multimask_output=False,
                )
                from dataloader import  post_process_training_mask
                pred=post_process_training_mask(mask_pred,(512,512))
            elif model_name == 'ASDG':
                predict1, predict2, aug_img1_, aug_img2_, aug_img1, aug_img2 = model.foward1(image_data_tensor)
                new_predict1, new_predict2, miloss = model.foward2(image_data_tensor, aug_img1_, aug_img1, aug_img2)
                pred = new_predict1

            else:
                pred = model(inp)

    # 统一取出 [B,1,H,W]
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    # 有些模型返回 dict
    if isinstance(pred, dict):
        # 你自己项目里常见 key 是 'masks'；若不同请改成你用的 key
        pred = pred.get("masks", None)
        assert pred is not None, "pred 是 dict，但未找到 'masks' 键，请根据你的模型输出调整。"

    # 标准化到 [B,1,H,W]
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    # with torch.no_grad():
    #     print(
    #         f"[pred stats] "
    #         f"min={pred.min().item():.4f}, "
    #         f"max={pred.max().item():.4f}, "
    #         f"mean={pred.mean().item():.4f}"
    #     )

    return torch.sigmoid(pred)  # [B,1,H,W]


def test(model, test_set, device, model_name,
         save_data=False, single_axis=False, eval=False):
    if eval:
        model.eval()
    model = model.to(device)

    metric_this_epoch_list = []
    sum_dict, count_dict = {}, {}

    if save_data:
        results = []

    loss_func = FixedLoss()
    for data in test_set:
        # ------- 取基础信息 -------
        mask_np = data['mask']  # [Z,Y,X]
        brain_mask = data['brain_mask']
        mask_np = mask_np.astype(np.uint8)  # 先转换 dtype
        mask = torch.from_numpy(mask_np).float()
        sag_size = data['sag_ori_size']  # (H,W)
        cor_size = data['cor_ori_size']  # (H,W)
        axi_size = data['axi_ori_size']  # (H,W)

        sag_len = data['sag_len']
        cor_len = data['cor_len']
        axi_len = data['axi_len']

        sid = data['id']
        info = data['info']
        affine = data['affine']

        # ------- 成对 loader -------
        sag_loader1 = data.get('sagittal_loader1')
        sag_loader2 = data.get('sagittal_loader2')
        cor_loader1 = data.get('coronal_loader1')
        cor_loader2 = data.get('coronal_loader2')
        axi_loader1 = data.get('axial_loader1')
        axi_loader2 = data.get('axial_loader2')

        # ------- 为三轴结果分配容器 -------
        sag_result = torch.zeros((sag_len, sag_size[0], sag_size[1]))  # [Z, H, W] (按你原逻辑)
        sag_x1= torch.zeros((sag_len, sag_size[0], sag_size[1]))
        sag_x2=torch.zeros((sag_len, sag_size[0], sag_size[1]))
        cor_result = torch.zeros((cor_size[0], cor_len, cor_size[1]))  # [H, Z, W]
        axi_result = torch.zeros((axi_size[0], axi_size[1], axi_len))  # [H, W, Z]

        # ------- 选择轴 -------
        if not single_axis:
            axes_to_slice = [0, 1, 2]

        else:

            # old:选择切片最多的轴
            # axes_to_slice = [[sag_len, cor_len, axi_len].index(max([sag_len, cor_len, axi_len]))]

            # new: 与 dataloader 对齐（谁有 loader 用谁）
            if data['sagittal_loader1'] is not None and data['sagittal_loader2'] is not None:

                axes_to_slice = [0]
            elif data['coronal_loader1'] is not None and data['coronal_loader2'] is not None:

                axes_to_slice = [1]
            elif data['axial_loader1'] is not None and data['axial_loader2'] is not None:

                axes_to_slice = [2]
            else:
                print("[WARN] no available loader for sample:", data.get('id'))
                continue

        # ------- 按轴推理 -------
        for axis in axes_to_slice:
            curr_sag_index = 0
            curr_cor_index = 0
            curr_axi_index = 0

            # ================== PNG 保存目录 ==================
            png_save_dir = "/home/sunyl/mount2T/change-detection/png"
            os.makedirs(png_save_dir, exist_ok=True)

            # ========== 矢状面 ==========
            if axis == 0 and sag_loader1 is not None and sag_loader2 is not None:
                for x1, x2 in zip(sag_loader1, sag_loader2):
                    # x1, x2: [B, 1, H, W]

                    nonzero_mask = (x1 != 0) & (x2 != 0)  # bool mask, shape [B,1,H,W]

                    x1 = x1 * nonzero_mask
                    x2 = x2 * nonzero_mask

                    pred = _pair_predict(model, model_name, device, x1, x2)  # [B,1,H,W]


                    #pred组装
                    pred = pred.squeeze(1).detach().cpu().numpy()  # [B,H,W]
                    pred = post_process_mask(pred, sag_size)  # list/np->[B,H,W]
                    pred = torch.from_numpy(pred).float()  # [B,H,W]

                    x1_0= x1[:,0,:,:].squeeze(1).detach().cpu().numpy()  # [B,H,W]
                    x1_0 = post_process_mask(x1_0, sag_size)  # list/np->[B,H,W]
                    x1_0 = torch.from_numpy(x1_0).float()  # [B,H,W]

                    x2_0 = x2[:,0,:,:].squeeze(1).detach().cpu().numpy()  # [B,H,W]
                    x2_0= post_process_mask(x2_0, sag_size)  # list/np->[B,H,W]
                    x2_0 = torch.from_numpy(x2_0).float()  # [B,H,W]

                    # ================== ✅ PNG 保存（就在这里） ==================
                    B = x1_0.size(0)
                    x2_1 = x2_0-x1_0
                    for b in range(B):
                        slice_index = curr_sag_index + b

                        slice_np = x2_1[b].numpy()
                        plt.imsave(
                            os.path.join(png_save_dir, f"sag_x21_{slice_index:04d}.png"),
                            slice_np,
                            cmap="gray"
                        )

                        slice_np = x1_0[b].numpy()
                        plt.imsave(
                            os.path.join(png_save_dir, f"sag_x1_{slice_index:04d}.png"),
                            slice_np,
                            cmap="gray"
                        )

                        slice_np = x2_0[b].numpy()
                        plt.imsave(
                            os.path.join(png_save_dir, f"sag_x2_{slice_index:04d}.png"),
                            slice_np,
                            cmap="gray"
                        )

                    # ================== 原有结果组装 ==================
                    B = pred.size(0)
                    sag_result[curr_sag_index:curr_sag_index + B, :, :] = pred
                    sag_x1[curr_sag_index:curr_sag_index + B, :, :] = x1_0
                    sag_x2[curr_sag_index:curr_sag_index + B, :, :] = x2_0
                    curr_sag_index += B
                torch.cuda.empty_cache()

            # ========== 冠状面 ==========
            if axis == 1 and cor_loader1 is not None and cor_loader2 is not None:
                for x1, x2 in zip(cor_loader1, cor_loader2):
                    # x1, x2: [B, 1, H, W]

                    nonzero_mask = (x1 != 0) & (x2 != 0)  # bool mask, shape [B,1,H,W]

                    x1 = x1 * nonzero_mask
                    x2 = x2 * nonzero_mask

                    pred = _pair_predict(model, model_name, device, x1, x2)  # [B,1,H,W]
                    pred = pred.squeeze(1).detach().cpu().numpy()  # [B,H,W]
                    pred = post_process_mask(pred, cor_size)
                    pred = torch.from_numpy(pred).float()  # [B,H,W]
                    B = pred.size(0)
                    # 注意你原逻辑把 batch 维映射到 Z（第二维），做了转置
                    cor_result[:, curr_cor_index:curr_cor_index + B, :] = pred.transpose(0, 1)
                    curr_cor_index += B
                torch.cuda.empty_cache()

            # ========== 轴位 ==========
            if axis == 2 and axi_loader1 is not None and axi_loader2 is not None:
                for x1, x2 in zip(axi_loader1, axi_loader2):
                    # x1, x2: [B, 1, H, W]

                    nonzero_mask = (x1 != 0) & (x2 != 0)  # bool mask, shape [B,1,H,W]

                    x1 = x1 * nonzero_mask
                    x2 = x2 * nonzero_mask

                    pred = _pair_predict(model, model_name, device, x1, x2)  # [B,1,H,W]
                    pred = pred.squeeze(1).detach().cpu().numpy()  # [B,H,W]
                    pred = post_process_mask(pred, axi_size)
                    pred = torch.from_numpy(pred).float()  # [B,H,W]
                    B = pred.size(0)
                    # 你的原逻辑把 batch 维填到 Z 这一维（第三维）
                    axi_result[:, :, curr_axi_index:curr_axi_index + B] = pred.permute(1, 2, 0)
                    curr_axi_index += B
                torch.cuda.empty_cache()

        # save_dir = "/home/sunyl/mount2T/change-detection/test_output1"
        # out_path = os.path.join(save_dir, f"{sid}_sag.nii.gz")
        # #print(final_result.shape, final_result.transpose(0, 2).shape)
        # sag = sag_result.transpose(0, 2).cpu().numpy().astype(np.float32)
        # nii_img = nib.Nifti1Image(sag, affine=affine)
        # nib.save(nii_img, out_path)

        # ------- 阈值化 -------
        sag_result = (sag_result >= 0.5).float()
        cor_result = (cor_result >= 0.5).float()
        axi_result = (axi_result >= 0.5).float()

        # ------- 三轴融合 OR 单轴 -------
        if not single_axis:
            final_result = torch.logical_or(torch.logical_or(sag_result, cor_result), axi_result).float()
        else:
            # 单轴时，final_result 应该取对应轴的结果
            if axes_to_slice[0] == 0:
                final_result = sag_result
            elif axes_to_slice[0] == 1:
                final_result = cor_result
            else:
                final_result = axi_result

        # ------- 计算损失 & 指标 -------
        #     # ------- 计算指标 -------
        if torch.sum(mask) == 0:
            continue  # 跳过该病例，不计入平均值
        final_result=final_result*brain_mask

        # --------- slice 清零规则 ----------
        D = final_result.shape[0]

        # 前 0–50 层清零
        final_result[0:30, :, :] = 0
        mask[0:30, :, :] = 0

        # 后 140–182 层清零（注意不要越界）
        final_result[140:min(182, D), :, :] = 0
        mask[140:min(182, D), :, :] = 0


        ##################

        losses = loss_func(final_result, mask)
        metric = compute_iter_metrics(final_result, mask, losses)
        print(metric)
        metric_this_epoch_list.append(metric)

        ## ✅✅ 这里新增：保存 final_result 为 nii.gz 文件
        # #mni_affine= nib.load(r"/home/wy3atjlu/zhaozq/mount8t/subjects/sunyl/MNI152_T1_1mm.nii.gz")
        save_dir="/home/sunyl/hdd/change-detection/test_miccai1"
        print(sid)
        out_path = os.path.join(save_dir, f"{sid}_pred.nii.gz")
        #print(final_result.shape, final_result.transpose(0, 2).shape)
        final_np = final_result.transpose(0, 2).cpu().numpy().astype(np.float32)
        nii_img = nib.Nifti1Image(final_np, affine=affine)
        nib.save(nii_img, out_path)
        # #
        # out_path = os.path.join(save_dir, f"{sid}_mask.nii.gz")
        # final_np = mask.transpose(0,2).cpu().numpy().astype(np.float32)
        # nii_img = nib.Nifti1Image(final_np, affine=affine)
        # nib.save(nii_img, out_path)

        # out_path = os.path.join(save_dir, f"{sid}_x1.nii.gz")
        # final_np = sag_x1.transpose(0, 2).cpu().numpy().astype(np.float32)
        # nii_img = nib.Nifti1Image(final_np, affine=affine)
        # nib.save(nii_img, out_path)
        #
        # out_path = os.path.join(save_dir, f"{sid}_x2.nii.gz")
        # final_np = sag_x2.transpose(0, 2).cpu().numpy().astype(np.float32)
        # nii_img = nib.Nifti1Image(final_np, affine=affine)
        # nib.save(nii_img, out_path)

        print(f"💾 已保存预测结果: {out_path}")
        print(sid)

        if save_data:
            results.append([sid, info, metric, final_result])

    # ------- 汇总平均 -------
    for d in metric_this_epoch_list:
        for k, v in d.items():
            if k in sum_dict:
                sum_dict[k] += v
                count_dict[k] += 1
            else:
                sum_dict[k] = v
                count_dict[k] = 1

    avg_dict = {k: (sum_dict[k] / count_dict[k]) for k in sum_dict}

    if save_data:
        return avg_dict, results
    else:
        return avg_dict






if __name__=='__main__':

    import nibabel
    from dataloader import Whole_dataset,test_dataset,test_dataloader
    # from model.mySAM.segment_anything.modeling.mysam_ import USam as USamAE
    # from model.mySAM.segment_anything.modeling.mysam_lora import USam as USamLoRA
    from model.unet_diff import UNet
    from model.sam.segment_anything import sam_model_registry as DAPSAM

    from model.mySAM_detection.segment_anything.modeling.mysam_custom_denseprompt_detection_x1x2 import \
        USamDiff as MySAM_GEGlora
    from model.mySAM_detection.segment_anything.modeling.mysam_simple_detection import \
        USamDiff as simple_detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model=DAPSAM['vit_b'](image_size=512,
    #                                         num_classes=1,
    #                                         checkpoint='/home/sunyl/hdd/change-detection/change_detection/model/mySAM_detection/sam_vit_b_01ec64.pth', pixel_mean=[0, 0, 0],
    #                                         pixel_std=[1, 1, 1])[0]
    #model=USamLoRA('/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/model/mySAM/sam_vit_b_01ec64.pth')
    #path='/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/ssdg_checkpoints/mysam/prostate_segmentation/BMC_USam_LoRA/best_model.pth'
    #model=MySAM_GEGlora('/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/model/mySAM/sam_vit_b_01ec64.pth')
    model=UNet()
    #model = MySAM_GEGlora('/home/sunyl/hdd/change-detection/change_detection/model/mySAM_detection/sam_vit_b_01ec64.pth')
    path="/home/sunyl/hdd/change-detection/change_detection/checkpoints/LesSeg_UNet/34_model.pth"
    # model = torch.nn.DataParallel(model)
    print("1")
    model.load_state_dict(torch.load(path))
    # state_dict=torch.load(path)
    # ckpt_keys = set(state_dict.keys())
    # model_keys = set(model.state_dict().keys())
    #
    # unexpected_keys = ckpt_keys - model_keys
    #
    # print("Parameters in checkpoint but NOT in model:")
    # for k in sorted(unexpected_keys):
    #     print(k)
    print("2")
    model = model.to(device)
    raw_data_folder="/home/sunyl/hdd/change-detection/change_detection/dataset/wmh_segmentation/final_dataset_select"
    #sites=['HK','RUNMC1','BIDMC','RUNMC','UCL']
    #sites= ['PediMS','LesSeg','openms','trainingmiccai21']
    #sites = ['ISBI']
    sites = ['testingmiccai21','trainingmiccai21']
    testing_dataset=Whole_dataset(dataset_path=raw_data_folder, sites=sites, task='WMH')
    testing_dataset = test_dataset(testing_dataset, single_axis=True)




    print("[CHECK] len(testing_dataset) =", len(testing_dataset))
    assert len(testing_dataset) > 0, "testing_dataset 为空"

    # # 看第一个样本的键和每个方向的切片数
    # s0 = testing_dataset[0]
    # print("[CHECK] sample[0] keys:", list(s0.keys()))
    # for k in ["sag_slices", "sag_slices1", "sag_slices2",
    #           "cor_slices", "cor_slices1", "cor_slices2",
    #           "axi_slices", "axi_slices1", "axi_slices2"]:
    #     if k in s0:
    #         print(f"[CHECK] {k} len =", len(s0[k]))
    # print("[CHECK] sizes: sag", s0.get("sag_size"), " cor", s0.get("cor_size"), " axi", s0.get("axi_size"))

    # model.eval()
    # torch.set_grad_enabled(False)

    test_dataloader=test_dataloader(testing_dataset, model, 'cuda',single_axis=True,model_name='MySam_GEGlora')
    #avg_dict=test(model,test_dataloader, 'cuda', save_data=False,single_axis=True,model_name='MySam_GEGlora',eval=True)
    avg_dict,results= test(model, test_dataloader, 'cuda', save_data=True, single_axis=True, model_name='MySam_GEGlora',
                    eval=True)
    print("avg_dicet",avg_dict)

    # ================== 保存为 Excel ==================
    rows = []

    for sid, info, metric, _ in results:
        row = {}
        row['sid'] = sid

        # 如果 info 里有 site / timepoint / hospital
        if isinstance(info, dict):
            for k, v in info.items():
                row[k] = v
        else:
            row['info'] = info

        # metric 是 dict（dice / iou / loss 等）
        for k, v in metric.items():
            row[k] = float(v)

        rows.append(row)

    df = pd.DataFrame(rows)

    save_dir ="/home/sunyl/hdd/change-detection/test_miccai1"
    os.makedirs(save_dir, exist_ok=True)

    excel_path = os.path.join(save_dir, "test_results.xlsx")
    df.to_excel(excel_path, index=False)

    print(f"📊 已保存测试结果到 Excel: {excel_path}")
    # for result in b:
    #     id=result[0]
    #     info=result[1]
    #     metric=result[2]
    #     array=result[3]
    #     image=nibabel.Nifti1Image(array, info[0], info[1])
    #     nibabel.save(image, os.path.join('/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/output/temp', id.replace('FLAIR', 'mask')))




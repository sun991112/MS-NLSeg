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
    TP, FP, FN, TN = 0, 0, 0, 0
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    # True Positive, False Positive, True Negative, False Negative
    TP += (pred * target).sum().item()
    FP += (pred * (1 - target)).sum().item()
    FN += ((1 - pred) * target).sum().item()
    TN += ((1 - pred) * (1 - target)).sum().item()

    # metrics
    dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    for k,lo in enumerate(loss):
        metrics[f'loss"{str(k)}'] = lo.item()
        
    # return results
    metrics['dice']=dice
    metrics['iou']=iou
    metrics['precision']=precision
    metrics['recall']=recall
    metrics['f1']=f1
    return metrics

def _pair_predict(model, model_name, device, x1, x2):
    x1 = x1.to(device).float()
    x2 = x2.to(device).float()
    #print(x1.device,next(model.parameters()).device)
    with torch.no_grad():
        if hasattr(model, "test_forward"):
            inp = torch.cat(
                [x1[:, 0:1, :, :], x2[:, 0:1, :, :],
                 x1[:, 0:1, :, :] - x2[:, 0:1, :, :]],
                dim=1  # channel 
            )
            pred = model.test_forward(inp)
        elif hasattr(model, "forward") and model.forward.__code__.co_argcount >= 3:
            inp = torch.cat(
                [x1[:, 0:1, :, :], x2[:, 0:1, :, :],
                 x1[:, 0:1, :, :] - x2[:, 0:1, :, :]],
                dim=1  # channel 
            )
            image_data_tensor=inp
            if model_name == "DAPSAM":
                pred = model(inp, False, 512)['masks']
            elif model_name == 'TriD':
                pred = model(inp, phase='test')
            elif model_name == 'ASDG':
                predict1, predict2, aug_img1_, aug_img2_, aug_img1, aug_img2 = model.foward1(image_data_tensor)
                new_predict1, new_predict2, miloss = model.foward2(image_data_tensor, aug_img1_, aug_img1, aug_img2)
                pred = new_predict1
            else:
                pred = model(x1, x2)
        else:
            # [B,3,H,W] + [B,3,H,W] -> [B,6,H,W]
            inp = torch.cat(
                [x1[:, 0:1, :, :], x2[:, 0:1, :, :],
                 x1[:, 0:1, :, :] - x2[:, 0:1, :, :]],
                dim=1  # channel 
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
    # [B,1,H,W]
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
        
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
        mask_np = data['mask']  # [Z,Y,X]
        brain_mask = data['brain_mask']
        mask_np = mask_np.astype(np.uint8)  #  dtype
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

        # ------- pair loader -------
        sag_loader1 = data.get('sagittal_loader1')
        sag_loader2 = data.get('sagittal_loader2')
        cor_loader1 = data.get('coronal_loader1')
        cor_loader2 = data.get('coronal_loader2')
        axi_loader1 = data.get('axial_loader1')
        axi_loader2 = data.get('axial_loader2')

        sag_result = torch.zeros((sag_len, sag_size[0], sag_size[1]))  # [Z, H, W] (按你原逻辑)
        sag_x1= torch.zeros((sag_len, sag_size[0], sag_size[1]))
        sag_x2=torch.zeros((sag_len, sag_size[0], sag_size[1]))
        cor_result = torch.zeros((cor_size[0], cor_len, cor_size[1]))  # [H, Z, W]
        axi_result = torch.zeros((axi_size[0], axi_size[1], axi_len))  # [H, W, Z]

        # ------- seclect axes -------
        if not single_axis:
            axes_to_slice = [0, 1, 2]
        else:
            if data['sagittal_loader1'] is not None and data['sagittal_loader2'] is not None:
                axes_to_slice = [0]
            elif data['coronal_loader1'] is not None and data['coronal_loader2'] is not None:
                axes_to_slice = [1]
            elif data['axial_loader1'] is not None and data['axial_loader2'] is not None:
                axes_to_slice = [2]
            else:
                print("[WARN] no available loader for sample:", data.get('id'))
                continue

        for axis in axes_to_slice:
            curr_sag_index = 0
            curr_cor_index = 0
            curr_axi_index = 0

            if axis == 0 and sag_loader1 is not None and sag_loader2 is not None:
                for x1, x2 in zip(sag_loader1, sag_loader2):
                    # x1, x2: [B, 1, H, W]
                    nonzero_mask = (x1 != 0) & (x2 != 0)  # bool mask, shape [B,1,H,W]
                    x1 = x1 * nonzero_mask
                    x2 = x2 * nonzero_mask
                    pred = _pair_predict(model, model_name, device, x1, x2)  # [B,1,H,W]
                    #pred
                    pred = pred.squeeze(1).detach().cpu().numpy()  # [B,H,W]
                    pred = post_process_mask(pred, sag_size)  # list/np->[B,H,W]
                    pred = torch.from_numpy(pred).float()  # [B,H,W]

                    x1_0= x1[:,0,:,:].squeeze(1).detach().cpu().numpy()  # [B,H,W]
                    x1_0 = post_process_mask(x1_0, sag_size)  # list/np->[B,H,W]
                    x1_0 = torch.from_numpy(x1_0).float()  # [B,H,W]

                    x2_0 = x2[:,0,:,:].squeeze(1).detach().cpu().numpy()  # [B,H,W]
                    x2_0= post_process_mask(x2_0, sag_size)  # list/np->[B,H,W]
                    x2_0 = torch.from_numpy(x2_0).float()  # [B,H,W]

                    # ================== ✅ PNG save ==================
                    # B = x1_0.size(0)
                    # x2_1 = x2_0-x1_0
                    # for b in range(B):
                    #     slice_index = curr_sag_index + b

                    #     slice_np = x2_1[b].numpy()
                    #     plt.imsave(
                    #         os.path.join(png_save_dir, f"sag_x21_{slice_index:04d}.png"),
                    #         slice_np,
                    #         cmap="gray"
                    #     )

                    #     slice_np = x1_0[b].numpy()
                    #     plt.imsave(
                    #         os.path.join(png_save_dir, f"sag_x1_{slice_index:04d}.png"),
                    #         slice_np,
                    #         cmap="gray"
                    #     )

                    #     slice_np = x2_0[b].numpy()
                    #     plt.imsave(
                    #         os.path.join(png_save_dir, f"sag_x2_{slice_index:04d}.png"),
                    #         slice_np,
                    #         cmap="gray"
                    #     )

                    B = pred.size(0)
                    sag_result[curr_sag_index:curr_sag_index + B, :, :] = pred
                    sag_x1[curr_sag_index:curr_sag_index + B, :, :] = x1_0
                    sag_x2[curr_sag_index:curr_sag_index + B, :, :] = x2_0
                    curr_sag_index += B
                torch.cuda.empty_cache()

            # ========== cor ==========
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
                    cor_result[:, curr_cor_index:curr_cor_index + B, :] = pred.transpose(0, 1)
                    curr_cor_index += B
                torch.cuda.empty_cache()

            # ========== axi ==========
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
                    axi_result[:, :, curr_axi_index:curr_axi_index + B] = pred.permute(1, 2, 0)
                    curr_axi_index += B
                torch.cuda.empty_cache()

        # save_dir = "/home/test_output"
        # out_path = os.path.join(save_dir, f"{sid}_sag.nii.gz")
        # #print(final_result.shape, final_result.transpose(0, 2).shape)
        # sag = sag_result.transpose(0, 2).cpu().numpy().astype(np.float32)
        # nii_img = nib.Nifti1Image(sag, affine=affine)
        # nib.save(nii_img, out_path)

        sag_result = (sag_result >= 0.5).float()
        cor_result = (cor_result >= 0.5).float()
        axi_result = (axi_result >= 0.5).float()

        if not single_axis:
            final_result = torch.logical_or(torch.logical_or(sag_result, cor_result), axi_result).float()
        else:
            if axes_to_slice[0] == 0:
                final_result = sag_result
            elif axes_to_slice[0] == 1:
                final_result = cor_result
            else:
                final_result = axi_result

        # ------- compute loss and metrics-------
        losses = loss_func(final_result, mask)
        metric = compute_iter_metrics(final_result, mask, losses)
        print(metric)
        metric_this_epoch_list.append(metric)

        ## ✅✅ save
        # #mni_affine= nib.load(r"/home/MNI152_T1_1mm.nii.gz")
        save_dir="/home/test"
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

        print(f"💾 save results: {out_path}")
        print(sid)

        if save_data:
            results.append([sid, info, metric, final_result])

    # ------- avg------
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
    from model.unet_diff import UNet
    from model.sam.segment_anything import sam_model_registry as DAPSAM
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=UNet()
    path="/home/model.pth"
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
    model = model.to(device)
    raw_data_folder="/home/data"
    sites = ['PediMS','LesSeg','openms']
    testing_dataset=Whole_dataset(dataset_path=raw_data_folder, sites=sites, task='WMH')
    testing_dataset = test_dataset(testing_dataset, single_axis=True)

    print("[CHECK] len(testing_dataset) =", len(testing_dataset))
  
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

    test_dataloader=test_dataloader(testing_dataset, model, 'cuda',single_axis=True,model_name='UNet')
    avg_dict,results= test(model, test_dataloader, 'cuda', save_data=True, single_axis=True, model_name='UNet',
                    eval=True)
    print("avg_dicet",avg_dict)

    # ==================  Excel ==================
    rows = []
    for sid, info, metric, _ in results:
        row = {}
        row['sid'] = sid

        if isinstance(info, dict):
            for k, v in info.items():
                row[k] = v
        else:
            row['info'] = info

        for k, v in metric.items():
            row[k] = float(v)
        rows.append(row)

    df = pd.DataFrame(rows)

    save_dir ="/home/test"
    os.makedirs(save_dir, exist_ok=True)

    excel_path = os.path.join(save_dir, "test_results.xlsx")
    df.to_excel(excel_path, index=False)

    print(f"📊 save Excel: {excel_path}")
    # for result in b:
    #     id=result[0]
    #     info=result[1]
    #     metric=result[2]
    #     array=result[3]
    #     image=nibabel.Nifti1Image(array, info[0], info[1])

import os.path
import torch
import torch.nn as nn
from dataloader import Whole_dataset,slice_dataset,test_dataset,test_dataloader
from torch.utils.data import random_split,ConcatDataset

from model.transunet import TransUNet
#from model.unet import UNet
from model.unet_diff import UNet
from model.randconv import UNet_with_randconv
from model.Trid.ResUnet import ResUnet
from model.sam.segment_anything import sam_model_registry as DAPSAM
from model.DeSAM.desam import sam_model_registry as DeSAM
from model.ASDG.GIN import ASDG_model
#from model.mySAM.segment_anything.modeling.mysam_custom_denseprompt_dg import USam as MySAM_dg
# from model.mySAM.segment_anything.modeling.mysam_custom_denseprompt_onlyGEG import USam as MySAM_onlyGEG
#from model.mySAM.segment_anything.modeling.mysam_custom_denseprompt_GEGlora import USam as MySAM_GEGlora
#from model.mySAM.segment_anything.modeling.mysam_custom_denseprompt_GEGloraFDRconloss import USam as MySAM_GEGloraFDRconloss
from model.loss import FixedLoss,RandConvLoss,ASDGLoss2,ASDGLoss1
from model.mySAM_detection.segment_anything.modeling.mysam_custom_denseprompt_detection_x1x2 import USamDiff as MySAM_GEGlora
#from model.mySAM_detection.segment_anything.modeling.mysam_detection_concatlora import USamDiff as MySAM_GEGlora
#from model.mySAM_detection.segment_anything.modeling.mysam_detection_twoadapter import USamDiff as MySAM_GEGlora


from model.mySAM_detection.segment_anything.modeling.mysam_simple_detection import USamDiff as simple_detection

from test import test
import random
import csv
import numpy as np
from torch.utils.data import DataLoader
from options import training_options
import pandas as pd
import time
import shutil
import itertools
from transform import collate_fn_w_transform,collate_fn_wo_transform


#
# #########################
# from PIL import Image
#
# save_dir = "/home/sunyl/hdd/change-detection/output3"
# os.makedirs(save_dir, exist_ok=True)
# #######################

torch.autograd.set_detect_anomaly(True)




def write_metric_to_csv(metric_dict,save_path):
    df=pd.DataFrame([metric_dict])
    columns_order = ['epoch'] + [col for col in df.columns if col != 'epoch']
    df = df[columns_order]
    if not os.path.isfile(save_path):
        df.to_csv(save_path, index=False)
    else:
        df.to_csv(save_path, mode='a',index=False,header=False)

def set_seed(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # make cudnn to be reproducible for performance
    # can be commented for faster training
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compute_iter_metrics(pred, target, loss,threshold=0.5):
    metrics={}
    # 初始化累积变量
    TP, FP, FN, TN = 0, 0, 0, 0

    # 遍历所有batch的预测结果和真实标签

    # 将预测结果二值化
    pred = (torch.sigmoid(pred) > threshold).float()


    # 展平张量
    pred = pred.view(-1)
    target = target.view(-1)

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
        metrics[f'loss"{str(k)}']=lo.item()
    metrics['dice']=dice
    metrics['iou']=iou
    metrics['precision']=precision
    metrics['recall']=recall
    metrics['f1']=f1


    return metrics




if __name__=='__main__':
    #prepare data
    parser = training_options()
    opt = parser.parse()
    seed = opt.seed
    raw_data_folder = opt.raw_data_folder
    test_site=opt.test_site
    model_name = opt.model
    exp_name = opt.exp_name
    checkpoint_dir = opt.checkpoint_dir
    epoch_num = opt.epochs
    base_lr = opt.lr
    model_save_freq = opt.model_save_freq
    test_freq = opt.test_freq
    gpu_ids = opt.gpu_ids
    alpha=opt.loss_param_alpha
    beta=opt.loss_param_beta
    early_stopping=opt.early_stopping
    batch_size=opt.batch_size
    task=opt.task
    single_axis=opt.single_axis
    augment=opt.augment
    eval=opt.eval
    continue_training=opt.continue_training
    continue_checkpoint=opt.continue_checkpoint
    training_slice_path=opt.training_slice_path
    model_dict={
        'UNet': lambda: UNet(),
        'transUNet': lambda: TransUNet(img_dim=512,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=1),
        'DAPSAM': lambda: DAPSAM['vit_b'](image_size=512,
                                            num_classes=1,
                                            checkpoint='/home/sunyl/hdd/change-detection/change_detection/model/mySAM_detection/sam_vit_b_01ec64.pth', pixel_mean=[0, 0, 0],
                                            pixel_std=[1, 1, 1])[0],
        'randConv': lambda: UNet_with_randconv(),
        'TriD': lambda: ResUnet(resnet='resnet34', num_classes=1, pretrained=False, mixstyle_layers=['layer1'], random_type='TriD'),
        'DeSAM': lambda: DeSAM['default'](checkpoint='/home/sunyl/hdd/change-detection/change_detection/model/DeSAM/desam/modeling/sam_vit_h_4b8939.pth'),
        'ASDG': lambda: ASDG_model(),
        'MySamDG':lambda :MySAM_dg('/home/sunyl/hdd/change-detection/change_detection/model/mySAM_detection/sam_vit_b_01ec64.pth'),
        'MySamOnlyGEG': lambda: MySAM_onlyGEG('/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/model/mySAM/sam_vit_b_01ec64.pth'),
        'MySam_GEGlora': lambda: MySAM_GEGlora('/home/sunyl/hdd/change-detection/change_detection/model/mySAM_detection/sam_vit_b_01ec64.pth'),
        'simple_detection': lambda: simple_detection(
            '/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/model/mySAM/sam_vit_b_01ec64.pth'),
        'MySAM_GEGloraFDRconloss' : lambda: MySAM_GEGloraFDRconloss('/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/model/mySAM/sam_vit_b_01ec64.pth'),

    }
    loss_dict={
        'UNet': lambda: FixedLoss(alpha=alpha,beta=beta),
        'transUNet': lambda: FixedLoss(alpha=alpha,beta=beta),
        'TriD': lambda: FixedLoss(alpha=alpha,beta=beta),
        'DAPSAM': lambda: FixedLoss(alpha=alpha,beta=beta),
        'randConv': lambda: RandConvLoss(),
        'DeSAM': lambda: FixedLoss(alpha=alpha,beta=beta),
        'ASDG': lambda: {'loss1':ASDGLoss1(),'loss2':ASDGLoss2()},
        'MySamDG': lambda: FixedLoss(alpha=alpha,beta=beta),
        'MySamOnlyGEG': lambda: FixedLoss(alpha=alpha,beta=beta),
        'MySam_GEGlora': lambda: FixedLoss(alpha=alpha,beta=beta),
        'simple_detection': lambda: FixedLoss(alpha=alpha, beta=beta),
        'MySAM_GEGloraFDRconloss': lambda: FixedLoss(alpha=alpha,beta=beta),

    }
    model_save_path = os.path.join(checkpoint_dir,exp_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    with open(os.path.join(model_save_path,'config.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Parameter', 'Value'])
        for key, value in vars(opt).items():
            writer.writerow([key, value])
    set_seed(seed)

    #--------------------------preparing dataset------------------------------

    testing_dataset=Whole_dataset(dataset_path=raw_data_folder, sites=test_site, task=task)
    #split data into train and test

    slice_training_set = slice_dataset(training_slice_path)
    if augment:
        slice_training_loader = DataLoader(slice_training_set, collate_fn=collate_fn_w_transform,batch_size=batch_size, shuffle=True,drop_last=True)
    else:
        slice_training_loader = DataLoader(slice_training_set, collate_fn=collate_fn_wo_transform, batch_size=batch_size, shuffle=True,drop_last=True)
    testing_dataset=test_dataset(testing_dataset,single_axis=single_axis)
    #--------------------------preparing dataset------------------------------

    #--------------------------preparing device,model------------------------------
    if len(gpu_ids.split(','))>1 and gpu_ids!='cpu':
        #multi gpu training
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        model = model_dict[model_name]()
        model = nn.DataParallel(model)
        device=torch.device('cuda')
    elif gpu_ids=='cpu':
        #cpu training
        device=torch.device('cpu')
        model = model_dict[model_name]()
    elif len(gpu_ids.split(',')) <= 1:
        #single gpu training
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        model=model_dict[model_name]()
        device=torch.device('cuda')
    if continue_training:
        model.load_state_dict(torch.load(continue_checkpoint))
    #--------------------------preparing device,model------------------------------


    # --------------------------transfer dataset to self-defined loader for batched 2D test-------------
    test_data_loader = test_dataloader(testing_dataset, model, device,single_axis=single_axis,model_name=model_name)
    # --------------------------transfer dataset to self-defined loader for batched 2D test-------------


    #---------------------------optimizer,lossfunc,model-----------------------------
    loss=loss_dict[model_name]()
    if model_name=='ASDG':
        opt_1=torch.optim.Adam(model.model.parameters(), lr=base_lr, betas=(0.5, 0.999))
        opt_2= torch.optim.Adam(itertools.chain(model.netG.parameters(), model.netF.parameters(), model.GIN.parameters()),
                                              lr=base_lr, betas=(0.5, 0.999))
        optimizer=None
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr,betas=(0.9, 0.999), weight_decay=0.1)
    best_dsc=0
    es_count=0

    #--------------------------start training---------------------------------------
    for epoch in range(epoch_num):
        train_time=time.time()

        model.to(device)
        model.train()
        metric_this_epoch_list = []
        sum_dict = {}
        count_dict = {}
        for i, data in enumerate(slice_training_loader):
            image_data1 = data['2dimage1']
            image_data2 = data['2dimage2']
            image_mask = data['mask']
            image_data1_tensor = torch.from_numpy(image_data1).float().to(device)
            image_data2_tensor = torch.from_numpy(image_data2).float().to(device)
            image_mask_tensor = torch.from_numpy(image_mask).float().to(device)
            image_data_tensor = torch.cat(
                [image_data1_tensor[:, 0:1, :, :], image_data2_tensor[:, 0:1, :, :],image_data1_tensor[:, 0:1, :, :]- image_data2_tensor[:, 0:1, :, :]],
                dim=1  # channel 维
            )
# ###########################################################
#             def to_numpy(x):
#                 if isinstance(x, torch.Tensor):
#                     return x.detach().cpu().numpy()
#                 elif isinstance(x, np.ndarray):
#                     return x
#                 else:
#                     raise TypeError(f"Unsupported type: {type(x)}")
#
#
#             image_data1 = to_numpy(image_data1)
#             image_data2 = to_numpy(image_data2)
#             image_mask = to_numpy(image_mask)
#
#             B = image_data1.shape[0]
#
#             for b in range(B):
#                 def save_png(arr, name):
#                     # arr: [1, H, W] or [H, W]
#                     if arr.ndim == 3:
#                         arr = arr[0]
#
#                     arr = arr.astype(np.float32)
#                     arr = arr - arr.min()
#                     if arr.max() > 0:
#                         arr = arr / arr.max()
#                     arr = (arr * 255).astype(np.uint8)
#
#                     Image.fromarray(arr).save(name)
#
#
#                 save_png(image_data1[b], f"{save_dir}/iter{i:04d}_b{b}_t0.png")
#                 save_png(image_data2[b], f"{save_dir}/iter{i:04d}_b{b}_t1.png")
#                 save_png(image_mask[b], f"{save_dir}/iter{i:04d}_b{b}_mask.png")
#
#     print("ok")
# #################################################







            if model_name == 'DAPSAM':
                pred = model(image_data_tensor, False, 512)['masks']

            elif  model_name == 'randConv':
                pred,pred2,pred3 = model(image_data_tensor)
            elif model_name == 'TriD':
                pred=model(image_data_tensor,phase='train')
            elif model_name == 'DeSAM':
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
                l1=loss['loss1'](predict1,predict2,image_mask_tensor)[0]
                opt_1.zero_grad()
                l1.backward()
                opt_1.step()
                new_predict1, new_predict2, miloss = model.foward2(image_data_tensor, aug_img1_, aug_img1, aug_img2)
                l2=loss['loss2'](new_predict1,new_predict2,miloss)[0]
                opt_2.zero_grad()
                l2.backward()
                opt_2.step()
                losses=[l1+l2,l1,l2,miloss]
                pred=new_predict1
            elif model_name == 'USamAE':
                pred,l1,l2=model(image_data_tensor)
            elif model_name == 'MySamDG':
                pred,l1,l2=model(image_data_tensor)
            elif model_name == 'MySAM_GEGloraFDRconloss':
                pred,l1=model(image_data_tensor)
            elif model_name == 'MySam_GEGlora':
                pred = model(image_data1_tensor,image_data2_tensor)[0]
            elif model_name == 'UNet':
                pred = model(image_data1_tensor, image_data2_tensor)[0]
            elif model_name == 'simple_detection':
                pred = model(image_data1_tensor, image_data2_tensor)[0]
            else:
                pred = model(image_data_tensor)[0]




            if model_name == 'randConv':
                losses=loss(pred,image_mask_tensor,pred2,pred3)
            elif model_name == 'ASDG':
                pass
            elif model_name=='USamAE':
                losses=[loss(pred,image_mask_tensor)[0]+l1+l2,l1,l2]
            elif model_name=='MySamDG':
                losses=[loss(pred,image_mask_tensor)[0]+l1+l2,l1,l2,loss(pred,image_mask_tensor)[0]]
            elif model_name=='MySAM_GEGloraFDRconloss':
                losses=[loss(pred,image_mask_tensor)[0]+l1,l1]
            else:
                losses= loss(pred, image_mask_tensor)
            total_loss=losses[0]

            if optimizer is not None:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()



            metric_this_iter = compute_iter_metrics(pred, image_mask_tensor,losses)
            metric_this_epoch_list.append(metric_this_iter)

            if i % 2000 == 0:
                print('iteration:{},loss:{}'.format(i, total_loss.item()))
        for d in metric_this_epoch_list:
            for key, value in d.items():
                # 累加相同键的值
                if key in sum_dict:
                    sum_dict[key] += value
                    count_dict[key] += 1
                else:
                    sum_dict[key] = value
                    count_dict[key] = 1

        # 计算平均值
        train_metric = {key: sum_dict[key] / count_dict[key] for key in sum_dict}




        formatted_metrics = {k: '{:.3f}'.format(v) for k, v in train_metric.items()}
        train_time=time.time() - train_time
        print('epoch{},time cost {} second; train_metric:{}'.format( epoch, train_time, formatted_metrics))
        train_metric['epoch']=epoch
        write_metric_to_csv(train_metric, os.path.join(model_save_path, 'train_metric.csv'))

        #开始test
        if epoch%test_freq==0:
            test_time=time.time()
            test_metric=test(model,test_data_loader,device,model_name,False,single_axis,eval)
            print(test_metric)
            test_time=time.time() - test_time
            formatted_metrics = {k: '{:.3f}'.format(v) for k, v in test_metric.items()}
            curr_dsc=test_metric['dice']
            print(' epoch{}, time cost {} second; test_metric:{}'.format( epoch, test_time, formatted_metrics))
            test_metric['epoch']=str(epoch)
            write_metric_to_csv(test_metric, os.path.join(model_save_path, 'test_metric.csv'))
            torch.save(model.state_dict(),os.path.join(model_save_path,str(epoch)+'_model.pth'))
            if curr_dsc>best_dsc:
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
                best_dsc = curr_dsc
                es_count=0
            # else:
            #     es_count+=test_freq
            print('best dsc:{}'.format(best_dsc))
            # if es_count>=early_stopping:
            #     break


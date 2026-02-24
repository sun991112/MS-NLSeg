import os
import pandas
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import binary_erosion
from scipy.spatial import KDTree
import pandas as pd
import numbers
def compute_iter_metrics(pred, target):
    # 初始化累积变量
    TP, FP, FN, TN = 0, 0, 0, 0

    # 遍历所有batch的预测结果和真实标签
    def get_hausdorff_95(test_array, result_array):
        """
            Compute the 95% Hausdorff distance between two binary numpy arrays.
            :param test_array: Numpy array of the ground truth, shape (D, H, W)
            :param result_array: Numpy array of the segmentation result, shape same as test_array
            :return: 95% Hausdorff distance
            """
        # Hausdorff distance is only defined when something is detected
        if np.sum(result_array) == 0:
            return float('nan')

        # Define a 3x3 kernel for erosion
        structure = np.ones((1,3, 3), dtype=bool)

        # Perform binary erosion
        eroded_test_array = binary_erosion(test_array, structure)
        eroded_result_array = binary_erosion(result_array, structure)

        # Edge detection by subtracting the eroded array from the original
        edge_test_array = test_array.astype(bool) & ~eroded_test_array
        edge_result_array = result_array.astype(bool) & ~eroded_result_array

        # Get coordinates of edge voxels
        test_coords = np.column_stack(np.nonzero(edge_test_array))
        result_coords = np.column_stack(np.nonzero(edge_result_array))

        # Function to compute distances between two sets of coordinates using KDTree
        def get_distances_from_a_to_b(a, b):
            if len(a) == 0 or len(b) == 0:
                return np.array([float('inf')])  # Return infinity if any set is empty
            kd_tree = KDTree(a)
            return kd_tree.query(b, k=1)[0]

        # Compute distances from test to result and vice versa
        d_test_to_result = get_distances_from_a_to_b(test_coords, result_coords)
        d_result_to_test = get_distances_from_a_to_b(result_coords, test_coords)

        # Return the 95th percentile of the Hausdorff distances
        return max(np.percentile(d_test_to_result, 95), np.percentile(d_result_to_test, 95))

    # 计算95% Hausdorff距离
    hausdorff_dist_95 = get_hausdorff_95(target,pred)


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



    # 定义一个计算两个坐标集合之间的95% directed Hausdorff距离的辅助函数

    # 返回结果
    metrics = {
        'Dice': dice,
        'IoU': iou,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'H95':hausdorff_dist_95
    }

    return metrics
def statistic(task):

    if task=='WMH':
        soure_mask_dir=r'/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/dataset/wmh_segmentation/raw_data'
        result_save_dir=r'/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/statistics/wmh_segmentation'
        out_put_dir=r'/home/zhaozq/mount2T/multiSite/WMH_segmentation/output'
    if task=='Pros':
        soure_mask_dir=r'/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/dataset/prostate_segmentation/raw_data'
        result_save_dir=r'/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/statistics/prostate_segmentation'
        out_put_dir=r'/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/output/prostate_segmentation'
    for exp in os.listdir(out_put_dir):
        if exp.find('temp')==-1:
            if not os.path.exists(os.path.join(result_save_dir,exp)):
                os.makedirs(os.path.join(result_save_dir,exp))

            groups = defaultdict(list)
            # 遍历字符串列表，将每个字符串按下划线分隔的第一个部分分组
            for s in os.listdir(os.path.join(out_put_dir,exp)):
                print(s)
                prefix = s.split('_')[0]  # 获取以 '_' 分隔的第一个部分
                groups[prefix].append(s)
            # 将字典中的值转换为 list of list
            result = list(groups.values())

            exp_metric={
                                'site':[i[0].split('_')[0] for i in result],
                                'Dice': [],
                                'IoU': [],
                                'Precision': [],
                                'Recall': [],
                                'F1': [],
                                'H95':[]
                            }
            for site_exp in tqdm(result):
                site_metric={
                                'Dice': 0,
                                'IoU': 0,
                                'Precision': 0,
                                'Recall': 0,
                                'F1': 0,
                                'H95':0
                            }
                for fold in site_exp:
                    fold_metric={
                                'Dice': 0,
                                'IoU': 0,
                                'Precision': 0,
                                'Recall': 0,
                                'F1': 0,
                                'H95':0
                            }
                    file_dir=os.path.join(out_put_dir,exp,fold)
                    print(file_dir)
                    for file in os.listdir(file_dir):
                        mask=nib.load(os.path.join(soure_mask_dir,file)).get_fdata()
                        pred=nib.load(os.path.join(out_put_dir,exp,fold,file)).get_fdata()
                        result=compute_iter_metrics(pred, mask)
                        for key in fold_metric.keys():
                            if isinstance(result[key], numbers.Number):
                                fold_metric[key] += result[key]
                            else:
                                # 如果不是数值，则将其置为 0
                                fold_metric[key] += 0
                    for key in fold_metric.keys():
                        fold_metric[key]/=len(os.listdir(file_dir))
                        site_metric[key] += fold_metric[key]
                for key in site_metric.keys():
                    site_metric[key]/=5
                for key in exp_metric.keys():
                    if key in site_metric.keys():
                        exp_metric[key].append(site_metric[key])
            df=pd.DataFrame(exp_metric)
            df.to_csv(os.path.join(result_save_dir,exp,'statistics.csv'))

def statistics_nnUnet():
    pros_path='/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/dataset/wmh_segmentation/nnUNet_output'
    for exp in os.listdir(pros_path):
        site=exp.split('_')[-1]
        exp_metric={        'Dice': 0,
        'IoU': 0,
        'Precision': 0,
        'Recall': 0,
        'F1': 0,
        'H95':0}
        for file in os.listdir(os.path.join(pros_path,exp)):
            if file.find('.nii.gz')!=-1:
                mask_data=nib.load(os.path.join('/home/wy3atjlu/zhaozq/mount8t/subjects/multisite_medsam/dataset/wmh_segmentation/raw_data',file.replace('.nii.gz','_mask.nii.gz'))).get_fdata()
                pred_data=nib.load(os.path.join(pros_path, exp, file)).get_fdata()
                result=compute_iter_metrics(pred_data, mask_data)
                for key in result.keys():
                    exp_metric[key] += result[key]
        for key in exp_metric.keys():
            exp_metric[key]/=len(os.listdir(os.path.join(pros_path,exp)))
        print(site)
        print(exp_metric)
if __name__ == '__main__':
     # statistic('Pros')
    statistics_nnUnet()
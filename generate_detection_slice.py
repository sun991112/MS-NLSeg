
from dataloader import Whole_dataset,normalization,normalization_pair
import json
import SimpleITK as sitk
import numpy as np
import os
import pickle
import random

def training_slice_generate(train_set, path, num_augments, label_sum, new_slice_pos, image_size, single_axis=None):
    assert single_axis is not None, 'You should specify the axis to slice'

    if not os.path.exists(path):
        os.makedirs(path)
    print('Generating 2D slices...')

    num_augments = num_augments
    lable_sum = label_sum
    new_slice_pos = new_slice_pos
    num = len(train_set) // 3
    print("num", num)
    jishu = 0


    def _resolve_two_image_paths(sample):
        """Extract the paths of two images from sample:
    - Prefer using data_path1 / data_path2
    - Otherwise, use data_path if it is a list/tuple of length 2
        """
        if 'data_path1' in sample and 'data_path2' in sample:
            return sample['data_path1'], sample['data_path2']
        elif 'data_path' in sample:
            dp = sample['data_path']
            if isinstance(dp, (list, tuple)) and len(dp) == 2:
                return dp[0], dp[1]
            else:
                raise ValueError(f"'data_path' must be a list/tuple of length 2, got: {type(dp)} with value: {dp}")
        else:
            raise KeyError("Sample must contain 'data_path1' & 'data_path2' or a 2-element 'data_path'.")

    for data in train_set:
        img_path1, img_path2 = _resolve_two_image_paths(data)
        image1 = sitk.ReadImage(img_path1)
        image2 = sitk.ReadImage(img_path2)
        mask   = sitk.ReadImage(data['mask_path'])

        image_data1 = sitk.GetArrayFromImage(image1)  # [D,H,W]
        image_data2 = sitk.GetArrayFromImage(image2)  # [D,H,W]
        mask_data   = sitk.GetArrayFromImage(mask)    # [D,H,W]
        id = data['id'].split('.nii.gz')[0]
        
        print("jishu",jishu)

        if single_axis is False:
            axes_to_slice = [0, 1, 2]
        else:
            #axes_to_slice = [np.argmin(image_data1.shape)]
            axes_to_slice = [0]

        for axis in axes_to_slice:
            if axis == 0:
                for i in range(image_data1.shape[0]):
                    if mask_data[i, :, :].sum() > lable_sum:                 
                        #img1n, img2n = normalization_pair(image_data1[i, :, :], image_data2[i, :, :], image_size)  
                        # normed_img1 = img1n[np.newaxis, :, :]  # [1,H,W]
                        # normed_img2 = img2n[np.newaxis, :, :]  # [1,H,W]
                        normed_img1 = normalization(image_data1[i, :, :], 'image', image_size)[np.newaxis, :, :]#resize to 512*512   #add batch dimension
                        normed_img2 = normalization(image_data2[i, :, :], 'image', image_size)[np.newaxis, :, :]
                        normed_mask = normalization(mask_data[i,  :, :], 'mask',  image_size)[np.newaxis, :, :]

                        with open(os.path.join(path, f"{id}_{i}_0.pkl"), 'wb') as f:
                            pickle.dump({'2dimage1': normed_img1,
                                         '2dimage2': normed_img2,
                                         'mask':      normed_mask}, f)

                        for j in range(num_augments):
                            if random.random() < new_slice_pos * (0.9 ** j):
                                with open(os.path.join(path, f"{id}_{i}_aug{j}.pkl"), 'wb') as f:
                                    pickle.dump({'2dimage1': normed_img1,
                                                 '2dimage2': normed_img2,
                                                 'mask':      normed_mask}, f)
                    elif mask_data[i, :, :].sum() == lable_sum and jishu<num:
                        # img1n, img2n = normalization_pair(image_data1[i, :, :], image_data2[i, :, :], image_size)  # 各自是 [H,W]
                        # normed_img1 = img1n[np.newaxis, :, :]  # [1,H,W]
                        # normed_img2 = img2n[np.newaxis, :, :]  # [1,H,W]
                        normed_img1 = normalization(image_data1[i, :, :], 'image', image_size)[np.newaxis, :, :]  # resize to 512*512   #add batch dimension
                        normed_img2 = normalization(image_data2[i, :, :], 'image', image_size)[np.newaxis, :, :]
                        normed_mask = normalization(mask_data[i, :, :], 'mask', image_size)[np.newaxis, :, :]

                        with open(os.path.join(path, f"{id}_{i}_0.pkl"), 'wb') as f:
                            pickle.dump({'2dimage1': normed_img1,
                                         '2dimage2': normed_img2,
                                         'mask': normed_mask}, f)

            elif axis == 1:
                for i in range(image_data1.shape[1]):
                    if mask_data[:, i, :].sum() > lable_sum:
                        #normed_img1, normed_img2 = normalization_pair(image_data1[i, :, :], image_data2[i, :, :], image_size)[np.newaxis, :, :]
                        normed_img1 = normalization(image_data1[:, i, :], 'image', image_size)[np.newaxis, :, :]
                        normed_img2 = normalization(image_data2[:, i, :], 'image', image_size)[np.newaxis, :, :]
                        normed_mask = normalization(mask_data[:,  i, :], 'mask',  image_size)[np.newaxis, :, :]

                        with open(os.path.join(path, f"{id}_{i}_1.pkl"), 'wb') as f:
                            pickle.dump({'2dimage1': normed_img1,
                                         '2dimage2': normed_img2,
                                         'mask':      normed_mask}, f)

                        for j in range(num_augments):
                            if random.random() < new_slice_pos * (0.9 ** j):
                                with open(os.path.join(path, f"{id}_{i}_aug{j}.pkl"), 'wb') as f:
                                    pickle.dump({'2dimage1': normed_img1,
                                                 '2dimage2': normed_img2,
                                                 'mask':      normed_mask}, f)

            elif axis == 2:
                for i in range(image_data1.shape[2]):
                    if mask_data[:, :, i].sum() > lable_sum:
                        #normed_img1, normed_img2 = normalization_pair(image_data1[i, :, :], image_data2[i, :, :], image_size)[np.newaxis, :, :]
                        normed_img1 = normalization(image_data1[:, :, i], 'image', image_size)[np.newaxis, :, :]
                        normed_img2 = normalization(image_data2[:, :, i], 'image', image_size)[np.newaxis, :, :]
                        normed_mask = normalization(mask_data[:,  :, i], 'mask',  image_size)[np.newaxis, :, :]

                        with open(os.path.join(path, f"{id}_{i}_2.pkl"), 'wb') as f:
                            pickle.dump({'2dimage1': normed_img1,
                                         '2dimage2': normed_img2,
                                         'mask':      normed_mask}, f)

                        for j in range(num_augments):
                            if random.random() < new_slice_pos * (0.9 ** j):
                                with open(os.path.join(path, f"{id}_{i}_aug{j}.pkl"), 'wb') as f:
                                    pickle.dump({'2dimage1': normed_img1,
                                                 '2dimage2': normed_img2,
                                                 'mask':      normed_mask}, f)
        jishu=jishu+1

if __name__ == '__main__':
    import json
    import pandas as pd

    path='/home/generate_slice_config.json'
    with open(path, 'r') as f:
        data = json.load(f)
    random.seed(data['seed'])
    # print(type(data))
    loader=Whole_dataset(dataset_path=data['dataset_path'],sites=data['sites'],task=data['task'])
    training_slice_generate(loader,data['output_path'],data['num_augments'],data['label_sum'],data['new_slice_pos'],data['image_size'],data['single_axis'])

    data_list = list(data.items())
    df = pd.DataFrame(data_list, columns=["key", "value"])
    df.to_csv(os.path.join(data['output_path'],'config.csv'),index=False, encoding="utf-8")
    # pd.DataFrame(data).to_csv(os.path.join(data['output_path'],'config.csv'),index=False)

    # loader=Whole_dataset()

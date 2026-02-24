import os
import pickle
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import pickle
import re
from torch.utils.data import DataLoader
import numpy as np
import os
import SimpleITK as sitk
import pickle
import torch
import torchvision.transforms as T
import random
import torch.nn.functional as F
import nibabel as nib


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class GammaCorrection(object):
    def __init__(self, gamma_range=(0.7, 1.5)):
        self.gamma = random.uniform(*gamma_range)

    def __call__(self, tensor):
        return torch.pow(tensor, self.gamma)

def normalization(image,type,size=512):
    if type == 'image':
        mask = image > 0

        if not np.any(mask):
            return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)

        y = image[mask]
        lower = np.percentile(y, 0.2)
        upper = np.percentile(y, 99.8)
        image[mask & (image < lower)] = lower
        image[mask & (image > upper)] = upper
        image = (image - image.mean()) / (image.std())

        resized_image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        padded_image=resized_image

    if type == 'mask':
        resized_image = cv2.resize(image, (size,size), interpolation=cv2.INTER_NEAREST)
        padded_image=resized_image

    return padded_image

def normalization_pair(image1, image2, size=512):
    """
Perform "joint normalization" on two images:
  1) Use the combined valid region of both images (image1 > 0 OR image2 > 0) to compute the 0.2/99.8 percentiles and apply unified clipping;
  2) Use the overall mean and standard deviation of the combined valid region to perform z-score normalization (both images use the same mu/std);
  3) Resize each image to size × size.
Return: img1_norm, img2_norm
    """
    img1 = image1.copy()
    img2 = image2.copy()

    mask1 = img1 > 0
    mask2 = img2 > 0
    joint_mask = mask1 | mask2

    if not np.any(joint_mask):
        return (
            cv2.resize(img1, (size, size), interpolation=cv2.INTER_LINEAR),
            cv2.resize(img2, (size, size), interpolation=cv2.INTER_LINEAR),
        )

    y_joint = np.concatenate([img1[joint_mask], img2[joint_mask]]).astype(np.float32)
    lower = np.percentile(y_joint, 0.2)
    upper = np.percentile(y_joint, 99.8)

    img1[mask1 & (img1 < lower)] = lower
    img1[mask1 & (img1 > upper)] = upper
    img2[mask2 & (img2 < lower)] = lower
    img2[mask2 & (img2 > upper)] = upper

    #mu/std
    mu = y_joint.mean()
    sigma = y_joint.std()
    sigma = sigma if sigma > 1e-8 else 1.0

    img1 = (img1 - mu) / sigma
    img2 = (img2 - mu) / sigma

    # 3) resize
    img1 = cv2.resize(img1, (size, size), interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, (size, size), interpolation=cv2.INTER_LINEAR)

    return img1, img2

class Whole_dataset(Dataset):
    def __init__(self, dataset_path,sites,task):
        super().__init__()
        self.data_paths=[]
        self.allowed_sites = {
            'WMH': ['trainingmiccai21','testingmiccai21','LesSeg','PediMS','openms']}
        print('Whole dataset loading data...')
        print(
            'Loading data need 2 arguments, which are: data_path (absolute path to the folder contains all files in this dataset),site(define which part of data are needed to be read)')
        assert task in ['WMH'], 'Task should be WMH'
        
        if task == 'WMH':
            modality = 'flair'
            all_files = set(os.listdir(dataset_path))  

            for site in sites:
                site_pat = re.compile(r'(?:^|[^a-zA-Z0-9])' + re.escape(site) + r'(?:[^a-zA-Z0-9]|$)')

                for file in all_files:
                    if not site_pat.search(file):
                        continue
                    if 'timepoints1' not in file:
                        continue
                    if modality not in file or not file.endswith('.nii.gz'):
                        continue

                    img1_file = file
                    img2_file = file.replace('timepoints1', 'timepoints2')

                    if img2_file not in all_files:
                        continue

                    base_no_tail = re.sub(rf'([_-]?timepoints1)?[_-]?{re.escape(modality)}\.nii\.gz$', '', img1_file)
                    mask_file = base_no_tail + '_mask.nii.gz'

                    if mask_file not in all_files:
                        alt_mask = img1_file.replace(modality, 'mask')
                        if alt_mask in all_files:
                            mask_file = alt_mask
                        else:
                            continue  #not find mask，skip

                    image_path1 = os.path.join(dataset_path, img1_file)
                    image_path2 = os.path.join(dataset_path, img2_file)
                    mask_path = os.path.join(dataset_path, mask_file)

                    self.data_paths.append({
                        'data_path1': image_path1,
                        'data_path2': image_path2,
                        'mask_path': mask_path,
                        'id': file
                    })

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        return self.data_paths[idx]

def post_process_training_mask(padded_image, original_size):
    """
    :param padded_image: A tensor of shape (batch_size, channels, height, width)
    :param original_size: A tuple (new_height, new_width) indicating the desired output size
    :return: A tensor of the resized images
    """
    result = F.interpolate(padded_image, size=original_size, mode='bilinear', align_corners=False)
    return result
def post_process_mask(padded_image,original_size):

    '''
    :param padded_image: size (b,w,h)
    :param original_size: size(w~,h~)
    :param size: padded_size
    :return:
    '''

    h, w = original_size

    batchsize=padded_image.shape[0]
    unpadded_image=np.transpose(padded_image,(1,2,0))
    restored_image = cv2.resize(unpadded_image, (w, h), interpolation=cv2.INTER_LINEAR)

    if batchsize ==1:
        restored_image=np.expand_dims(restored_image,axis=0)
    else:
        restored_image=np.transpose(restored_image, (2, 0, 1))

    return restored_image

class slice_dataset(Dataset):
    def __init__(self,slice_path):
        super().__init__()
        print('slice dataset loading 2d slices...')
        assert len(os.listdir(slice_path)) > 0, 'you should generate training slices first'
        self.slices=[]
        for file in os.listdir(slice_path):
            if file.endswith('.pkl'):
                with open(os.path.join(slice_path,file), 'rb') as f:
                    data=pickle.load(f)
                    self.slices.append(data)
                    mask=data['mask']
                    print(mask[(mask != 0) & (mask != 1)]) if ((mask != 0) & (mask != 1)).any() else None

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        return self.slices[idx]

class test_dataset(Dataset):
    def __init__(self,dataset_iterable,single_axis=None):
        assert single_axis is not None, 'You should specify the axis to slice'
        self.single_axis = single_axis
        super().__init__()
        self.data_set=dataset_iterable

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):

        data=self.data_set[idx]
        image1=sitk.ReadImage(data['data_path1'])
        image2 = sitk.ReadImage(data['data_path2'])
        mask_image=sitk.ReadImage(data['mask_path'])
        slices_this_patient = {}
        image_data1=sitk.GetArrayFromImage(image1)
        image_data2 = sitk.GetArrayFromImage(image2)

        if self.single_axis is False:
            axes_to_slice = [0, 1, 2]  
        else:
            axes_to_slice = [0]
            
        slices_this_patient['affine'] = (nib.load(data['data_path1'])).affine
        slices_this_patient['brain_mask'] = (image_data1 != 0) & (image_data2 != 0)
        slices_this_patient['id'] = data['id']
        slices_this_patient['info'] = [image1.GetOrigin(), image1.GetSpacing(),image1.GetDirection()]
        slices_this_patient['mask'] = sitk.GetArrayFromImage(mask_image)
        slices_this_patient['sag_slices1'] = []
        slices_this_patient['cor_slices1'] = []
        slices_this_patient['axi_slices1'] = []
        slices_this_patient['sag_slices2'] = []
        slices_this_patient['cor_slices2'] = []
        slices_this_patient['axi_slices2'] = []
        slices_this_patient['sag_size'] = image_data1[0, :, :].shape[:2]
        slices_this_patient['cor_size'] = image_data1[:, 0, :].shape[:2]
        slices_this_patient['axi_size'] = image_data1[:, :, 0].shape[:2]
        for axis in axes_to_slice:
            if axis==0:
                for i in range(image_data1.shape[0]):
                    nonzero_mask = (image_data1 != 0) & (image_data2 != 0)  # bool mask, shape [B,1,H,W]

                    image_data1 = image_data1 * nonzero_mask
                    image_data2 = image_data2 * nonzero_mask

                    normed_data1 = normalization(image_data1[i, :, :], 'image')
                    #normed_data1,normed_data2=normalization_pair(image_data1[i, :, :], image_data2[i, :, :])
                    normed_data1 = np.repeat(normed_data1[np.newaxis, :, :], 3, axis=0)
                    slices_this_patient['sag_slices1'].append(normed_data1)

                    normed_data2 = normalization(image_data2[i, :, :], 'image')
                    normed_data2 = np.repeat(normed_data2[np.newaxis, :, :], 3, axis=0)
                    slices_this_patient['sag_slices2'].append(normed_data2)
            if axis==1:
                for i in range(image_data1.shape[1]):
                    nonzero_mask = (image_data1 != 0) & (image_data2 != 0)  # bool mask, shape [B,1,H,W]

                    image_data1 = image_data1 * nonzero_mask
                    image_data2 = image_data2 * nonzero_mask
                    #normed_data1, normed_data2 = normalization_pair(image_data1[i, :, :], image_data2[i, :, :])
                    normed_data1 = normalization(image_data1[:, i, :], 'image')
                    normed_data1 = np.repeat(normed_data1[np.newaxis, :, :], 3, axis=0)
                    slices_this_patient['cor_slices1'].append(normed_data1)

                    normed_data2 = normalization(image_data2[:, i, :], 'image')
                    normed_data2 = np.repeat(normed_data2[np.newaxis, :, :], 3, axis=0)
                    slices_this_patient['cor_slices2'].append(normed_data2)
            if axis==2:
                for i in range(image_data1.shape[2]):
                    nonzero_mask = (image_data1 != 0) & (image_data2 != 0)  # bool mask, shape [B,1,H,W]

                    image_data1 = image_data1 * nonzero_mask
                    image_data2 = image_data2 * nonzero_mask
                    #normed_data1, normed_data2 = normalization_pair(image_data1[i, :, :], image_data2[i, :, :])
                    normed_data1 = normalization(image_data1[:, :, i], 'image')
                    normed_data1 = np.repeat(normed_data1[np.newaxis, :, :], 3, axis=0)
                    slices_this_patient['axi_slices1'].append(normed_data1)

                    normed_data2 = normalization(image_data2[:, :, i], 'image')
                    normed_data2 = np.repeat(normed_data2[np.newaxis, :, :], 3, axis=0)
                    slices_this_patient['axi_slices2'].append(normed_data2)
        return slices_this_patient


class test_dataloader:
    def __init__(self, dataset_iterable, model, device,
                 max_batch_size=64, min_batch_size=1,
                 single_axis=None, model_name=None):
        assert single_axis is not None, 'You should specify the axis to slice'
        assert model_name is not None, 'You should specify the model name'
        self.single_axis = single_axis
        self.dataset = dataset_iterable         
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.cached_batch_sizes = {}             # {'sag': B, 'cor': B, 'axi': B}
        self.model_name = model_name

    def estimate_GPU_memory_pair(self, in1, in2):
        in1, in2 = in1.to(self.device), in2.to(self.device)
        model = self.model.to(self.device)
        try:
            with torch.no_grad():
                if self.model_name in ("UNet"):
                    _ = model(in1, in2)
                elif self.model_name in ("DAPSAM"):
                    _= model(in1, False, 512)
                elif self.model_name in ("TriD"):
                    _ = model(in1, phase='train')
                else:
                    _ = model(in1)  
            torch.cuda.empty_cache()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            return False

    def find_batch_size(self, sample1, sample2, axis):
        if axis in self.cached_batch_sizes:
            return self.cached_batch_sizes[axis]

        x1 = torch.from_numpy(sample1).float().to(self.device)  # [3,H,W]
        x2 = torch.from_numpy(sample2).float().to(self.device)

        low, high = self.min_batch_size, self.max_batch_size
        while low <= high:
            mid = (low + high) // 2
            in1 = torch.stack([x1] * mid, dim=0)  # [B,3,H,W]
            in2 = torch.stack([x2] * mid, dim=0)
            #ok = self.estimate_GPU_memory_pair(in1, in2)
            ok=True
            if ok:
                low = mid + 1
            else:
                high = mid - 1
        optimal = max(1, high)
        #self.cached_batch_sizes[axis] = optimal
        self.cached_batch_sizes[axis] = 32
        return optimal

    def precompute_batch_sizes(self):
        try:
            n = len(self.dataset)
            for idx in range(n):
                try:
                    data = self.dataset[idx]  
                except Exception as e:
                    import traceback as tb
                    tb.print_exc()
                    raise  

                if data is None:
                    print(f"[DBG] precompute: idx={idx} got None sample, skip")
                    continue
                if data.get('sag_slices1') and data.get('sag_slices2'):
                    self.find_batch_size(data['sag_slices1'][0], data['sag_slices2'][0], axis='sag')
                if data.get('cor_slices1') and data.get('cor_slices2'):
                    self.find_batch_size(data['cor_slices1'][0], data['cor_slices2'][0], axis='cor')
                if data.get('axi_slices1') and data.get('axi_slices2'):
                    self.find_batch_size(data['axi_slices1'][0], data['axi_slices2'][0], axis='axi')
                break  
                
        except Exception as e:
            import traceback as tb
            print("[ERR] precompute_batch_sizes crashed:", repr(e))
            tb.print_exc()
            raise

    def __iter__(self):
        if not self.cached_batch_sizes:
            self.precompute_batch_sizes()

        if not self.cached_batch_sizes:
            self.cached_batch_sizes = {'sag': 1, 'cor': 1, 'axi': 1}
        try:
            yielded = 0
            for data in self.dataset:
                if data is None:
                    print("[DBG] skip None sample from dataset")
                    continue

                sag_loader1 = sag_loader2 = None
                cor_loader1 = cor_loader2 = None
                axi_loader1 = axi_loader2 = None

                if self.single_axis:
                    if data.get('sag_slices1') and data.get('sag_slices2'):

                        bs = self.cached_batch_sizes.get('sag', 1)
                        sag_loader1 = DataLoader(data['sag_slices1'], batch_size=bs, shuffle=False)
                        sag_loader2 = DataLoader(data['sag_slices2'], batch_size=bs, shuffle=False)
                    elif data.get('cor_slices1') and data.get('cor_slices2'):

                        bs = self.cached_batch_sizes.get('cor', 1)
                        cor_loader1 = DataLoader(data['cor_slices1'], batch_size=bs, shuffle=False)
                        cor_loader2 = DataLoader(data['cor_slices2'], batch_size=bs, shuffle=False)
                    elif data.get('axi_slices1') and data.get('axi_slices2'):

                        bs = self.cached_batch_sizes.get('axi', 1)
                        axi_loader1 = DataLoader(data['axi_slices1'], batch_size=bs, shuffle=False)
                        axi_loader2 = DataLoader(data['axi_slices2'], batch_size=bs, shuffle=False)
                    else:
                        print("[WARN] single_axis=True but no slices; skip sample id:", data.get('id'))
                        continue
                else:
                    if data.get('sag_slices1') and data.get('sag_slices2'):
                        bs = self.cached_batch_sizes.get('sag', 1)
                        sag_loader1 = DataLoader(data['sag_slices1'], batch_size=bs, shuffle=False)
                        sag_loader2 = DataLoader(data['sag_slices2'], batch_size=bs, shuffle=False)
                    if data.get('cor_slices1') and data.get('cor_slices2'):
                        bs = self.cached_batch_sizes.get('cor', 1)
                        cor_loader1 = DataLoader(data['cor_slices1'], batch_size=bs, shuffle=False)
                        cor_loader2 = DataLoader(data['cor_slices2'], batch_size=bs, shuffle=False)
                    if data.get('axi_slices1') and data.get('axi_slices2'):
                        bs = self.cached_batch_sizes.get('axi', 1)
                        axi_loader1 = DataLoader(data['axi_slices1'], batch_size=bs, shuffle=False)
                        axi_loader2 = DataLoader(data['axi_slices2'], batch_size=bs, shuffle=False)

                out = {
                    'axi_ori_size': data.get('axi_size'),
                    'cor_ori_size': data.get('cor_size'),
                    'sag_ori_size': data.get('sag_size'),
                    'axi_len': len(data.get('axi_slices1', [])),
                    'sag_len': len(data.get('sag_slices1', [])),
                    'cor_len': len(data.get('cor_slices1', [])),
                    'id': data.get('id'),
                    'info': data.get('info'),
                    'mask': data.get('mask'),
                    'affine': data.get('affine'),
                    'sagittal_loader1': sag_loader1,
                    'sagittal_loader2': sag_loader2,
                    'coronal_loader1': cor_loader1,
                    'coronal_loader2': cor_loader2,
                    'axial_loader1': axi_loader1,
                    'axial_loader2': axi_loader2,
                    'brain_mask': data.get('brain_mask'),
                }
                yielded += 1
                # print(f"[DBG] __iter__: yield sample#{yielded} id={out['id']}, "
                #       f"lens(sag,cor,axi)={(out['sag_len'], out['cor_len'], out['axi_len'])}")
                yield out

        except Exception as e:
            import traceback as tb
            print("[ERR] __iter__ crashed:", repr(e))
            tb.print_exc()
            raise

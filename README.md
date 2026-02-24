# MS-NLSeg  
**A Multi-Center Dataset and Benchmark for Multiple Sclerosis New Lesion Segmentation**

---

## Introduction

This repository is for our **MICCAI 2026** paper:

> **MS-NLSeg: A Multi-Center Dataset and Benchmark for Multiple Sclerosis New Lesion Segmentation**

MS-NLSeg is a multi-center dataset designed for multiple sclerosis (MS) new lesion segmentation.  
It contains **108 annotated paired longitudinal MRI scans** collected from **five imaging centers**.

---

## Dataset Overview

The dataset consists of five sub-centers:

| Center        | Source                                      | Access                  |
|--------------|---------------------------------------------|-------------------------|
| Miccai_Ing   | MSSEG-II Challenge (2021)                   | [Application required](https://portal.fli-iam.irisa.fr/msseg-2/) |
| Miccai_Mix   | MSSEG-II Challenge (2021)                   | [Application required](https://portal.fli-iam.irisa.fr/msseg-2/) |
| MSLesSeg     | Guarnera et al., Scientific Data 2025       | Processed version included |
| PediMS       | Popa et al., Scientific Data 2025           | Processed version included |
| OpenMS       | Lesjak et al., Neuroinformatics 2016        | Processed version included |

---

## Annotation Strategy

For the **Miccai_Ing** and **Miccai_Mix** centers, new lesion annotations are directly provided by the original MSSEG-II Challenge dataset.

For the remaining three centers (**MSLesSeg**, **PediMS**, and **OpenMS**), new lesion masks were generated using the **Automatic Annotation Generation (AAG) pipeline** proposed in our paper. The implementation of the AAG pipeline is available in [`AAG.py`](./AAG.py).

Processed images and annotations are released under the `dataset/` directory (subject to the original dataset licenses).

---

## Experimental Benchmark


### Training

Model training is performed using `ssdg_train.py`. The general command structure is:

```bash
python ssdg_train.py \
    --checkpoint_dir /path/to/checkpoints \
    --exp_name <experiment_name> \
    --task <task_name> \
    --training_slice_path /path/to/training_slices \
    --test_site <site_list> \
    --gpu_ids <gpu_id> \
    --raw_data_folder /path/to/dataset_root \
    --model <model_name> \
    --single_axis \
    --seed <random_seed> \
    --epochs <num_epochs> \
    --augment \
    --eval
```

### Testing

```bash
python test.py
```

---


## References

[1] Guarnera, Francesco, et al. “MSLesSeg: Baseline and benchmarking of a new Multiple Sclerosis Lesion Segmentation dataset.” *Scientific Data*, vol. 12, no. 1, 2025, p. 920.

[2] Lesjak, Žiga, et al. “Validation of white-matter lesion change detection methods on a novel publicly available MRI image database.” *Neuroinformatics*, vol. 14, no. 4, 2016, pp. 403–420.

[3] Popa, Maria, Gabriela Adriana Vișa, and Ciprian Radu Șofariu. “PediMS: A pediatric multiple sclerosis lesion segmentation dataset.” *Scientific Data*, vol. 12, no. 1, 2025, p. 1184.

---

## Acknowledgement

We sincerely acknowledge and thank the authors of the publicly available datasets used in this work. In particular, we are grateful to the contributors of the MSLesSeg dataset, the PediMS dataset, and the white-matter lesion database introduced by Lesjak et al. Their significant efforts in data collection, curation, validation, and public release have greatly facilitated research on multiple sclerosis lesion segmentation and lesion change detection. We deeply appreciate their commitment to open science and their valuable contributions to the neuroimaging community.

---

## License

Please follow the original dataset licenses.  
Released for research purposes only.

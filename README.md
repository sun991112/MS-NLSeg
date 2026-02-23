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

## Citation
11.	[6] Guarnera, Francesco, et al. "MSLesSeg: baseline and benchmarking of a new Multiple Sclerosis Lesion Segmentation dataset." Scientific Data 12.1 (2025): 920.

13.	[8] Popa, Maria, Gabriela Adriana Vișa, and Ciprian Radu Șofariu. "PediMS: A pediatric multiple sclerosis lesion segmentation dataset." Scientific Data 12.1 (2025): 1184.
	12.	[7] Lesjak, Žiga, et al. "Validation of white-matter lesion change detection methods on a novel publicly available MRI image database." Neuroinformatics 14.4 (2016): 403-420.
通过论文中的an Automatic Annotation Generation (AAG) pipeline to generate reliable masks without additional manual effort.并且发布于dataset文件夹中。
```bibtex
@inproceedings{msnlseg2026,
  title={MS-NLSeg: A Multi-Center Dataset and Benchmark for Multiple Sclerosis New Lesion Segmentation},
  author={Anonymous},
  booktitle={MICCAI},
  year={2026}
}
```

---

## License

Please follow the original dataset licenses.  
Released for research purposes only.

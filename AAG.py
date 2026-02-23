# -*- coding: utf-8 -*-
"""
OpenMS diff-mask pipeline (all-in-one)
=====================================
规则（你确认的）：
- 分组 ID / 保存前缀：group_id = parts[0] + "_" + parts[1]  (例如 openms_XXX)
- 以 group_id 为单位把 t0/t1/diff 对齐，然后顺序执行：
  Step1  : Top-percent 差异筛选（需要 t0/t1/diff）
  Step2  : 受 diff 约束的膨胀（固定半径，seed=Step1 输出）
  Step3  : 逐 slice 2D 小连通域清理（Step2 输出）
  Step4  : 自适应 2D 膨胀（seed=Step3 输出，约束=diff）
  Step5  : 逐 slice 2D 小连通域清理 + 填洞（Step4 输出）

用法示例：
python openms_pipeline.py --root_dir "D:\dataset\wmh\peizhun_make_mask\mnipeizhun\mni_brain\openms_mni_brain"

依赖：
pip install numpy nibabel scipy SimpleITK tqdm
"""

import os
import argparse
from collections import defaultdict
from scipy.ndimage import binary_fill_holes
import numpy as np
import nibabel as nib
from scipy import ndimage
import SimpleITK as sitk
from tqdm import tqdm


# =========================
# 0) 文件名解析 / 分组
# =========================
def parse_group_id(fname: str) -> str | None:
    """openms_XXX_YYYY_0000.nii.gz -> openms_XXX"""
    parts = fname.split("_")
    if len(parts) < 2:
        return None
    return f"{parts[0]}_{parts[1]}"


def list_nii(root_dir: str):
    return [
        f for f in os.listdir(root_dir)
        if f.lower().endswith((".nii", ".nii.gz"))
    ]


def build_case_index(root_dir: str):
    """
    在 root_dir 下扫描并按 group_id 建索引：
    - t0: 含 "0000" 且不含 "mask" 且不含 "diff"
    - t1: 含 "0001" 且不含 "mask" 且不含 "diff"
    - diff: 文件名包含 "diff"（你原脚本是 "diff.nii.gz" / "diff" 关键词）
    """
    files = list_nii(root_dir)
    groups = defaultdict(dict)

    for f in files:
        gid = parse_group_id(f)
        if gid is None:
            continue

        fl = f.lower()

        # diff（尽量稳健：包含 diff 且不是各种 grow mask）
        # 如果你有更严格命名，比如固定 "diff.nii.gz"，可再收紧
        if "diff" in fl and ("mask_pseudo_new_grow" not in fl) and ("adaptive" not in fl):
            # 更偏向真正 diff：含 diff 且不含 0000/0001 的也行
            groups[gid]["diff"] = f
            continue

        # t0 / t1
        if ("0000" in fl) and ("mask" not in fl) and ("diff" not in fl):
            groups[gid]["t0"] = f
        elif ("0001" in fl) and ("mask" not in fl) and ("diff" not in fl):
            groups[gid]["t1"] = f

    return groups


# =========================
# 1) Step1: top-percent 差异筛选
# =========================
def robust_zscore(image: np.ndarray, mask: np.ndarray, p_low=0.2, p_high=99.8):
    img = image.copy()
    y = img[mask > 0]
    if y.size == 0:
        raise ValueError("Mask 内无有效体素（brain mask 为空）")

    lo = np.percentile(y, p_low)
    hi = np.percentile(y, p_high)

    img[(mask > 0) & (img < lo)] = lo
    img[(mask > 0) & (img > hi)] = hi

    mu = img[mask > 0].mean()
    sigma = img[mask > 0].std() + 1e-8
    return (img - mu) / sigma


def step1_process_case_top_percent(
    img0_path: str,
    img1_path: str,
    diff_mask_path: str,
    top_percent: float = 0.05,
    min_keep: int = 1,
    min_cc_size: int = 10,
    min_cc_p90: float = 0.5,
):
    """
    输出：3D uint8 mask (0/1)，ref_nii 用于 affine/header
    关键：逐 z-slice 做 2D 8联通连通域，在 diff_mask 内对 D=I1-I0 做 top-percent
    """
    print(img0_path)
    img0_nii = nib.load(img0_path)
    img1_nii = nib.load(img1_path)
    diff_nii = nib.load(diff_mask_path)

    I0 = img0_nii.get_fdata().astype(np.float32)
    I1 = img1_nii.get_fdata().astype(np.float32)
    diff_mask = (diff_nii.get_fdata() > 0)

    # brain mask
    brain_mask = (I0 != 0) | (I1 != 0)

    I0_n = robust_zscore(I0, brain_mask)
    I1_n = robust_zscore(I1, brain_mask)

    D = I1_n - I0_n
    D_roi = D * diff_mask

    output = np.zeros_like(diff_mask, dtype=np.uint8)
    output_all = np.zeros_like(diff_mask, dtype=np.uint8)

    structure_2d = np.ones((3, 3), dtype=np.int8)  # 8 联通
    z_slices = diff_mask.shape[2]

    total_cc = 0
    kept_cc = 0

    for z in range(z_slices):
        mask_z = diff_mask[:, :, z]
        if mask_z.sum() == 0:
            continue

        labeled_z, num_cc = ndimage.label(mask_z, structure=structure_2d)
        total_cc += num_cc

        D_z = D_roi[:, :, z]

        for cc_id in range(1, num_cc + 1):
            cc_mask_raw = (labeled_z == cc_id)

            # 🔴 新增：对单个连通域做 2D 孔洞填充
            cc_mask = binary_fill_holes(cc_mask_raw)

            cc_size = int(cc_mask.sum())

            # 规则1：体积过滤
            if cc_size < min_cc_size:
                continue

            # ---------- 规则1.5：外接矩形覆盖率（shape filter） ----------
            ys, xs = np.where(cc_mask)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
            fill_ratio = cc_size / max(bbox_area, 1)

            # if fill_ratio < 0.5:
            #     print(fill_ratio)
            #     continue

            values = D_z[cc_mask]
            values = values[values > 0]
            if values.size == 0:
                continue

            # 规则2：整体变化强度
            p90 = float(np.percentile(values, 90))

            if p90 <1 and fill_ratio < 0.5:
                print(z,"p90 <1 and fill_ratio < 0.5")
                continue
            if p90 < min_cc_p90 :
                print(z,"p90 < min_cc_p90")
                continue

            # 🔴 修改点：保留整个连通域，而不是 top-percent 点
            output_all[:, :, z][cc_mask] = 1

            # top-percent
            k = max(int(np.ceil(top_percent * values.size)), min_keep)
            thresh = np.partition(values, -k)[-k]
            output[:, :, z][cc_mask & (D_z >= thresh)] = 1
            kept_cc += 1

    return output, output_all,img0_nii, {"total_cc_2d": total_cc, "kept_cc_2d": kept_cc}


def run_step1(root_dir: str, out_dir: str, groups: dict,
              top_percent=0.05, min_keep=1, min_cc_size=10, min_cc_p90=0.5):
    os.makedirs(out_dir, exist_ok=True)

    for gid, items in tqdm(sorted(groups.items()), desc="Step1: top-percent"):
        if not all(k in items for k in ["t0", "t1", "diff"]):
            print(f"[Step1 Skip] {gid}: missing t0/t1/diff")
            continue

        t0_path = os.path.join(root_dir, items["t0"])
        t1_path = os.path.join(root_dir, items["t1"])
        diff_path = os.path.join(root_dir, items["diff"])

        try:
            out_mask, output_all,ref_nii, stat = step1_process_case_top_percent(
                t0_path, t1_path, diff_path,
                top_percent=top_percent,
                min_keep=min_keep,
                min_cc_size=min_cc_size,
                min_cc_p90=min_cc_p90,
            )
        except Exception as e:
            print(f"[Step1 Error] {gid}: {e}")
            continue

        out_nii = nib.Nifti1Image(out_mask.astype(np.uint8), ref_nii.affine, ref_nii.header)
        output_all_nii = nib.Nifti1Image(output_all.astype(np.uint8), ref_nii.affine, ref_nii.header)
        out_name = f"{gid}_diff_lesion_top5p.nii.gz"
        output_all_name = f"{gid}_all.nii.gz"
        nib.save(out_nii, os.path.join(out_dir, out_name))
        nib.save(output_all_nii , os.path.join(out_dir, output_all_name))

# =========================
# ===== NEW (Step1.5) =====
# =========================
def remove_small_cc_per_slice_2d_seed(mask3d: sitk.Image,
                                      min_size=2,
                                      fully_connected=True) -> sitk.Image:
    """
    专用于 seed 的 2D 清理：
    - 逐 slice
    - 8 联通
    - 连通域 < min_size 删除
    """
    mask3d = sitk.Cast(mask3d > 0, sitk.sitkUInt8)
    size = list(mask3d.GetSize())  # (x, y, z)

    extractor = sitk.ExtractImageFilter()
    extractor.SetSize([size[0], size[1], 0])

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(bool(fully_connected))  # True = 8 联通

    relabel = sitk.RelabelComponentImageFilter()
    relabel.SetMinimumObjectSize(int(min_size))

    cleaned_slices = []
    for z in range(size[2]):
        extractor.SetIndex([0, 0, z])
        sl = extractor.Execute(mask3d)

        cc = cc_filter.Execute(sl)
        cc = relabel.Execute(cc)
        sl_clean = sitk.Cast(cc > 0, sitk.sitkUInt8)

        cleaned_slices.append(sl_clean)

    out3d = sitk.JoinSeries(cleaned_slices)
    out3d = sitk.Cast(out3d > 0, sitk.sitkUInt8)
    out3d.CopyInformation(mask3d)
    return out3d

# =========================
# SITK 工具（Step2-5 共用）
# =========================
def sitk_read_bin(path: str) -> sitk.Image:
    img = sitk.ReadImage(path)
    return sitk.Cast(img > 0, sitk.sitkUInt8)


def sitk_voxel_sum(img: sitk.Image) -> int:
    return int(np.sum(sitk.GetArrayViewFromImage(img)))


# =========================
# 2) Step2: 受 diff 约束的固定半径膨胀（seed=Step1 输出）
# =========================
def run_step2_fixed_grow(step1_dir: str, root_dir: str, out_dir: str, groups: dict,
                         radius=(1, 1, 0), max_iter=1):
    os.makedirs(out_dir, exist_ok=True)

    for gid, items in tqdm(sorted(groups.items()), desc="Step2: fixed grow"):
        if "diff" not in items:
            print(f"[Step2 Skip] {gid}: missing diff")
            continue

        seed_path = os.path.join(step1_dir, f"{gid}_diff_lesion_top5p.nii.gz")
        if not os.path.exists(seed_path):
            print(f"[Step2 Skip] {gid}: missing seed from step1 -> {os.path.basename(seed_path)}")
            continue

        diff_path = os.path.join(root_dir, items["diff"])
        if not os.path.exists(diff_path):
            print(f"[Step2 Skip] {gid}: diff file not found on disk")
            continue

        seed = sitk_read_bin(seed_path)
        diff = sitk_read_bin(diff_path)

        # 空间一致
        seed.CopyInformation(diff)

        # ===== Step1.5：2D 八连通，小于 3 的 seed 删除 =====
        seed = remove_small_cc_per_slice_2d_seed(
            seed,
            min_size=3,
            fully_connected=True
        )

        diff_sum = sitk_voxel_sum(diff)
        seed_sum = sitk_voxel_sum(seed)

        if diff_sum == 0 or seed_sum == 0:
            out = sitk.Image(diff.GetSize(), sitk.sitkUInt8)
            out.CopyInformation(diff)
        else:
            out = sitk.And(seed, diff)
            out.CopyInformation(diff)

            prev_sum = sitk_voxel_sum(out)
            for _ in range(int(max_iter)):
                grown = sitk.BinaryDilate(out, list(radius))
                grown = sitk.And(grown, diff)
                cur_sum = sitk_voxel_sum(grown)
                if cur_sum == prev_sum:
                    break
                out = grown
                prev_sum = cur_sum

        out_name = f"{gid}_mask_pseudo_new_grow.nii.gz"
        sitk.WriteImage(out, os.path.join(out_dir, out_name))


# =========================
# 3) Step3: 逐 slice 2D 小连通域清理
# =========================
def remove_small_cc_per_slice_2d(mask3d: sitk.Image, min_size=5, fully_connected=True) -> sitk.Image:
    mask3d = sitk.Cast(mask3d > 0, sitk.sitkUInt8)
    size = list(mask3d.GetSize())  # (x, y, z)

    extractor = sitk.ExtractImageFilter()
    extractor.SetSize([size[0], size[1], 0])

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(bool(fully_connected))

    relabel = sitk.RelabelComponentImageFilter()
    relabel.SetMinimumObjectSize(int(min_size))

    cleaned_slices = []
    for z in range(size[2]):
        extractor.SetIndex([0, 0, z])
        sl = extractor.Execute(mask3d)
        cc = cc_filter.Execute(sl)
        cc = relabel.Execute(cc)
        sl_clean = sitk.Cast(cc > 0, sitk.sitkUInt8)
        cleaned_slices.append(sl_clean)

    out3d = sitk.JoinSeries(cleaned_slices)
    out3d = sitk.Cast(out3d > 0, sitk.sitkUInt8)
    out3d.CopyInformation(mask3d)
    return out3d


def run_step3_clean(step2_dir: str, out_dir: str, min_cc_size_2d=5, fully_connected=True):
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in list_nii(step2_dir) if f.lower().endswith(".nii.gz")]
    for fname in tqdm(sorted(files), desc="Step3: clean small CC"):
        in_path = os.path.join(step2_dir, fname)
        img = sitk.ReadImage(in_path, sitk.sitkUInt8)
        cleaned = remove_small_cc_per_slice_2d(img, min_size=min_cc_size_2d, fully_connected=fully_connected)
        sitk.WriteImage(cleaned, os.path.join(out_dir, fname))


# =========================
# 4) Step4: 自适应 2D 膨胀（seed=Step3 输出，约束=diff）
# =========================
def max_iter_from_size(v: int) -> int:
    if v < 10:
        return 1
    elif v < 20:
        return 2
    elif v < 30:
        return 3
    elif v < 40:
        return 4
    elif v < 50:
        return 6
    else:
        return 10


def run_step4_adaptive_grow(step3_dir: str, root_dir: str, out_dir: str, groups: dict,
                            radius_2d=(1, 1, 0), growth_thr=0.05):
    os.makedirs(out_dir, exist_ok=True)

    for gid, items in tqdm(sorted(groups.items()), desc="Step4: adaptive grow"):
        if "diff" not in items:
            print(f"[Step4 Skip] {gid}: missing diff")
            continue

        seed_path = os.path.join(step3_dir, f"{gid}_mask_pseudo_new_grow.nii.gz")
        if not os.path.exists(seed_path):
            print(f"[Step4 Skip] {gid}: missing seed from step3 -> {os.path.basename(seed_path)}")
            continue

        diff_path = os.path.join(root_dir, items["diff"])
        if not os.path.exists(diff_path):
            print(f"[Step4 Skip] {gid}: diff file not found on disk")
            continue

        seed = sitk_read_bin(seed_path)
        diff = sitk_read_bin(diff_path)
        seed.CopyInformation(diff)

        if sitk_voxel_sum(diff) == 0 or sitk_voxel_sum(seed) == 0:
            out = sitk.Image(diff.GetSize(), sitk.sitkUInt8)
            out.CopyInformation(diff)
        else:
            init = sitk.And(seed, diff)

            seed_np = sitk.GetArrayFromImage(init)   # [Z, Y, X]
            diff_np = sitk.GetArrayFromImage(diff)
            out_np  = np.zeros_like(seed_np, dtype=np.uint8)

            # 逐 slice 自适应（2D CC 粒度）
            for z in range(seed_np.shape[0]):
                seed_z = seed_np[z]
                diff_z = diff_np[z]

                if seed_z.sum() == 0:
                    continue

                seed_img = sitk.GetImageFromArray(seed_z.astype(np.uint8))
                seed_img = sitk.Cast(seed_img > 0, sitk.sitkUInt8)

                cc = sitk.ConnectedComponent(seed_img)
                stats = sitk.LabelShapeStatisticsImageFilter()
                stats.Execute(cc)

                diff_img = sitk.GetImageFromArray((diff_z > 0).astype(np.uint8))
                diff_img = sitk.Cast(diff_img > 0, sitk.sitkUInt8)

                for label in stats.GetLabels():
                    comp = sitk.Cast(cc == label, sitk.sitkUInt8)
                    v = int(stats.GetNumberOfPixels(label))

                    max_it = max_iter_from_size(v)
                    prev_sum = int(np.sum(sitk.GetArrayFromImage(comp)))

                    for it in range(max_it):
                        grown = sitk.BinaryDilate(comp, list(radius_2d))
                        grown = sitk.And(grown, diff_img)

                        cur_sum = int(np.sum(sitk.GetArrayFromImage(grown)))
                        if it >= 1:
                            gr = (cur_sum - prev_sum) / max(prev_sum, 1)
                            if gr < float(growth_thr):
                                break

                        comp = grown
                        prev_sum = cur_sum

                    out_np[z] |= sitk.GetArrayFromImage(comp).astype(np.uint8)

            out = sitk.GetImageFromArray(out_np.astype(np.uint8))
            out.CopyInformation(diff)

        out_name = f"{gid}_mask_pseudo_new_grow_adaptive2D.nii.gz"
        sitk.WriteImage(out, os.path.join(out_dir, out_name))


# =========================
# 5) Step5: 2D 清理 + 填洞（最终）
# =========================
def remove_small_cc_and_fill_holes_2d(mask3d: sitk.Image, min_size=10,
                                      fully_connected=True, fill_holes=True) -> sitk.Image:
    mask3d = sitk.Cast(mask3d > 0, sitk.sitkUInt8)
    size = list(mask3d.GetSize())  # (x, y, z)

    extractor = sitk.ExtractImageFilter()
    extractor.SetSize([size[0], size[1], 0])

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(bool(fully_connected))

    relabel = sitk.RelabelComponentImageFilter()
    relabel.SetMinimumObjectSize(int(min_size))

    cleaned_slices = []
    for z in range(size[2]):
        extractor.SetIndex([0, 0, z])
        sl = extractor.Execute(mask3d)

        cc = cc_filter.Execute(sl)
        cc = relabel.Execute(cc)
        sl_clean = sitk.Cast(cc > 0, sitk.sitkUInt8)

        if fill_holes:
            sl_clean = sitk.BinaryFillhole(sl_clean, fullyConnected=fully_connected)

        cleaned_slices.append(sl_clean)

    out3d = sitk.JoinSeries(cleaned_slices)
    out3d = sitk.Cast(out3d > 0, sitk.sitkUInt8)
    out3d.CopyInformation(mask3d)
    return out3d


def run_step5_final(step4_dir: str, out_dir: str,
                    min_cc_size_2d=10, fully_connected=True, fill_holes=True):
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in list_nii(step4_dir) if f.lower().endswith(".nii.gz")]
    for fname in tqdm(sorted(files), desc="Step5: final clean+fill"):
        in_path = os.path.join(step4_dir, fname)
        img = sitk.ReadImage(in_path, sitk.sitkUInt8)

        out = remove_small_cc_and_fill_holes_2d(
            img,
            min_size=min_cc_size_2d,
            fully_connected=fully_connected,
            fill_holes=fill_holes
        )
        sitk.WriteImage(out, os.path.join(out_dir, fname))


# =========================
# Pipeline 入口：一键跑完
# =========================
def run_openms_pipeline(
    root_dir: str,
    save_intermediate: bool = True,
    # step1
    top_percent: float = 0.1,#0.05
    min_keep: int = 1,
    min_cc_size: int = 10,
    min_cc_p90: float = 0.5,
    # step2
    fixed_radius=(1, 1, 0),
    fixed_max_iter: int = 1,
    # step3
    step3_min_cc_2d: int = 5,#5
    step3_fully_connected: bool = True,
    # step4
    adaptive_radius_2d=(1, 1, 0),
    adaptive_growth_thr: float = 0.05,
    # step5
    step5_min_cc_2d: int = 15,#10
    step5_fully_connected: bool = True,
    step5_fill_holes: bool = True,
):
    root_dir = os.path.abspath(root_dir)

    # 中间目录（都在 root_dir 下，避免你复制来复制去）
    step1_dir = os.path.join(root_dir, "step1_top5p")
    step2_dir = os.path.join(root_dir, "step2_fixed_grow")
    step3_dir = os.path.join(root_dir, "step3_clean2d")
    step4_dir = os.path.join(root_dir, "step4_adaptive2d")
    step5_dir = os.path.join(root_dir, "step5_final")

    groups = build_case_index(root_dir)
    print(f"[Index] groups found: {len(groups)}")

    # Step1
    run_step1(
        root_dir=root_dir,
        out_dir=step1_dir,
        groups=groups,
        top_percent=top_percent,
        min_keep=min_keep,
        min_cc_size=min_cc_size,
        min_cc_p90=min_cc_p90
    )

    # Step2
    run_step2_fixed_grow(
        step1_dir=step1_dir,
        root_dir=root_dir,
        out_dir=step2_dir,
        groups=groups,
        radius=fixed_radius,
        max_iter=fixed_max_iter
    )

    # Step3
    run_step3_clean(
        step2_dir=step2_dir,
        out_dir=step3_dir,
        min_cc_size_2d=step3_min_cc_2d,
        fully_connected=step3_fully_connected
    )

    # Step4
    run_step4_adaptive_grow(
        step3_dir=step3_dir,
        root_dir=root_dir,
        out_dir=step4_dir,
        groups=groups,
        radius_2d=adaptive_radius_2d,
        growth_thr=adaptive_growth_thr
    )

    # Step5
    run_step5_final(
        step4_dir=step4_dir,
        out_dir=step5_dir,
        min_cc_size_2d=step5_min_cc_2d,
        fully_connected=step5_fully_connected,
        fill_holes=step5_fill_holes
    )

    print("\n✅ Pipeline finished.")
    print("Outputs:")
    print("  Step1:", step1_dir)
    print("  Step2:", step2_dir)
    print("  Step3:", step3_dir)
    print("  Step4:", step4_dir)
    print("  Step5:", step5_dir)


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="包含 openms_* 的 nii/nii.gz 文件夹")
    parser.add_argument("--top_percent", type=float, default=0.05)
    parser.add_argument("--min_cc_size", type=int, default=10)
    parser.add_argument("--min_cc_p90", type=float, default=0.5)

    parser.add_argument("--fixed_max_iter", type=int, default=1)
    parser.add_argument("--step3_min_cc_2d", type=int, default=5)

    parser.add_argument("--adaptive_growth_thr", type=float, default=0.05)

    parser.add_argument("--step5_min_cc_2d", type=int, default=10)
    parser.add_argument("--no_fill_holes", action="store_true")

    args = parser.parse_args()

    run_openms_pipeline(
        root_dir=args.root_dir,
        top_percent=args.top_percent,
        min_cc_size=args.min_cc_size,
        min_cc_p90=args.min_cc_p90,
        fixed_max_iter=args.fixed_max_iter,
        step3_min_cc_2d=args.step3_min_cc_2d,
        adaptive_growth_thr=args.adaptive_growth_thr,
        step5_min_cc_2d=args.step5_min_cc_2d,
        step5_fill_holes=(not args.no_fill_holes)
    )


if __name__ == "__main__":
    #main()
    run_openms_pipeline(
        root_dir=r"D:\dataset\wmh\peizhun_make_mask\peizhun\try_make_dataset\LesSeg-20",
        top_percent=0.2,
        min_cc_size=5,
        min_cc_p90=0.5,
        fixed_max_iter=1,
        step3_min_cc_2d=5,
        adaptive_growth_thr=0.1,
        step5_min_cc_2d=10,
        step5_fill_holes="True"
    )



import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from typing import Tuple, List, Dict


CATEGORY_PATHS = {
    "chair": {
        "fisheye": r"C:\Users\Jayasree\FCV_PROJECT\York-Fisheye-Image-Rectification-Dataset-master\Chair\fisheye",
        "gt":      r"C:\Users\Jayasree\FCV_PROJECT\York-Fisheye-Image-Rectification-Dataset-master\Chair\perspective",
    },
    "cigarette_box": {
        "fisheye": r"C:\Users\Jayasree\FCV_PROJECT\York-Fisheye-Image-Rectification-Dataset-master\Cigarette box\fisheye",
        "gt":      r"C:\Users\Jayasree\FCV_PROJECT\York-Fisheye-Image-Rectification-Dataset-master\Cigarette box\perspective",
    },
    "skull": {
        "fisheye": r"C:\Users\Jayasree\FCV_PROJECT\York-Fisheye-Image-Rectification-Dataset-master\Skull\fisheye",
        "gt":      r"C:\Users\Jayasree\FCV_PROJECT\York-Fisheye-Image-Rectification-Dataset-master\Skull\perspective",
    },
    "teddy": {
        "fisheye": r"C:\Users\Jayasree\FCV_PROJECT\York-Fisheye-Image-Rectification-Dataset-master\Teddy\fisheye",
        "gt":      r"C:\Users\Jayasree\FCV_PROJECT\York-Fisheye-Image-Rectification-Dataset-master\Teddy\perspective",
    }
}


OUTPUT_ROOT = r"C:\Users\Jayasree\FCV_PROJECT\York-Fisheye-Image-Rectification-Dataset-master\OUTPUT"


START_INDEX = 1
END_INDEX = 10  

FOCAL_LENGTH = 300.0


def spherical_reprojection(img: np.ndarray, f: float = FOCAL_LENGTH) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    x_rect, y_rect = np.meshgrid(np.arange(w), np.arange(h))
    x_n = (x_rect - cx) / f
    y_n = (y_rect - cy) / f

    Zs = np.ones_like(x_n)
    norm = np.sqrt(x_n**2 + y_n**2 + Zs**2)
    Xs = x_n / norm
    Ys = y_n / norm
    Zs = Zs / norm

    theta = np.arccos(np.clip(Zs, -1.0, 1.0))
    sin_theta = np.sin(theta)
    sin_theta[sin_theta == 0] = 1e-8

    r_fish = f * theta
    u = cx + (r_fish * (Xs / sin_theta))
    v = cy + (r_fish * (Ys / sin_theta))

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    rectified = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rectified


def compute_metrics(corrected: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float, float]:
    gt_resized = cv2.resize(gt, (corrected.shape[1], corrected.shape[0]), interpolation=cv2.INTER_AREA)
    gray_corr = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    gray_gt = cv2.cvtColor(gt_resized, cv2.COLOR_BGR2GRAY)

    mse_val = float(np.mean((gray_corr.astype(np.float32) - gray_gt.astype(np.float32)) ** 2))
    psnr_val = float(cv2.PSNR(gray_gt, gray_corr))

    data_range = float(gray_gt.max() - gray_gt.min()) if gray_gt.max() != gray_gt.min() else 255.0
    try:
        ssim_val = float(ssim(gray_gt, gray_corr, data_range=data_range))
    except Exception:
        ssim_val = -1.0

    valid_pixels = np.count_nonzero(gray_corr)
    total_pixels = gray_corr.size
    fov_retention = float((valid_pixels / total_pixels) * 100.0)

    return mse_val, psnr_val, ssim_val, fov_retention


def format_fname(idx: int) -> str:
    return f"{idx:04d}.bmp"



def process_category(name: str, fisheye_dir: str, gt_dir: str, output_root: str) -> Dict[str, float]:
    print(f"\n=== Processing category: {name} ===")
    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    per_image_results: List[Tuple[str, float, float, float, float]] = []

    for idx in range(START_INDEX, END_INDEX + 1):
        fname = format_fname(idx)
        fisheye_path = os.path.join(fisheye_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        if not os.path.isfile(fisheye_path):
            print(f"    Missing fisheye: {fisheye_path} — skipping")
            continue
        if not os.path.isfile(gt_path):
            print(f"    Missing ground-truth: {gt_path} — skipping")
            continue

 
        print(f"   Processing image: {fname}")

        fisheye_img = cv2.imread(fisheye_path, cv2.IMREAD_COLOR)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        if fisheye_img is None or gt_img is None:
            print(f"  Error reading pair {fname} — skipping")
            continue

        corrected = spherical_reprojection(fisheye_img, f=FOCAL_LENGTH)

        out_fname = f"rectified_{fname}"
        out_full = os.path.join(out_dir, out_fname)
        cv2.imwrite(out_full, corrected)

        mse_v, psnr_v, ssim_v, fov_v = compute_metrics(corrected, gt_img)
        per_image_results.append((fname, mse_v, psnr_v, ssim_v, fov_v))

        print(f"     MSE:{mse_v:8.3f}  PSNR:{psnr_v:6.2f} dB  SSIM:{ssim_v:6.3f}  FoV:{fov_v:6.2f}%")

    if per_image_results:
        mse_avg = float(np.mean([r[1] for r in per_image_results]))
        psnr_avg = float(np.mean([r[2] for r in per_image_results]))
        ssim_vals = [r[3] for r in per_image_results if r[3] >= 0]
        ssim_avg = float(np.mean(ssim_vals)) if ssim_vals else -1.0
        fov_avg = float(np.mean([r[4] for r in per_image_results]))

        print(f"\n-- {name} summary: images processed: {len(per_image_results)}")
        print(f"   Average MSE:  {mse_avg:.3f}")
        print(f"   Average PSNR: {psnr_avg:.3f} dB")
        print(f"   Average SSIM: {ssim_avg:.3f}")
        print(f"   Average FoV:  {fov_avg:.2f}%")
    else:
        mse_avg = psnr_avg = ssim_avg = fov_avg = float('nan')
        print(f"\n-- {name} summary: no images processed.")

    return {
        "category": name,
        "count": len(per_image_results),
        "mse_avg": mse_avg,
        "psnr_avg": psnr_avg,
        "ssim_avg": ssim_avg,
        "fov_avg": fov_avg
    }


def main():
    for cat, paths in CATEGORY_PATHS.items():
        if not os.path.isdir(paths["fisheye"]):
            print(f"ERROR: fisheye directory missing for '{cat}': {paths['fisheye']}")
            return
        if not os.path.isdir(paths["gt"]):
            print(f"ERROR: ground-truth directory missing for '{cat}': {paths['gt']}")
            return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    ordered_cats = ["chair", "cigarette_box", "skull", "teddy"]
    all_category_results = []
    for cat in ordered_cats:
        paths = CATEGORY_PATHS.get(cat)
        if paths is None:
            print(f"Skipping unknown category: {cat}")
            continue
        res = process_category(cat, paths["fisheye"], paths["gt"], OUTPUT_ROOT)
        all_category_results.append(res)

    print("\n\n=== FINAL AVERAGED RESULTS (per category) ===")
    header = f"{'Category':15s} {'Count':>5s} {'MSE_avg':>12s} {'PSNR_avg(dB)':>14s} {'SSIM_avg':>12s} {'FoV_avg(%)':>12s}"
    print(header)
    print("-" * len(header))
    total_counts = 0
    accum = {"mse": [], "psnr": [], "ssim": [], "fov": []}
    for r in all_category_results:
        print(f"{r['category']:15s} {r['count']:5d} {r['mse_avg']:12.3f} {r['psnr_avg']:14.3f} {r['ssim_avg']:12.3f} {r['fov_avg']:12.2f}")
        if r['count'] > 0:
            total_counts += r['count']
            accum["mse"].append(r['mse_avg'])
            accum["psnr"].append(r['psnr_avg'])
            accum["ssim"].append(r['ssim_avg'])
            accum["fov"].append(r['fov_avg'])

    if accum["mse"]:
        overall = {
            "mse": float(np.mean(accum["mse"])),
            "psnr": float(np.mean(accum["psnr"])),
            "ssim": float(np.mean(accum["ssim"])),
            "fov": float(np.mean(accum["fov"]))
        }
        print("-" * len(header))
        print(f"{'OVERALL':15s} {total_counts:5d} {overall['mse']:12.3f} {overall['psnr']:14.3f} {overall['ssim']:12.3f} {overall['fov']:12.2f}")
    else:
        print("-- No valid category results to aggregate.")


if __name__ == "__main__":
    main()

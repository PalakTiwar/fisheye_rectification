'''CODE FOR BOX'''
# =============================================
# Improved Global Polynomial Fisheye Rectifier + Metrics (MSE, PSNR, SSIM)
# =============================================
import os, cv2, numpy as np, json, csv
from scipy.optimize import least_squares
from zipfile import ZipFile
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

# ---------- PATH CONFIG ----------
BASE = "/content/drive/MyDrive/FCV_PROJECT"
FISHEYE_DIR = os.path.join(BASE, "fisheye_box")
GT_DIR      = os.path.join(BASE, "ground_truth_box")
OUT_DIR     = os.path.join(BASE, "results_polynomial_chair_refined_box")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- IMAGE LOADING ----------
def list_imgs_sorted(d):
    exts = ('.bmp','.png','.jpg','.jpeg','.tif')
    return [f for f in sorted(os.listdir(d)) if f.lower().endswith(exts)]

fisheye_files = list_imgs_sorted(FISHEYE_DIR)
gt_files      = list_imgs_sorted(GT_DIR)
common = sorted(list(set(fisheye_files).intersection(gt_files)))
print("Found pairs:", len(common))
if len(common) == 0:
    raise RuntimeError("No matching filenames found between fisheye and ground truth folders.")

# ---------- FEATURE MATCHING ----------
def mutual_sift_matches(img1, img2, ratio=0.75, max_feats=5000):
    img2 = cv2.GaussianBlur(img2, (5,5), 0)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    try:
        sift = cv2.SIFT_create(max_feats)
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
    except:
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher()
    m12 = bf.knnMatch(des1, des2, k=2)
    good12 = [m for (m,n) in m12 if len((m,n))==2 and m.distance < ratio * n.distance]
    m21 = bf.knnMatch(des2, des1, k=2)
    good21 = [m for (m,n) in m21 if len((m,n))==2 and m.distance < ratio * n.distance]
    idx21 = {m.queryIdx: m.trainIdx for m in good21}
    mutual = []
    for m in good12:
        if idx21.get(m.trainIdx, -1) == m.queryIdx:
            mutual.append(m)
    if len(mutual) < 8:
        return np.empty((0,2)), np.empty((0,2))
    pts1 = np.float32([kp1[m.queryIdx].pt for m in mutual])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in mutual])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    if mask is None:
        return np.empty((0,2)), np.empty((0,2))
    mask = mask.ravel().astype(bool)
    return pts1[mask], pts2[mask]

# ---------- Distortion Model ----------
def distort_points(xu, yu, k, p):
    r2 = xu**2 + yu**2
    r4 = r2**2
    r6 = r2**3
    r8 = r4**2
    k1, k2, k3, k4 = k
    p1, p2 = p
    radial = 1 + k1*r2 + k2*r4 + k3*r6 + k4*r8
    xrd = xu * radial + 2*p1*xu*yu + p2*(r2 + 2*xu**2)
    yrd = yu * radial + p1*(r2 + 2*yu**2) + 2*p2*xu*yu
    return xrd, yrd

def project_pixels(xd, yd, fx, fy, cx, cy):
    return fx * xd + cx, fy * yd + cy

def residuals_with_reg(params, src_pts, dst_pts, w, h, reg_weight=1e-3):
    fx, fy, cx, cy = params[0:4]
    k = params[4:8]
    p = params[8:10]
    xu = (src_pts[:,0] - cx)/fx
    yu = (src_pts[:,1] - cy)/fy
    xd, yd = distort_points(xu, yu, k, p)
    u_proj, v_proj = project_pixels(xd, yd, fx, fy, cx, cy)
    res = np.concatenate([u_proj - dst_pts[:,0], v_proj - dst_pts[:,1]])
    reg = np.concatenate([
        np.sqrt(reg_weight) * np.array(k),
        np.sqrt(reg_weight*0.5) * np.array(p),
        np.sqrt(reg_weight*1e-2) * ((np.array([cx, cy]) - np.array([w/2, h/2])) / np.array([w,h]))
    ])
    return np.concatenate([res, reg])

# ---------- Collect all correspondences ----------
all_src, all_dst, used = [], [], []
for fname in common:
    gt = cv2.imread(os.path.join(GT_DIR, fname))
    fis = cv2.imread(os.path.join(FISHEYE_DIR, fname))
    if gt is None or fis is None:
        continue
    s,d = mutual_sift_matches(gt, fis)
    if s.shape[0] < 8:
        orb = cv2.ORB_create(8000)
        gray1 = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(fis, cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:50]
            s = np.float32([kp1[m.queryIdx].pt for m in matches])
            d = np.float32([kp2[m.trainIdx].pt for m in matches])
    if s.shape[0] >= 8:
        if s.shape[0] > 1500:
            idx = np.random.choice(np.arange(s.shape[0]), 1500, replace=False)
            s, d = s[idx], d[idx]
        all_src.append(s)
        all_dst.append(d)
        used.append(fname)
        print(f"Using {fname} with {s.shape[0]} matches.")
    else:
        print("Still skipping", fname, "— too few matches:", s.shape[0])

if not all_src:
    raise RuntimeError("No valid feature correspondences found!")

all_src = np.vstack(all_src)
all_dst = np.vstack(all_dst)
print("Accumulated correspondences:", all_src.shape[0])

# ---------- Optimize Global Parameters ----------
h, w = cv2.imread(os.path.join(FISHEYE_DIR, used[0])).shape[:2]
init = np.array([0.9*w, 0.9*w, w/2, h/2, -0.5, 0.1, 0.01, 0.0, 0.0, 0.0])
lb = [0.3*w, 0.3*w, 0, 0, -5, -5, -1, -1, -1, -1]
ub = [3*w, 3*w, w, h, 5, 5, 1, 1, 1, 1]

print("Starting optimization...")
res = least_squares(residuals_with_reg, init, args=(all_src, all_dst, w, h),
                    loss='soft_l1', f_scale=2.0, bounds=(lb, ub),
                    verbose=2, max_nfev=10000)
params = res.x
print("Optimized Parameters:", params)

# ---------- Undistortion Function (with inpainting fix) ----------
def undistort_image_forward(fisheye_img, params, scale=1.2):
    h, w = fisheye_img.shape[:2]
    fx, fy, cx, cy = params[0:4]
    fx *= scale; fy *= scale
    k = params[4:8]; p = params[8:10]
    u_t, v_t = np.meshgrid(np.arange(w), np.arange(h))
    xu = (u_t - cx) / fx; yu = (v_t - cy) / fy
    xd, yd = distort_points(xu, yu, k, p)
    map_x = (fx * xd + cx).astype(np.float32)
    map_y = (fy * yd + cy).astype(np.float32)
    und = cv2.remap(fisheye_img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
    hole_mask = (gray <= 5).astype(np.uint8) * 255
    if np.count_nonzero(hole_mask) > 0:
        kernel = np.ones((7,7), np.uint8)
        hole_mask = cv2.dilate(hole_mask, kernel, iterations=2)
        und = cv2.inpaint(und, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return und

# ---------- Apply to All and Evaluate ----------
results = []
for fname in common:
    fis = cv2.imread(os.path.join(FISHEYE_DIR, fname))
    gt  = cv2.imread(os.path.join(GT_DIR, fname))
    if fis is None or gt is None: continue
    s,d = mutual_sift_matches(gt, fis)
    if s.shape[0] < 8: continue
    und = undistort_image_forward(fis, params)
    cv2.imwrite(os.path.join(OUT_DIR, f"rectified_{fname}"), und)

    # --- Compute metrics (MSE, PSNR, SSIM) ---
    gt_resized = cv2.resize(gt, (und.shape[1], und.shape[0]))
    mse_val = np.mean((und.astype(np.float32) - gt_resized.astype(np.float32)) ** 2)
    psnr_val = cv2.PSNR(gt_resized, und)
    ssim_val = ssim(cv2.cvtColor(gt_resized, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), data_range=255)

    xu = (s[:,0]-params[2])/params[0]; yu = (s[:,1]-params[3])/params[1]
    xd, yd = distort_points(xu, yu, params[4:8], params[8:10])
    u_proj, v_proj = project_pixels(xd, yd, params[0], params[1], params[2], params[3])
    errs = np.sqrt((u_proj - d[:,0])**2 + (v_proj - d[:,1])**2)
    mean_rpe = float(errs.mean()); std_rpe = float(errs.std())
    fov_ret = 100.0 * np.sum(np.any(und != 0, axis=2)) / (und.shape[0]*und.shape[1])

    results.append({
    "filename": str(fname),
    "mean_rpe": float(mean_rpe),
    "std_rpe": float(std_rpe),
    "fov_retention": float(fov_ret),
    "mse": float(mse_val),
    "psnr": float(psnr_val),
    "ssim": float(ssim_val)
    })


    print(f"{fname}: RPE={mean_rpe:.3f}px  FOV={fov_ret:.2f}%  "
          f"MSE={mse_val:.3f}  PSNR={psnr_val:.2f}dB  SSIM={ssim_val:.4f}")

# ---------- Save Outputs ----------
csv_path = os.path.join(OUT_DIR, "results_summary_refined.csv")
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ["filename","mean_rpe","std_rpe","fov_retention","mse","psnr","ssim"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

json_path = os.path.join(OUT_DIR, "results_summary_refined.json")
with open(json_path,'w') as jf:
    json.dump(results, jf, indent=2)

zip_name = os.path.join(BASE, f"polynomial_refined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
with ZipFile(zip_name, 'w') as zipf:
    for root, _, files in os.walk(OUT_DIR):
        for f in files:
            zipf.write(os.path.join(root,f), arcname=os.path.join(os.path.relpath(root, OUT_DIR), f))

print("\n Finished! Saved refined results with MSE/PSNR/SSIM to:", OUT_DIR)

'''CODE FOR TEDDY'''
# =============================================
# Improved Global Polynomial Fisheye Rectifier + Metrics (MSE, PSNR, SSIM)
# =============================================
import os, cv2, numpy as np, json, csv
from scipy.optimize import least_squares
from zipfile import ZipFile
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

# ---------- PATH CONFIG ----------
BASE = "/content/drive/MyDrive/FCV_PROJECT"
FISHEYE_DIR = os.path.join(BASE, "fisheye_teddy")
GT_DIR      = os.path.join(BASE, "ground_truth_teddy")
OUT_DIR     = os.path.join(BASE, "results_polynomial_chair_refined_teddy")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- IMAGE LOADING ----------
def list_imgs_sorted(d):
    exts = ('.bmp','.png','.jpg','.jpeg','.tif')
    return [f for f in sorted(os.listdir(d)) if f.lower().endswith(exts)]

fisheye_files = list_imgs_sorted(FISHEYE_DIR)
gt_files      = list_imgs_sorted(GT_DIR)
common = sorted(list(set(fisheye_files).intersection(gt_files)))
print("Found pairs:", len(common))
if len(common) == 0:
    raise RuntimeError("No matching filenames found between fisheye and ground truth folders.")

# ---------- FEATURE MATCHING ----------
def mutual_sift_matches(img1, img2, ratio=0.75, max_feats=5000):
    img2 = cv2.GaussianBlur(img2, (5,5), 0)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    try:
        sift = cv2.SIFT_create(max_feats)
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
    except:
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher()
    m12 = bf.knnMatch(des1, des2, k=2)
    good12 = [m for (m,n) in m12 if len((m,n))==2 and m.distance < ratio * n.distance]
    m21 = bf.knnMatch(des2, des1, k=2)
    good21 = [m for (m,n) in m21 if len((m,n))==2 and m.distance < ratio * n.distance]
    idx21 = {m.queryIdx: m.trainIdx for m in good21}
    mutual = []
    for m in good12:
        if idx21.get(m.trainIdx, -1) == m.queryIdx:
            mutual.append(m)
    if len(mutual) < 8:
        return np.empty((0,2)), np.empty((0,2))
    pts1 = np.float32([kp1[m.queryIdx].pt for m in mutual])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in mutual])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    if mask is None:
        return np.empty((0,2)), np.empty((0,2))
    mask = mask.ravel().astype(bool)
    return pts1[mask], pts2[mask]

# ---------- Distortion Model ----------
def distort_points(xu, yu, k, p):
    r2 = xu**2 + yu**2
    r4 = r2**2
    r6 = r2**3
    r8 = r4**2
    k1, k2, k3, k4 = k
    p1, p2 = p
    radial = 1 + k1*r2 + k2*r4 + k3*r6 + k4*r8
    xrd = xu * radial + 2*p1*xu*yu + p2*(r2 + 2*xu**2)
    yrd = yu * radial + p1*(r2 + 2*yu**2) + 2*p2*xu*yu
    return xrd, yrd

def project_pixels(xd, yd, fx, fy, cx, cy):
    return fx * xd + cx, fy * yd + cy

def residuals_with_reg(params, src_pts, dst_pts, w, h, reg_weight=1e-3):
    fx, fy, cx, cy = params[0:4]
    k = params[4:8]
    p = params[8:10]
    xu = (src_pts[:,0] - cx)/fx
    yu = (src_pts[:,1] - cy)/fy
    xd, yd = distort_points(xu, yu, k, p)
    u_proj, v_proj = project_pixels(xd, yd, fx, fy, cx, cy)
    res = np.concatenate([u_proj - dst_pts[:,0], v_proj - dst_pts[:,1]])
    reg = np.concatenate([
        np.sqrt(reg_weight) * np.array(k),
        np.sqrt(reg_weight*0.5) * np.array(p),
        np.sqrt(reg_weight*1e-2) * ((np.array([cx, cy]) - np.array([w/2, h/2])) / np.array([w,h]))
    ])
    return np.concatenate([res, reg])

# ---------- Collect all correspondences ----------
all_src, all_dst, used = [], [], []
for fname in common:
    gt = cv2.imread(os.path.join(GT_DIR, fname))
    fis = cv2.imread(os.path.join(FISHEYE_DIR, fname))
    if gt is None or fis is None:
        continue
    s,d = mutual_sift_matches(gt, fis)
    if s.shape[0] < 8:
        orb = cv2.ORB_create(8000)
        gray1 = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(fis, cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:50]
            s = np.float32([kp1[m.queryIdx].pt for m in matches])
            d = np.float32([kp2[m.trainIdx].pt for m in matches])
    if s.shape[0] >= 8:
        if s.shape[0] > 1500:
            idx = np.random.choice(np.arange(s.shape[0]), 1500, replace=False)
            s, d = s[idx], d[idx]
        all_src.append(s)
        all_dst.append(d)
        used.append(fname)
        print(f"Using {fname} with {s.shape[0]} matches.")
    else:
        print("Still skipping", fname, "— too few matches:", s.shape[0])

if not all_src:
    raise RuntimeError("No valid feature correspondences found!")

all_src = np.vstack(all_src)
all_dst = np.vstack(all_dst)
print("Accumulated correspondences:", all_src.shape[0])

# ---------- Optimize Global Parameters ----------
h, w = cv2.imread(os.path.join(FISHEYE_DIR, used[0])).shape[:2]
init = np.array([0.9*w, 0.9*w, w/2, h/2, -0.5, 0.1, 0.01, 0.0, 0.0, 0.0])
lb = [0.3*w, 0.3*w, 0, 0, -5, -5, -1, -1, -1, -1]
ub = [3*w, 3*w, w, h, 5, 5, 1, 1, 1, 1]

print("Starting optimization...")
res = least_squares(residuals_with_reg, init, args=(all_src, all_dst, w, h),
                    loss='soft_l1', f_scale=2.0, bounds=(lb, ub),
                    verbose=2, max_nfev=10000)
params = res.x
print("Optimized Parameters:", params)

# ---------- Undistortion Function (with inpainting fix) ----------
def undistort_image_forward(fisheye_img, params, scale=1.2):
    h, w = fisheye_img.shape[:2]
    fx, fy, cx, cy = params[0:4]
    fx *= scale; fy *= scale
    k = params[4:8]; p = params[8:10]
    u_t, v_t = np.meshgrid(np.arange(w), np.arange(h))
    xu = (u_t - cx) / fx; yu = (v_t - cy) / fy
    xd, yd = distort_points(xu, yu, k, p)
    map_x = (fx * xd + cx).astype(np.float32)
    map_y = (fy * yd + cy).astype(np.float32)
    und = cv2.remap(fisheye_img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
    hole_mask = (gray <= 5).astype(np.uint8) * 255
    if np.count_nonzero(hole_mask) > 0:
        kernel = np.ones((7,7), np.uint8)
        hole_mask = cv2.dilate(hole_mask, kernel, iterations=2)
        und = cv2.inpaint(und, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return und

# ---------- Apply to All and Evaluate ----------
results = []
for fname in common:
    fis = cv2.imread(os.path.join(FISHEYE_DIR, fname))
    gt  = cv2.imread(os.path.join(GT_DIR, fname))
    if fis is None or gt is None: continue
    s,d = mutual_sift_matches(gt, fis)
    if s.shape[0] < 8: continue
    und = undistort_image_forward(fis, params)
    cv2.imwrite(os.path.join(OUT_DIR, f"rectified_{fname}"), und)

    # --- Compute metrics (MSE, PSNR, SSIM) ---
    gt_resized = cv2.resize(gt, (und.shape[1], und.shape[0]))
    mse_val = np.mean((und.astype(np.float32) - gt_resized.astype(np.float32)) ** 2)
    psnr_val = cv2.PSNR(gt_resized, und)
    ssim_val = ssim(cv2.cvtColor(gt_resized, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), data_range=255)

    xu = (s[:,0]-params[2])/params[0]; yu = (s[:,1]-params[3])/params[1]
    xd, yd = distort_points(xu, yu, params[4:8], params[8:10])
    u_proj, v_proj = project_pixels(xd, yd, params[0], params[1], params[2], params[3])
    errs = np.sqrt((u_proj - d[:,0])**2 + (v_proj - d[:,1])**2)
    mean_rpe = float(errs.mean()); std_rpe = float(errs.std())
    fov_ret = 100.0 * np.sum(np.any(und != 0, axis=2)) / (und.shape[0]*und.shape[1])

    results.append({
    "filename": str(fname),
    "mean_rpe": float(mean_rpe),
    "std_rpe": float(std_rpe),
    "fov_retention": float(fov_ret),
    "mse": float(mse_val),
    "psnr": float(psnr_val),
    "ssim": float(ssim_val)
    })


    print(f"{fname}: RPE={mean_rpe:.3f}px  FOV={fov_ret:.2f}%  "
          f"MSE={mse_val:.3f}  PSNR={psnr_val:.2f}dB  SSIM={ssim_val:.4f}")

# ---------- Save Outputs ----------
csv_path = os.path.join(OUT_DIR, "results_summary_refined.csv")
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ["filename","mean_rpe","std_rpe","fov_retention","mse","psnr","ssim"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

json_path = os.path.join(OUT_DIR, "results_summary_refined.json")
with open(json_path,'w') as jf:
    json.dump(results, jf, indent=2)

zip_name = os.path.join(BASE, f"polynomial_refined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
with ZipFile(zip_name, 'w') as zipf:
    for root, _, files in os.walk(OUT_DIR):
        for f in files:
            zipf.write(os.path.join(root,f), arcname=os.path.join(os.path.relpath(root, OUT_DIR), f))

print("\n Finished! Saved refined results with MSE/PSNR/SSIM to:", OUT_DIR)

'''CODE FOR SKULL'''
# =============================================
# Improved Global Polynomial Fisheye Rectifier + Metrics (MSE, PSNR, SSIM)
# =============================================
import os, cv2, numpy as np, json, csv
from scipy.optimize import least_squares
from zipfile import ZipFile
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

# ---------- PATH CONFIG ----------
BASE = "/content/drive/MyDrive/FCV_PROJECT"
FISHEYE_DIR = os.path.join(BASE, "fisheye_skull")
GT_DIR      = os.path.join(BASE, "ground_truth_skull")
OUT_DIR     = os.path.join(BASE, "results_polynomial_chair_refined_skull")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- IMAGE LOADING ----------
def list_imgs_sorted(d):
    exts = ('.bmp','.png','.jpg','.jpeg','.tif')
    return [f for f in sorted(os.listdir(d)) if f.lower().endswith(exts)]

fisheye_files = list_imgs_sorted(FISHEYE_DIR)
gt_files      = list_imgs_sorted(GT_DIR)
common = sorted(list(set(fisheye_files).intersection(gt_files)))
print("Found pairs:", len(common))
if len(common) == 0:
    raise RuntimeError("No matching filenames found between fisheye and ground truth folders.")

# ---------- FEATURE MATCHING ----------
def mutual_sift_matches(img1, img2, ratio=0.75, max_feats=5000):
    img2 = cv2.GaussianBlur(img2, (5,5), 0)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    try:
        sift = cv2.SIFT_create(max_feats)
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
    except:
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher()
    m12 = bf.knnMatch(des1, des2, k=2)
    good12 = [m for (m,n) in m12 if len((m,n))==2 and m.distance < ratio * n.distance]
    m21 = bf.knnMatch(des2, des1, k=2)
    good21 = [m for (m,n) in m21 if len((m,n))==2 and m.distance < ratio * n.distance]
    idx21 = {m.queryIdx: m.trainIdx for m in good21}
    mutual = []
    for m in good12:
        if idx21.get(m.trainIdx, -1) == m.queryIdx:
            mutual.append(m)
    if len(mutual) < 8:
        return np.empty((0,2)), np.empty((0,2))
    pts1 = np.float32([kp1[m.queryIdx].pt for m in mutual])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in mutual])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    if mask is None:
        return np.empty((0,2)), np.empty((0,2))
    mask = mask.ravel().astype(bool)
    return pts1[mask], pts2[mask]

# ---------- Distortion Model ----------
def distort_points(xu, yu, k, p):
    r2 = xu**2 + yu**2
    r4 = r2**2
    r6 = r2**3
    r8 = r4**2
    k1, k2, k3, k4 = k
    p1, p2 = p
    radial = 1 + k1*r2 + k2*r4 + k3*r6 + k4*r8
    xrd = xu * radial + 2*p1*xu*yu + p2*(r2 + 2*xu**2)
    yrd = yu * radial + p1*(r2 + 2*yu**2) + 2*p2*xu*yu
    return xrd, yrd

def project_pixels(xd, yd, fx, fy, cx, cy):
    return fx * xd + cx, fy * yd + cy

def residuals_with_reg(params, src_pts, dst_pts, w, h, reg_weight=1e-3):
    fx, fy, cx, cy = params[0:4]
    k = params[4:8]
    p = params[8:10]
    xu = (src_pts[:,0] - cx)/fx
    yu = (src_pts[:,1] - cy)/fy
    xd, yd = distort_points(xu, yu, k, p)
    u_proj, v_proj = project_pixels(xd, yd, fx, fy, cx, cy)
    res = np.concatenate([u_proj - dst_pts[:,0], v_proj - dst_pts[:,1]])
    reg = np.concatenate([
        np.sqrt(reg_weight) * np.array(k),
        np.sqrt(reg_weight*0.5) * np.array(p),
        np.sqrt(reg_weight*1e-2) * ((np.array([cx, cy]) - np.array([w/2, h/2])) / np.array([w,h]))
    ])
    return np.concatenate([res, reg])

# ---------- Collect all correspondences ----------
all_src, all_dst, used = [], [], []
for fname in common:
    gt = cv2.imread(os.path.join(GT_DIR, fname))
    fis = cv2.imread(os.path.join(FISHEYE_DIR, fname))
    if gt is None or fis is None:
        continue
    s,d = mutual_sift_matches(gt, fis)
    if s.shape[0] < 8:
        orb = cv2.ORB_create(8000)
        gray1 = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(fis, cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:50]
            s = np.float32([kp1[m.queryIdx].pt for m in matches])
            d = np.float32([kp2[m.trainIdx].pt for m in matches])
    if s.shape[0] >= 8:
        if s.shape[0] > 1500:
            idx = np.random.choice(np.arange(s.shape[0]), 1500, replace=False)
            s, d = s[idx], d[idx]
        all_src.append(s)
        all_dst.append(d)
        used.append(fname)
        print(f"Using {fname} with {s.shape[0]} matches.")
    else:
        print("Still skipping", fname, "— too few matches:", s.shape[0])

if not all_src:
    raise RuntimeError("No valid feature correspondences found!")

all_src = np.vstack(all_src)
all_dst = np.vstack(all_dst)
print("Accumulated correspondences:", all_src.shape[0])

# ---------- Optimize Global Parameters ----------
h, w = cv2.imread(os.path.join(FISHEYE_DIR, used[0])).shape[:2]
init = np.array([0.9*w, 0.9*w, w/2, h/2, -0.5, 0.1, 0.01, 0.0, 0.0, 0.0])
lb = [0.3*w, 0.3*w, 0, 0, -5, -5, -1, -1, -1, -1]
ub = [3*w, 3*w, w, h, 5, 5, 1, 1, 1, 1]

print("Starting optimization...")
res = least_squares(residuals_with_reg, init, args=(all_src, all_dst, w, h),
                    loss='soft_l1', f_scale=2.0, bounds=(lb, ub),
                    verbose=2, max_nfev=10000)
params = res.x
print("Optimized Parameters:", params)

# ---------- Undistortion Function (with inpainting fix) ----------
def undistort_image_forward(fisheye_img, params, scale=1.12):
    h, w = fisheye_img.shape[:2]
    fx, fy, cx, cy = params[0:4]
    fx *= scale; fy *= scale
    k = params[4:8]; p = params[8:10]

    # --- Compute undistortion maps ---
    u_t, v_t = np.meshgrid(np.arange(w), np.arange(h))
    xu = (u_t - cx) / fx
    yu = (v_t - cy) / fy
    xd, yd = distort_points(xu, yu, k, p)
    map_x = (fx * xd + cx).astype(np.float32)
    map_y = (fy * yd + cy).astype(np.float32)

    # --- Use Lanczos interpolation for natural texture and reflective border for continuity ---
    und = cv2.remap(
        fisheye_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # --- Optional mild sharpening for edge definition without artifacts ---
    blur = cv2.GaussianBlur(und, (0, 0), 0.8)
    und = cv2.addWeighted(und, 1.15, blur, -0.15, 0)

    return und



# ---------- Apply to All and Evaluate ----------
results = []
for fname in common:
    fis = cv2.imread(os.path.join(FISHEYE_DIR, fname))
    gt  = cv2.imread(os.path.join(GT_DIR, fname))
    if fis is None or gt is None: continue
    s,d = mutual_sift_matches(gt, fis)
    if s.shape[0] < 8: continue
    und = undistort_image_forward(fis, params)
    cv2.imwrite(os.path.join(OUT_DIR, f"rectified_{fname}"), und)

    # --- Compute metrics (MSE, PSNR, SSIM) ---
    gt_resized = cv2.resize(gt, (und.shape[1], und.shape[0]))
    mse_val = np.mean((und.astype(np.float32) - gt_resized.astype(np.float32)) ** 2)
    psnr_val = cv2.PSNR(gt_resized, und)
    ssim_val = ssim(cv2.cvtColor(gt_resized, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), data_range=255)

    xu = (s[:,0]-params[2])/params[0]; yu = (s[:,1]-params[3])/params[1]
    xd, yd = distort_points(xu, yu, params[4:8], params[8:10])
    u_proj, v_proj = project_pixels(xd, yd, params[0], params[1], params[2], params[3])
    errs = np.sqrt((u_proj - d[:,0])**2 + (v_proj - d[:,1])**2)
    mean_rpe = float(errs.mean()); std_rpe = float(errs.std())
    fov_ret = 100.0 * np.sum(np.any(und != 0, axis=2)) / (und.shape[0]*und.shape[1])

    results.append({
    "filename": str(fname),
    "mean_rpe": float(mean_rpe),
    "std_rpe": float(std_rpe),
    "fov_retention": float(fov_ret),
    "mse": float(mse_val),
    "psnr": float(psnr_val),
    "ssim": float(ssim_val)
    })


    print(f"{fname}: RPE={mean_rpe:.3f}px  FOV={fov_ret:.2f}%  "
          f"MSE={mse_val:.3f}  PSNR={psnr_val:.2f}dB  SSIM={ssim_val:.4f}")

# ---------- Save Outputs ----------
csv_path = os.path.join(OUT_DIR, "results_summary_refined.csv")
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ["filename","mean_rpe","std_rpe","fov_retention","mse","psnr","ssim"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

json_path = os.path.join(OUT_DIR, "results_summary_refined.json")
with open(json_path,'w') as jf:
    json.dump(results, jf, indent=2)

zip_name = os.path.join(BASE, f"polynomial_refined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
with ZipFile(zip_name, 'w') as zipf:
    for root, _, files in os.walk(OUT_DIR):
        for f in files:
            zipf.write(os.path.join(root,f), arcname=os.path.join(os.path.relpath(root, OUT_DIR), f))

print("\n Finished! Saved refined results with MSE/PSNR/SSIM to:", OUT_DIR)


'''CODE FOR CHAIR'''
# =============================================
# Improved Global Polynomial Fisheye Rectifier + Metrics (MSE, PSNR, SSIM)
# =============================================
import os, cv2, numpy as np, json, csv
from scipy.optimize import least_squares
from zipfile import ZipFile
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

# ---------- PATH CONFIG ----------
BASE = "/content/drive/MyDrive/FCV_PROJECT"
FISHEYE_DIR = os.path.join(BASE, "fisheye_chair")
GT_DIR      = os.path.join(BASE, "ground_truth_chair")
OUT_DIR     = os.path.join(BASE, "results_polynomial_chair_refined")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- IMAGE LOADING ----------
def list_imgs_sorted(d):
    exts = ('.bmp','.png','.jpg','.jpeg','.tif')
    return [f for f in sorted(os.listdir(d)) if f.lower().endswith(exts)]

fisheye_files = list_imgs_sorted(FISHEYE_DIR)
gt_files      = list_imgs_sorted(GT_DIR)
common = sorted(list(set(fisheye_files).intersection(gt_files)))
print("Found pairs:", len(common))
if len(common) == 0:
    raise RuntimeError("No matching filenames found between fisheye and ground truth folders.")

# ---------- FEATURE MATCHING ----------
def mutual_sift_matches(img1, img2, ratio=0.75, max_feats=5000):
    img2 = cv2.GaussianBlur(img2, (5,5), 0)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    try:
        sift = cv2.SIFT_create(max_feats)
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
    except:
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher()
    m12 = bf.knnMatch(des1, des2, k=2)
    good12 = [m for (m,n) in m12 if len((m,n))==2 and m.distance < ratio * n.distance]
    m21 = bf.knnMatch(des2, des1, k=2)
    good21 = [m for (m,n) in m21 if len((m,n))==2 and m.distance < ratio * n.distance]
    idx21 = {m.queryIdx: m.trainIdx for m in good21}
    mutual = []
    for m in good12:
        if idx21.get(m.trainIdx, -1) == m.queryIdx:
            mutual.append(m)
    if len(mutual) < 8:
        return np.empty((0,2)), np.empty((0,2))
    pts1 = np.float32([kp1[m.queryIdx].pt for m in mutual])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in mutual])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    if mask is None:
        return np.empty((0,2)), np.empty((0,2))
    mask = mask.ravel().astype(bool)
    return pts1[mask], pts2[mask]

# ---------- Distortion Model ----------
def distort_points(xu, yu, k, p):
    r2 = xu**2 + yu**2
    r4 = r2**2
    r6 = r2**3
    r8 = r4**2
    k1, k2, k3, k4 = k
    p1, p2 = p
    radial = 1 + k1*r2 + k2*r4 + k3*r6 + k4*r8
    xrd = xu * radial + 2*p1*xu*yu + p2*(r2 + 2*xu**2)
    yrd = yu * radial + p1*(r2 + 2*yu**2) + 2*p2*xu*yu
    return xrd, yrd

def project_pixels(xd, yd, fx, fy, cx, cy):
    return fx * xd + cx, fy * yd + cy

def residuals_with_reg(params, src_pts, dst_pts, w, h, reg_weight=1e-3):
    fx, fy, cx, cy = params[0:4]
    k = params[4:8]
    p = params[8:10]
    xu = (src_pts[:,0] - cx)/fx
    yu = (src_pts[:,1] - cy)/fy
    xd, yd = distort_points(xu, yu, k, p)
    u_proj, v_proj = project_pixels(xd, yd, fx, fy, cx, cy)
    res = np.concatenate([u_proj - dst_pts[:,0], v_proj - dst_pts[:,1]])
    reg = np.concatenate([
        np.sqrt(reg_weight) * np.array(k),
        np.sqrt(reg_weight*0.5) * np.array(p),
        np.sqrt(reg_weight*1e-2) * ((np.array([cx, cy]) - np.array([w/2, h/2])) / np.array([w,h]))
    ])
    return np.concatenate([res, reg])

# ---------- Collect all correspondences ----------
all_src, all_dst, used = [], [], []
for fname in common:
    gt = cv2.imread(os.path.join(GT_DIR, fname))
    fis = cv2.imread(os.path.join(FISHEYE_DIR, fname))
    if gt is None or fis is None:
        continue
    s,d = mutual_sift_matches(gt, fis)
    if s.shape[0] < 8:
        orb = cv2.ORB_create(8000)
        gray1 = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(fis, cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:50]
            s = np.float32([kp1[m.queryIdx].pt for m in matches])
            d = np.float32([kp2[m.trainIdx].pt for m in matches])
    if s.shape[0] >= 8:
        if s.shape[0] > 1500:
            idx = np.random.choice(np.arange(s.shape[0]), 1500, replace=False)
            s, d = s[idx], d[idx]
        all_src.append(s)
        all_dst.append(d)
        used.append(fname)
        print(f"Using {fname} with {s.shape[0]} matches.")
    else:
        print("Still skipping", fname, "— too few matches:", s.shape[0])

if not all_src:
    raise RuntimeError("No valid feature correspondences found!")

all_src = np.vstack(all_src)
all_dst = np.vstack(all_dst)
print("Accumulated correspondences:", all_src.shape[0])

# ---------- Optimize Global Parameters ----------
h, w = cv2.imread(os.path.join(FISHEYE_DIR, used[0])).shape[:2]
init = np.array([0.9*w, 0.9*w, w/2, h/2, -0.5, 0.1, 0.01, 0.0, 0.0, 0.0])
lb = [0.3*w, 0.3*w, 0, 0, -5, -5, -1, -1, -1, -1]
ub = [3*w, 3*w, w, h, 5, 5, 1, 1, 1, 1]

print("Starting optimization...")
res = least_squares(residuals_with_reg, init, args=(all_src, all_dst, w, h),
                    loss='soft_l1', f_scale=2.0, bounds=(lb, ub),
                    verbose=2, max_nfev=10000)
params = res.x
print("Optimized Parameters:", params)

# ---------- Undistortion Function (with inpainting fix) ----------
def undistort_image_forward(fisheye_img, params, scale=1.2):
    h, w = fisheye_img.shape[:2]
    fx, fy, cx, cy = params[0:4]
    fx *= scale; fy *= scale
    k = params[4:8]; p = params[8:10]
    u_t, v_t = np.meshgrid(np.arange(w), np.arange(h))
    xu = (u_t - cx) / fx; yu = (v_t - cy) / fy
    xd, yd = distort_points(xu, yu, k, p)
    map_x = (fx * xd + cx).astype(np.float32)
    map_y = (fy * yd + cy).astype(np.float32)
    und = cv2.remap(fisheye_img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
    hole_mask = (gray <= 5).astype(np.uint8) * 255
    if np.count_nonzero(hole_mask) > 0:
        kernel = np.ones((7,7), np.uint8)
        hole_mask = cv2.dilate(hole_mask, kernel, iterations=2)
        und = cv2.inpaint(und, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return und

# ---------- Apply to All and Evaluate ----------
results = []
for fname in common:
    fis = cv2.imread(os.path.join(FISHEYE_DIR, fname))
    gt  = cv2.imread(os.path.join(GT_DIR, fname))
    if fis is None or gt is None: continue
    s,d = mutual_sift_matches(gt, fis)
    if s.shape[0] < 8: continue
    und = undistort_image_forward(fis, params)
    cv2.imwrite(os.path.join(OUT_DIR, f"rectified_{fname}"), und)

    # --- Compute metrics (MSE, PSNR, SSIM) ---
    gt_resized = cv2.resize(gt, (und.shape[1], und.shape[0]))
    mse_val = np.mean((und.astype(np.float32) - gt_resized.astype(np.float32)) ** 2)
    psnr_val = cv2.PSNR(gt_resized, und)
    ssim_val = ssim(cv2.cvtColor(gt_resized, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), data_range=255)

    xu = (s[:,0]-params[2])/params[0]; yu = (s[:,1]-params[3])/params[1]
    xd, yd = distort_points(xu, yu, params[4:8], params[8:10])
    u_proj, v_proj = project_pixels(xd, yd, params[0], params[1], params[2], params[3])
    errs = np.sqrt((u_proj - d[:,0])**2 + (v_proj - d[:,1])**2)
    mean_rpe = float(errs.mean()); std_rpe = float(errs.std())
    fov_ret = 100.0 * np.sum(np.any(und != 0, axis=2)) / (und.shape[0]*und.shape[1])

    results.append({
    "filename": str(fname),
    "mean_rpe": float(mean_rpe),
    "std_rpe": float(std_rpe),
    "fov_retention": float(fov_ret),
    "mse": float(mse_val),
    "psnr": float(psnr_val),
    "ssim": float(ssim_val)
    })


    print(f"{fname}: RPE={mean_rpe:.3f}px  FOV={fov_ret:.2f}%  "
          f"MSE={mse_val:.3f}  PSNR={psnr_val:.2f}dB  SSIM={ssim_val:.4f}")

# ---------- Save Outputs ----------
csv_path = os.path.join(OUT_DIR, "results_summary_refined.csv")
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ["filename","mean_rpe","std_rpe","fov_retention","mse","psnr","ssim"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

json_path = os.path.join(OUT_DIR, "results_summary_refined.json")
with open(json_path,'w') as jf:
    json.dump(results, jf, indent=2)

zip_name = os.path.join(BASE, f"polynomial_refined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
with ZipFile(zip_name, 'w') as zipf:
    for root, _, files in os.walk(OUT_DIR):
        for f in files:
            zipf.write(os.path.join(root,f), arcname=os.path.join(os.path.relpath(root, OUT_DIR), f))

print("\n Finished! Saved refined results with MSE/PSNR/SSIM to:", OUT_DIR)

# Evaluation of Distortion Correction Techniques for Fisheye Camera Images

A comparative computer vision research project that evaluates two classical fisheye image distortion correction techniques—*Polynomial Calibration* and *Spherical Re-projection*—to analyze their effectiveness in geometric accuracy, perceptual quality, and field-of-view preservation. :contentReference[oaicite:0]{index=0}

---

##  Overview

Fisheye cameras provide an ultra-wide field of view, making them ideal for applications such as:

- Autonomous Driving (ADAS)
- Robotics
- Surveillance Systems
- Panoramic Imaging
- Virtual Reality

However, the nonlinear distortion introduced by fisheye lenses significantly affects image geometry and computer vision tasks.

This research performs a detailed comparison between two distortion correction approaches:

- Polynomial Calibration Model
- Spherical Re-projection Model

The objective is to determine the trade-offs between geometric precision and field-of-view preservation for real-world computer vision applications. :contentReference[oaicite:1]{index=1}

---

##  Objectives

- Compare two classical fisheye distortion correction techniques.
- Evaluate geometric accuracy after correction.
- Measure perceptual image quality.
- Analyze Field-of-View (FoV) retention.
- Study strengths and limitations of each approach.

---

##  Technologies Used

- Python
- OpenCV
- NumPy
- SciPy
- Matplotlib

---

##  Dataset

*York Fisheye Image Rectification Dataset*

Object categories used:

- Chair
- Cigarette Box
- Skull
- Teddy Bear

Each sample consists of:

- Distorted fisheye image
- Ground truth perspective image


---

##  Methodology

### 1. Polynomial Calibration Model

This method estimates intrinsic camera parameters and distortion coefficients through feature matching and nonlinear optimization.

Pipeline:

- Feature Detection (SIFT / ORB)
- Feature Matching
- Camera Parameter Estimation
- Radial & Tangential Distortion Modeling
- Image Rectification
- Bicubic Interpolation
- Image Inpainting

---

### 2. Spherical Re-projection

Instead of calibration, this method models fisheye geometry by projecting pixels onto a virtual sphere and re-projecting them onto a perspective plane.

Pipeline:

- Normalize Image Coordinates
- Map Pixels to Virtual Sphere
- Spherical Projection
- Perspective Reprojection
- Bilinear Interpolation
- Rectified Output


---

##  Evaluation Metrics

Performance was evaluated using:

- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Field-of-View (FoV) Retention

These metrics evaluate both geometric accuracy and perceptual quality.



---

##  Results

### Polynomial Calibration

| Metric | Average |
|---------|---------|
| MSE | 209.95 |
| PSNR | 22.08 dB |
| SSIM | 0.792 |
| FoV Retention | *100%* |

### Spherical Re-projection

| Metric | Average |
|---------|---------|
| MSE | 1570.34 |
| PSNR | 17.56 dB |
| SSIM | 0.725 |
| FoV Retention | 92.77% |

*Key Findings*

- Polynomial Calibration achieved superior geometric accuracy.
- Spherical Re-projection preserved perceptual realism without requiring calibration.
- The two methods demonstrate a trade-off between numerical precision and flexibility.


---

##  Applications

- Autonomous Vehicles (ADAS)
- Robotics
- Visual SLAM
- Image Rectification
- Panoramic Imaging
- Surveillance Systems
- 3D Reconstruction
- AR/VR Vision Systems

---

##  Repository Structure


├── dataset/
├── polynomial_calibration/
├── spherical_reprojection/
├── results/
├── figures/
├── notebooks/
├── requirements.txt
└── README.md


---

##  Future Work

Possible improvements include:

- Hybrid correction framework combining both methods
- Deep Learning-based distortion correction
- Region-adaptive distortion models
- Automatic camera calibration
- Real-time video distortion correction
- Temporal consistency for video processing


---

##  Research Paper

*Evaluation of Distortion Correction Techniques for Fisheye Camera Images*

Authors:

- Palak Tiwari
- Dr. Muralikrishna S. N.
- Dr. Raghurama Holla

Manipal Institute of Technology, Manipal Academy of Higher Education


---

##  Citation

If you use this work in your research, please cite the associated paper.

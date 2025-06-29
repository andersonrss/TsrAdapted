import open3d as o3d
from PIL import Image
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim

from math import log10, sqrt
import cv2
import numpy as np

import argparse
import os
import csv

# -----------------------------------------------------------------------------
# This script compares pairs of rendered and reference images using PSNR, SSIM,
# and LPIPS metrics. It processes all image pairs from two directories, computes
# the metrics for each pair, prints the results, and saves them in a CSV file.
# -----------------------------------------------------------------------------


# LPIPS calculation
def LPIPS(img_path_1, img_path_2):
    img1 = np.array(Image.open(img_path_1).convert("RGB"))
    img2 = np.array(Image.open(img_path_2).convert("RGB"))

    if img1.shape != img2.shape:
        img2 = np.array(Image.fromarray(img2).resize((img1.shape[1], img1.shape[0])))

    loss_fn = lpips.LPIPS(net='alex')
    def prep(im): return torch.tensor(im).permute(2,0,1).unsqueeze(0).float()/127.5 - 1

    im1 = prep(img1)
    im2 = prep(img2)

    with torch.no_grad():
        lp = loss_fn(im1, im2).item()

    return lp
    
# PSNR calculation
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):   # MSE = 0 -> no noise is present in the signal
        return 100  # Therefore PSNR have no importance
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# SSIM calculation
def SSIM(original, compressed):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    return ssim(original, compressed)

# Function to get image pairs from directories
def get_image_pairs(ref_dir, obj_dir):
    ref_images = sorted([f for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    obj_images = sorted([f for f in os.listdir(obj_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    pairs = []
    for ref, obj in zip(ref_images, obj_images):
        pairs.append((os.path.join(ref_dir, ref), os.path.join(obj_dir, obj)))
    return pairs

if __name__ == "__main__":

    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Mesh rendering and image comparison.")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory with reference images")
    parser.add_argument("--obj_dir", type=str, required=True, help="Directory with object rendered images")
    parser.add_argument("--csv_path", type=str, default="metrics_manual_render.csv", help="CSV file to save metrics")
    args = parser.parse_args()

    image_pairs = get_image_pairs(args.ref_dir, args.obj_dir)

    results = []

    # Process each pair of images
    for idx, (image_reference, obj_image) in enumerate(image_pairs[:10]):
        print(f"\n--- Pair {idx+1}: {os.path.basename(image_reference)} vs {os.path.basename(obj_image)} ---")

        # Load images
        original = cv2.imread(image_reference)
        compressed = cv2.imread(obj_image, 1)

        # Check images dimensions and resize if necessary
        if original.shape != compressed.shape:
            compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))

        # Metrics calculations
        psnr_value = PSNR(original, compressed)
        print(f"PSNR value is {psnr_value} dB")

        ssim_value = SSIM(original, compressed)
        print(f"SSIM value is {ssim_value}")

        lpips_value = LPIPS(image_reference, obj_image)
        print(f"LPIPS value is {lpips_value}")

        results.append({
            "pair": f"{os.path.basename(image_reference)} vs {os.path.basename(obj_image)}",
            "psnr": psnr_value,
            "ssim": ssim_value,
            "lpips": lpips_value
        })

    # Calculate mean values
    mean_psnr = np.mean([r["psnr"] for r in results])
    mean_ssim = np.mean([r["ssim"] for r in results])
    mean_lpips = np.mean([r["lpips"] for r in results])

    # CSV output
    with open(args.csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Pair", "PSNR", "SSIM", "LPIPS"])
        for r in results:
            writer.writerow([r["pair"], r["psnr"], r["ssim"], r["lpips"]])
        writer.writerow([])
        writer.writerow(["Med", mean_psnr, mean_ssim, mean_lpips])

    # Path to the CSV file 
    print(f"\nResults saved in {args.csv_path}")
    print('\n')
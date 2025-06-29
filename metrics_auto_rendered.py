import open3d as o3d
from PIL import Image
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

from math import log10, sqrt
import cv2
import numpy as np

import argparse
import os
import csv


# .obj rendering function using Open3D in order to generate an image from a 3D mesh
def render_obj_from_view(obj_path, img_reference_path, output_img_path):
    img_ref = Image.open(img_reference_path).convert("RGB")
    h, w = img_ref.size[1], img_ref.size[0]

    # Computing vertex normals and rendering the mesh
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=w, height=h)
    vis.add_geometry(mesh)

    ctr = vis.get_view_control()
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    # Visualization adjustments (due to some inconsistencies in Open3D)
    ctr.set_lookat(center)
    ctr.set_front([0.4, -4, 2])
    ctr.set_up([0, 3, 0])
    ctr.rotate(0.0, -10.0)
    ctr.set_zoom(0.7)

    # Rendering and saving the image
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_img_path)
    vis.destroy_window()

    return output_img_path

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

# Function to get pairs of .obj files and reference images
def get_pairs(obj_dir, ref_dir):
    obj_files = sorted([f for f in os.listdir(obj_dir) if f.lower().endswith('.obj')])
    ref_images = sorted([f for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    pairs = []
    for obj, ref in zip(obj_files, ref_images):
        pairs.append((os.path.join(obj_dir, obj), os.path.join(ref_dir, ref)))
    return pairs

if __name__ == "__main__":

    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Mesh rendering and image comparison for multiple pairs.")
    parser.add_argument("--obj_dir", type=str, required=True, help="Directory with .obj files")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory with reference images")
    parser.add_argument("--out_dir", type=str, default="renders", help="Directory to save rendered images")
    parser.add_argument("--csv_path", type=str, default="metrics_auto_render.csv", help="CSV file to save metrics")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = get_pairs(args.obj_dir, args.ref_dir)

    results = []

    # Process each pair of .obj file and reference image
    for idx, (obj_file, image_reference) in enumerate(pairs[:10]):
        output_img = os.path.join(args.out_dir, f"render_{idx+1}.png")
        print(f"\n--- Pair {idx+1}: {os.path.basename(obj_file)} vs {os.path.basename(image_reference)} ---")

        render_obj_from_view(obj_file, image_reference, output_img)

        # Load images
        original = cv2.imread(image_reference)
        compressed = cv2.imread(output_img, 1)

        # Check images dimensions and resize if necessary
        if original.shape != compressed.shape:
            compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))

        # Metrics calculations
        psnr_value = PSNR(original, compressed)
        print(f"PSNR value is {psnr_value} dB")

        ssim_value = SSIM(original, compressed)
        print(f"SSIM value is {ssim_value}")

        lpips_value = LPIPS(image_reference, output_img)
        print(f"LPIPS value is {lpips_value}")

        results.append({
            "pair": f"{os.path.basename(obj_file)} vs {os.path.basename(image_reference)}",
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
        writer.writerow(["Média", mean_psnr, mean_ssim, mean_lpips])

    print(f"\nResults saved in {args.csv_path}")

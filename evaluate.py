import os
import argparse
import numpy as np
import torch
import cv2
from model import NAFNetRCANPipeline

def run_evaluation(input_dir, output_dir, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = NAFNetRCANPipeline(in_channels=1, num_features=64, scale_factor=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

    print(f"Evaluating {len(files)} files from '{input_dir}'...")

    for fname in files:
        file_path = os.path.join(input_dir, fname)
        arr = np.load(file_path).astype(np.float32)

        # Normalize to [0, 1] if required
        tensor_in = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(tensor_in)

        # Convert back to NumPy array
        # Convert back to NumPy array
        output_arr = output_tensor.squeeze().cpu().numpy()
        output_arr = np.clip(output_arr, 0.0, 1.0)

        # Save output as .npy file
        npy_save_path = os.path.join(output_dir, fname)
        np.save(npy_save_path, output_arr)

        # Save visual comparison as .png
        png_save_path = os.path.join(output_dir, fname.replace(".npy", ".png"))
        img_visual = (output_arr * 255.0).astype(np.uint8)
        cv2.imwrite(png_save_path, img_visual)

        print(f"Saved: {npy_save_path} & {png_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semiconductor Image Restoration Inference")
    parser.add_argument("--input_dir", type=str, default="LRnoise", help="Path to degraded input folder")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Path to save outputs")
    parser.add_argument("--weights", type=str, default="saved_models/best_nafnet_rcan.pt", help="Model weights path")
    args = parser.parse_args()

    run_evaluation(args.input_dir, args.output_dir, args.weights)
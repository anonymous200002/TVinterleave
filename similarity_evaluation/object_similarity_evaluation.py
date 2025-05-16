from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch, os
import lpips
from glob import glob
import torchvision.transforms as T
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage import io
import numpy as np
import argparse


def compute_similarity(img1_path, img2_path, model, processor, transform, loss_fn, device):
    # 加载图像
    image1 = Image.open(img1_path).convert("RGB")
    image2 = Image.open(img2_path).convert("RGB")

    # === 1. CLIP  ===
    inputs = processor(images=[image1, image2], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        clip_similarity = (image_features[0] @ image_features[1].T).item()

    # === 2. Resize  LPIPS and SSIM ===
    target_size = (384, 384)
    image1_resized = image1.resize(target_size)
    image2_resized = image2.resize(target_size)

    # === 3. LPIPS  ===
    img1_tensor = transform(image1_resized).unsqueeze(0).to(device)
    img2_tensor = transform(image2_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        lpips_distance = loss_fn(img1_tensor, img2_tensor).item()
        lpips_similarity = 1 - lpips_distance  

    # === 4. SSIM  ===
    image1_gray = np.array(image1_resized.convert("L"))
    image2_gray = np.array(image2_resized.convert("L"))
    ssim_score = ssim(image1_gray, image2_gray)
    return clip_similarity, lpips_similarity, ssim_score



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate similarity based on object boxes or middle frames.")

    parser.add_argument(
        "--reference_root",
        type=str,
        required=True,
        help=(
            "Path to reference images.\n"
            "If evaluating object boxes, use the extracted path from 'rectangles.zip'.\n"
            "If evaluating middle frames, use the extracted path from 'middle_frames_for_sim_eval.zip'."
        )
    )

    parser.add_argument(
        "--result_root",
        type=str,
        required=True,
        help=(
            "Path to result folder containing images to be evaluated.\n"
            "Structure: result_root/<sample_id>/<clip_id>/*.jpg"
        )
    )

    return parser.parse_args()


def evaluate(reference_root, result_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    loss_fn = lpips.LPIPS(net='alex').to(device)  


    sample_scores_clip = []
    sample_scores_lpips = []
    sample_scores_ssim = []

    for sample_id in tqdm(sorted(os.listdir(result_root))):
        sample_path = os.path.join(result_root, sample_id)
        if not os.path.isdir(sample_path):
            continue

        ref_sample_root = os.path.join(reference_root, sample_id)
        if not os.path.exists(ref_sample_root):
            print(f"⚠️ Missing reference path: {ref_sample_root}")
            continue

        clip_scores_per_clip = []
        lpips_scores_per_clip = []
        ssim_scores_per_clip = []

        for ref_clip_name in sorted(os.listdir(ref_sample_root)):
            ref_clip_path = os.path.join(ref_sample_root, ref_clip_name)
            if not os.path.isdir(ref_clip_path):
                continue

            if ref_clip_name == "001_clip":
                print('overlook 001_clip')
                continue

            clip_idx = ref_clip_name.split("_")[0]
            clip_idx = f"{int(clip_idx):02d}"

            result_slice_path = os.path.join(sample_path, clip_idx)
            if not os.path.isdir(result_slice_path):
                print(f"⚠️ Missing corresponding detection slice: {result_slice_path}")
                continue

            ref_images = glob(os.path.join(ref_clip_path, "*.jpg"))
            crop_images = glob(os.path.join(result_slice_path, "*.jpg"))

            if not ref_images or not crop_images:
                continue

            max_clip = 0.0
            max_lpips = 0.0
            max_ssim = 0.0

            for crop_path in crop_images:
                for ref_path in ref_images:
                    clip_sim, lpips_dist, ssim_val = compute_similarity(crop_path, ref_path, model, processor, transform, loss_fn, device)
                    max_clip = max(max_clip, clip_sim)
                    max_lpips = max(max_lpips, lpips_dist)
                    max_ssim = max(max_ssim, ssim_val)

            clip_scores_per_clip.append(max_clip)
            lpips_scores_per_clip.append(max_lpips)
            ssim_scores_per_clip.append(max_ssim)

        if clip_scores_per_clip:
            score_clip = sum(clip_scores_per_clip) / len(clip_scores_per_clip)
            score_lpips = sum(lpips_scores_per_clip) / len(lpips_scores_per_clip)
            score_ssim = sum(ssim_scores_per_clip) / len(ssim_scores_per_clip)

            sample_scores_clip.append(score_clip)
            sample_scores_lpips.append(score_lpips)
            sample_scores_ssim.append(score_ssim)

            print(f"✅ sample {sample_id} - CLIP: {score_clip:.4f}, LPIPS: {score_lpips:.4f}, SSIM: {score_ssim:.4f}")

    # avg
    if sample_scores_clip:
        avg_clip = sum(sample_scores_clip) / len(sample_scores_clip)
        avg_lpips = sum(sample_scores_lpips) / len(sample_scores_lpips)
        avg_ssim = sum(sample_scores_ssim) / len(sample_scores_ssim)
        print(f"\n🌟 The average similarity of all samples - CLIP: {avg_clip:.4f}, LPIPS: {avg_lpips:.4f}, SSIM: {avg_ssim:.4f}")
    else:
        print("❌ No valid sample score can be calculated。")

    return avg_clip, avg_lpips, avg_ssim

def main():
    args = parse_args()
    evaluate(args.reference_root, args.result_root)

if __name__ == "__main__":
    main()


import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, required=True)
args = parser.parse_args()
img_id = f"{args.id:05d}"
id_str4 = f"{args.id:04d}"

source_img_path = f"results/{img_id}/img{img_id}_edit_eyes_source.png"
source_mask_path = f"data/labels_base/label{id_str4}.pt"
edit_mask_path = f"data/test_image/images_test_eyes/{img_id}/seg{id_str4}.png"
result_img_path = f"results/{img_id}/img{img_id}_edit_eyes.png"

source_img = Image.open(source_img_path).convert("RGB")
mask_tensor = torch.load(source_mask_path, weights_only=True)
source_mask = mask_tensor.cpu().numpy()
edit_mask = Image.open(edit_mask_path).convert("RGB")
result_img = Image.open(result_img_path).convert("RGB")

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0, 0].imshow(source_img)
axs[0, 0].set_title("Source Face")
axs[0, 0].axis("off")

axs[0, 1].imshow(source_mask, cmap="tab20")
axs[0, 1].set_title("Source Mask")
axs[0, 1].axis("off")

axs[1, 0].imshow(edit_mask)
axs[1, 0].set_title("Edit Mask")
axs[1, 0].axis("off")

axs[1, 1].imshow(result_img)
axs[1, 1].set_title("Result Image")
axs[1, 1].axis("off")

plt.tight_layout()
plt.savefig(f"results/{img_id}/visualization_{img_id}.png")
plt.close()


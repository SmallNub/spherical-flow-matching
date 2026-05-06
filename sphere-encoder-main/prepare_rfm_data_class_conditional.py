import torch
import os

INPUT = "workspace/experiments/sphere-small-small-cifar-10-32px/encoding/encoded_dataset.pt"
OUTPUT_DIR = "workspace/experiments/sphere-small-small-cifar-10-32px/encoding/rfm_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = torch.load(INPUT)

Z = data["encodings"].float()
labels = data["labels"]
split_ids = data["split_ids"]

assert labels is not None, "Need labels for class-conditional RFM data."

def normalize(z):
    N = z.shape[0]
    z = z.reshape(N, -1)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return z

classes = torch.unique(labels).tolist()

for c in classes:
    class_mask = labels == c
    Z_c = Z[class_mask]

    N = Z_c.shape[0]
    # perm = torch.randperm(N)
    # Z_c = Z_c[perm]

    train = Z_c[: int(0.8 * N)]
    val = Z_c[int(0.8 * N): int(0.9 * N)]
    test = Z_c[int(0.9 * N):]

    train = normalize(train)
    val = normalize(val)
    test = normalize(test)

    Z_all = torch.cat([train, val, test], dim=0)

    torch.save(Z_all, os.path.join(OUTPUT_DIR, f"class_{int(c)}.pt"))

    print(f"class {int(c)}")
    print("all:", Z_all.shape)
    print("train:", train.shape)
    print("val:", val.shape)
    print("test:", test.shape)

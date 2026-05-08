import torch
import os

INPUT = "workspace/experiments/sphere-small-small-cifar-10-32px/encoding/encoded_dataset.pt"
OUTPUT_DIR = "workspace/experiments/sphere-small-small-cifar-10-32px/encoding/rfm_data"
CLASS_CONDITIONAL = False  # If True, creates separate .pt files per class. If False, creates a single all.pt file.

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = torch.load(INPUT)

Z = data["encodings"].float()  # IMPORTANT: cast to float32
labels = data["labels"]
split_ids = data["split_ids"]

# -------------------------------------------------
# CASE 1: splits exist (your script)
# -------------------------------------------------
if split_ids is not None:
    train_full = Z[split_ids == 0]
    train = train_full[:-len(train_full)//10]
    val = train_full[-len(train_full)//10:]
    test = Z[split_ids == 1]

# -------------------------------------------------
# CASE 2: no splits → create them
# -------------------------------------------------
else:
    N = Z.shape[0]
    train = Z[: int(0.8 * N)]
    val = Z[int(0.8 * N): int(0.9 * N)]
    test = Z[int(0.9 * N):]


# -------------------------------------------------
# CRITICAL: enforce sphere constraint again
# -------------------------------------------------
def normalize(z):
    N, T, D = z.shape  # e.g. 256, 4
    z = z.reshape(N, T * D)   # [N, 1024]
    z = z / z.norm(dim=-1, keepdim=True)
    return z


train = normalize(train)
val = normalize(val)
test = normalize(test)

# -------------------------------------------------
# SAVE
# -------------------------------------------------

if CLASS_CONDITIONAL:
    classes = torch.unique(labels).tolist()

    for c in classes:
        class_mask = labels == c
        train_c = train[class_mask[:len(train)]]
        val_c = val[class_mask[len(train):len(train)+len(val)]]
        test_c = test[class_mask[len(train)+len(val):]]

        Z_all = torch.cat([train_c, val_c, test_c], dim=0)
        torch.save(Z_all, os.path.join(OUTPUT_DIR, f"class_{int(c)}.pt"))

else:
    torch.save(Z, os.path.join(OUTPUT_DIR, "all.pt"))

print("Saved:", Z.shape)

print("Saved RFM dataset:")
print("train:", train.shape)
print("val:", val.shape)
print("test:", test.shape)

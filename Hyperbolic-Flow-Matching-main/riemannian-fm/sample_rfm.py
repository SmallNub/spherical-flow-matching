import torch
from manifm.model_pl import ManifoldFMLitModule
from omegaconf import OmegaConf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. LOAD MODEL
cfg = OmegaConf.load("configs/train.yaml")
ckpt_path = "outputs/runs/euclidean/fm/2026.05.04/145358/checkpoints/epoch-021_step-0_loss-0.0000.ckpt"

model = ManifoldFMLitModule.load_from_checkpoint(
    ckpt_path,
    cfg=cfg
).to(DEVICE)
model.eval()

# Use the manifold and dim defined in the model/config
manifold = model.manifold
dim = model.dim

# 2. VECTOR FIELD ACCESS
# Using model.vecfield(t, x) directly is safer because it handles 
# the EMA, Unbatch, and ProjectToTangent logic correctly.
def get_v(x, t_val):
    # model.vecfield expects (t, x) based on the training loop logic
    t = torch.full((x.shape[0], 1), t_val, device=DEVICE)
    return model.vecfield(t, x)

# 3. FLOW SAMPLING
@torch.no_grad()
def sample_flow(n_samples=10000, steps=100):
    # Use the manifold's own base sampling to ensure correct shape/distribution
    z = manifold.random_base(n_samples, dim).to(DEVICE)
    
    # Standard Euler integration on the manifold
    dt = 1.0 / steps
    
    for i in range(steps):
        current_t = i / steps
        v = get_v(z, current_t)
        
        # Euler Step
        z = z + dt * v
        
        # Projection step: Crucial for staying on the manifold (e.g., S2)
        z = manifold.projx(z)
        
    return z

# 4. GENERATE
Z_gen = sample_flow(n_samples=50000, steps=100)

# 5. SAVE
torch.save({
    "encodings": Z_gen.cpu(),
    "labels": None,
    "split_ids": None,
    "split_names": ["generated"],
}, "../../sphere-encoder-main/workspace/experiments/sphere-small-small-cifar-10-32px/encoding/generated_encodings.pt")

print("Saved:", Z_gen.shape)
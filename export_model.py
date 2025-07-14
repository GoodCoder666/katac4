import numpy as np
import torch
from scipy.stats import t

from model import Net

model_path = './weights/b3c128nbt_2025-05-24_20-47-22/katac4_b3c128nbt_29000.pth'

device = torch.device('cpu')
net = Net().eval()
net.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

model = torch.jit.script(net)
model = torch.jit.freeze(model)

torch.jit.save(model, './saiblo/model.pt')

ALPHA = 1e-5
Z_ENTRIES = 1000
OUT_FILE = './saiblo/z_lookup.npy'

Z = np.empty(Z_ENTRIES, dtype=np.float32)
for df in range(1, Z_ENTRIES + 1):
    Z[df - 1] = t.isf(ALPHA, df)

np.save(OUT_FILE, Z, allow_pickle=False)

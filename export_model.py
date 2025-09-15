import numpy as np
import torch
from scipy.stats import t

from model import Net

model_path = './weights/b3c128nbt_2025-08-18_19-22-00/katac4_b3c128nbt_30000.pth'

def load_model(path):
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    state_dict['policy_head.conv2.conv.weight'] = \
        state_dict['policy_head.conv2.conv.weight'][0:1]
    net = Net(c_policy=1)
    net.load_state_dict(state_dict)
    return net

device = torch.device('cpu')
net = load_model(model_path).eval().to(device)

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

import torch
from model import Net

input_file = './weights/b3c128nbt_2025-08-18_19-22-00/katac4_b3c128nbt_30000.pth'
output_file = './model.onnx'

def load_model(path):
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    state_dict['policy_head.conv2.conv.weight'] = \
        state_dict['policy_head.conv2.conv.weight'][0:1]
    net = Net(c_policy=1)
    net.load_state_dict(state_dict)
    return net

model = load_model(input_file).eval()

dummy_input = torch.empty(1, 6, 9, 9)

torch.onnx.export(
    model,
    dummy_input,
    output_file,
    input_names=['input'],
    output_names=['policy', 'value'],
    dynamic_axes={
        'input': {2: 'height', 3: 'width'},
        'policy': {2: 'height', 3: 'width'},
    },
    opset_version=14
)

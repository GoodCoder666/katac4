"""
References:
- https://arxiv.org/pdf/1902.10565v5
- https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
- https://github.com/lightvector/KataGo/blob/master/python/katago/train/model_pytorch.py
- https://github.com/shindavid/AlphaZeroArcade/blob/main/py/shared/net_modules.py
"""


from torch import nn
import torch
import torch.nn.functional as F


class KataGPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        width_scale = (x.size(3) - 10.5) / 3
        g_mean = torch.mean(x, dim=(2, 3))
        g_max, _ = torch.max(x.flatten(start_dim=2), dim=-1)
        return torch.cat([g_mean, g_mean * width_scale, g_max], dim=1)


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.norm = nn.BatchNorm2d(c_in)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False)

    def forward(self, x):
        out = x
        out = F.relu(self.norm(out))
        out = self.conv(out)
        return out


class ConvBlockWithGPool(nn.Module):
    def __init__(self, c_in: int, c_out: int, c_gpool: int):
        super().__init__()
        self.norm = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True)
        )
        self.conv_r = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.pool = nn.Sequential(
            nn.Conv2d(c_in, c_gpool, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_gpool),
            nn.ReLU(inplace=True),
            KataGPool(),
            nn.Linear(3 * c_gpool, c_out, bias=False)
        )

    def forward(self, x):
        x = self.norm(x)
        out_r = self.conv_r(x)
        out_g = self.pool(x)[..., None, None]
        return out_r + out_g


class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_gpool=None):
        super().__init__()
        if c_gpool:
            c_mid -= c_gpool
            self.conv1 = ConvBlockWithGPool(c_in, c_mid, c_gpool)
        else:
            self.conv1 = ConvBlock(c_in, c_mid)
        self.conv2 = ConvBlock(c_mid, c_in)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class Bottlenest(nn.Module):
    # https://raw.githubusercontent.com/lightvector/KataGo/master/images/docs/bottlenecknestedresblock.png
    def __init__(self, c_in, c_gpool=None):
        super().__init__()
        c_mid = c_in // 2
        self.bottlenest = nn.Sequential(
            ConvBlock(c_in, c_mid, kernel_size=1, padding=0),
            ResBlock(c_mid, c_mid, c_gpool),
            ResBlock(c_mid, c_mid),
            ConvBlock(c_mid, c_in, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return x + self.bottlenest(x)


class PolicyHead(nn.Module):
    def __init__(self, c_policy, c_in, c_head):
        super().__init__()
        self.conv1 = ConvBlockWithGPool(c_in, c_head, c_head)
        self.conv2 = ConvBlock(c_head, c_policy, kernel_size=1, padding=0)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class ValueHead(nn.Module):
    def __init__(self, c_in, c_head):
        super().__init__()
        self.conv = ConvBlock(c_in, c_head)
        self.pool = KataGPool()
        self.linear = nn.Linear(3 * c_head, 3)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        return self.linear(x)


class Net(nn.Module):
    """
    Policy-value network module.
    """
    def __init__(self, c_policy=1, c_trunk=128, c_gpool=32, c_head=32):
        super().__init__()

        # common layers
        self.input_conv = nn.Conv2d(6, c_trunk, kernel_size=3, stride=1, padding=1)
        self.trunk = nn.Sequential(
            Bottlenest(c_trunk),
            Bottlenest(c_trunk, c_gpool),
            Bottlenest(c_trunk)
        )

        # policy & value heads
        self.policy_head = PolicyHead(c_policy, c_trunk, c_head)
        self.value_head = ValueHead(c_trunk, c_head)

    def forward(self, state_input):
        x = self.input_conv(state_input)
        x = self.trunk(x)
        policy = self.policy_head(x).unbind(dim=1)
        value = self.value_head(x)
        return policy, value

    def export_state_dict(self, policy_channels):
        state_dict = self.state_dict()
        state_dict['policy_head.conv2.conv.weight'] = \
            state_dict['policy_head.conv2.conv.weight'][policy_channels]
        return state_dict


class InferenceGraph:
    """
    Wrapper around the model, enabling CUDA graph inference.
    """
    def __init__(self, net, device, board_height, board_width):
        self.net = net.to(device).eval()
        self.device = device
        self._graph = torch.cuda.CUDAGraph()
        self._state = torch.zeros(1, 6, board_height, board_width, dtype=torch.float32, device=device)

        with torch.inference_mode():
            # warmup
            self._stream = torch.cuda.Stream(device)
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                for _ in range(3):
                    self.net.forward(self._state)
            torch.cuda.current_stream().wait_stream(self._stream)

            # capture
            with torch.cuda.graph(self._graph, stream=self._stream):
                (policy_logits,), value_logits = self.net(self._state)
                self._value = F.softmax(value_logits.squeeze(0), dim=0)
                self._policy_logits = policy_logits.squeeze(0)

    def __del__(self):
        self._graph.reset()

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def policy_value_fn(self, game, policy_temp=1.0):
        with torch.inference_mode():
            self._state[0, :, :, :] = torch.tensor(game.state(), dtype=torch.float32, device=self.device)
            self._graph.replay()
            sensible_moves = game.sensible_moves()
            policy_logits = self._policy_logits[game.top[sensible_moves], sensible_moves]
            policy = F.softmax(policy_logits / policy_temp, dim=0).cpu().numpy()
            win_rate, loss_rate, _ = self._value.tolist()
            value = win_rate - loss_rate
        return sensible_moves, policy, value

import copy
import json
import math
import random
import time
from array import array
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass

from scipy import stats
import torch
import torch.nn.functional as F

from kivy.config import Config

Config.set('input', 'mouse', 'mouse,disable_multitouch')

from kivy.app import App
from kivy.clock import Clock
from kivy.core.text import Label as CoreLabel
from kivy.core.text.markup import MarkupLabel
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, Line, Rectangle, RoundedRectangle
from kivy.properties import NumericProperty, StringProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget

from kivy_garden.graph import Graph, MeshLinePlot

from model import InferenceGraph, Net
from saiblo.game import ConnectFour
from plyer import filechooser

Window.size = (1400, 800)

@dataclass
class Configuration:
    """Application configuration settings."""
    model_path: str = './weights/b3c128nbt_2025-08-18_19-22-00/katac4_b3c128nbt_30000.pth'
    use_gpu: bool = True
    n_playout: int = 1000
    c_puct: float = 1.1
    c_fpu: float = 0.2
    player_names: dict = None

    color_bg: tuple = (0.08, 0.09, 0.11, 1)        # Midnight Blue-Black
    color_board: tuple = (0.13, 0.15, 0.18, 1)     # Dark Slate
    color_grid: tuple = (0.25, 0.28, 0.32, 1)      # Soft Steel
    color_p1: tuple = (0.95, 0.75, 0.25, 1)        # Soft Gold
    color_p2: tuple = (0.55, 0.35, 0.9, 1)         # Vibrant Indigo
    color_forbidden: tuple = (0.85, 0.25, 0.25, 1) # Soft Red
    color_highlight: tuple = (1, 1, 1, 0.1)
    color_lcb: tuple = (0, 0.9, 0.9, 1)            # Cyan

    def __post_init__(self):
        if self.player_names is None:
            self.player_names = {1: "Orange", -1: "Purple"}

config = Configuration()


class GameMode(str, Enum):
    PLAY = 'play'
    ANALYSIS = 'analysis'


def win_rate_for_player(q_value: float, player_to_move: int, perspective_player: int) -> float:
    """Convert a Q-value into a win probability for the given perspective player."""
    q_value *= player_to_move * perspective_player
    return 0.5 * (q_value + 1.0)


def orange_win_rate(q_value: float, player_to_move: int) -> float:
    """Specialized helper that reports Orange's (player=1) win probability."""
    return win_rate_for_player(q_value, player_to_move, perspective_player=1)

def compute_board_geometry(widget, rows, cols, padding=0.9):
    if widget.height == 0: return 0, 0, 0, 0, 0

    board_aspect = cols / rows
    screen_aspect = widget.width / widget.height

    if screen_aspect > board_aspect:
        h = widget.height * padding
        w = h * board_aspect
    else:
        w = widget.width * padding
        h = w / board_aspect

    off_x = widget.x + (widget.width - w) / 2
    off_y = widget.y + (widget.height - h) / 2
    cell_size = w / cols
    return off_x, off_y, w, h, cell_size

def draw_piece(cx, cy, size, color):
    r, g, b, a = color
    Color(0, 0, 0, 0.4)
    Ellipse(pos=(cx + size*0.05, cy - size*0.05), size=(size, size))

    Color(r*0.8, g*0.8, b*0.8, a)
    Ellipse(pos=(cx, cy), size=(size, size))

    Color(r, g, b, a)
    Ellipse(pos=(cx + size*0.05, cy + size*0.1), size=(size*0.9, size*0.9))

    Color(1, 1, 1, 0.3)
    Ellipse(pos=(cx + size*0.25, cy + size*0.55), size=(size*0.3, size*0.2))

def draw_board_grid_and_pieces(rows, cols, off_x, off_y, cs, forbidden_point=None, board=None):
    for r in range(rows):
        for c in range(cols):
            cx = off_x + c * cs
            cy = off_y + r * cs

            Color(*config.color_grid)
            Line(rectangle=(cx, cy, cs, cs), width=1)

            val = 0
            if board is not None:
                val = board[r, c]

            is_forbidden = (forbidden_point is not None and (r, c) == forbidden_point)

            if val == 1:
                padding = cs * 0.1
                sz = cs - 2*padding
                draw_piece(cx + padding, cy + padding, sz, config.color_p1)
            elif val == -1:
                padding = cs * 0.1
                sz = cs - 2*padding
                draw_piece(cx + padding, cy + padding, sz, config.color_p2)
            elif is_forbidden:
                Color(*config.color_forbidden)
                m = cs * 0.25
                Line(points=[cx+m, cy+m, cx+cs-m, cy+cs-m], width=2)
                Line(points=[cx+m, cy+cs-m, cx+cs-m, cy+m], width=2)

class RoundedShapeMixin:
    def update_shape(self, bg_color_normal, bg_color_down, radius=[6]):
        self.canvas.before.clear()
        with self.canvas.before:
            if self.state == 'down':
                c = bg_color_down or (bg_color_normal[0]*0.7, bg_color_normal[1]*0.7, bg_color_normal[2]*0.7, bg_color_normal[3])
                Color(*c)
            else:
                Color(*bg_color_normal)
            RoundedRectangle(pos=self.pos, size=self.size, radius=radius)

class RoundedButton(Button, RoundedShapeMixin):
    def __init__(self, **kwargs):
        self.bg_color = kwargs.pop('bg_color', (0.25, 0.3, 0.35, 1))
        super().__init__(**kwargs)
        self.background_color = (0, 0, 0, 0)
        self.bind(pos=self.redraw, size=self.redraw, state=self.redraw)

    def redraw(self, *args):
        self.update_shape(self.bg_color, None)

class RoundedToggleButton(ToggleButton, RoundedShapeMixin):
    def __init__(self, **kwargs):
        self.bg_normal = kwargs.pop('bg_normal', (0.2, 0.22, 0.25, 1))
        self.bg_down = kwargs.pop('bg_down', (0.2, 0.6, 0.8, 1))
        super().__init__(**kwargs)
        self.background_color = (0, 0, 0, 0)
        self.bind(pos=self.redraw, size=self.redraw, state=self.redraw)

    def redraw(self, *args):
        self.update_shape(self.bg_normal, self.bg_down)

# --- Inference Helper ---
class CPUInference:
    def __init__(self, net, height, width):
        dummy_input = torch.empty(1, 6, height, width, device='cpu', dtype=torch.float32)
        net = torch.jit.trace(net.to('cpu'), dummy_input)
        self.net = torch.jit.freeze(net)

    def policy_value_fn(self, game):
        with torch.inference_mode():
            state_tensor = torch.FloatTensor(game.state()).unsqueeze(0)
            (policy_logits,), value_logits = self.net(state_tensor)

            value_probs = F.softmax(value_logits.squeeze(0), dim=0)
            policy_logits = policy_logits.squeeze(0)

            sensible_moves = game.sensible_moves()
            relevant_logits = policy_logits[game.top[sensible_moves], sensible_moves]

            policy = F.softmax(relevant_logits, dim=0).numpy()
            win_rate, loss_rate, _ = value_probs.tolist()
            value = win_rate - loss_rate
        return sensible_moves, policy, value

def load_model(path):
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    if 'policy_head.conv2.conv.weight' in state_dict:
        state_dict['policy_head.conv2.conv.weight'] = \
            state_dict['policy_head.conv2.conv.weight'][0:1]
    net = Net(c_policy=1)
    net.load_state_dict(state_dict)
    return net

class LazyZTable:
    def __init__(self, size=1000, p=1e-5):
        self._size = size
        self._p = p
        self._cache = {}

    def __getitem__(self, idx):
        if val := self._cache.get(idx):
            return val
        val = stats.t.isf(self._p, idx + 1)
        self._cache[idx] = val
        return val

    def __len__(self):
        return self._size

# --- Core MCGS Logic ---
class TreeNode:
    __slots__ = ('children', 'edge_N', 'edge_P', 'N', 'Q', 'avg_Q2', 'U')

    def __init__(self):
        self.children = []       # List[TreeNode | None]
        self.edge_N = array('I') # Array of unsigned int
        self.edge_P = array('f') # Array of float
        self.N = 0
        self.Q = 0.0
        self.avg_Q2 = 0.0
        self.U = 0.0

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, policy_fn, state):
        acts, probs, value = policy_fn(state)
        width = state.width

        self.children = [None] * width
        zeros = b'\x00' * 4 * width
        self.edge_N = array('I', zeros)
        self.edge_P = array('f', zeros)

        for a, p in zip(acts, probs):
            if a < width:
                self.edge_P[a] = float(p)

        self.Q = self.U = value

    def select(self, c_puct, c_fpu):
        best_action = -1
        best_uct = -float('inf')

        # Sum edge_P where edge_N > 0 for FPU calculation
        p_explored = 0.0
        for i in range(len(self.children)):
            if self.edge_N[i] > 0:
                p_explored += self.edge_P[i]

        fpu_penalty = c_fpu * math.sqrt(p_explored)

        if self.N > 2:
            k = 4 * math.sqrt(self.var) / self.N
            k = min(max(k, 0.5), 1.4)
            child_N_sum = sum(self.edge_N)
            alpha = 1.0 / (1 + math.sqrt(child_N_sum / 10000.0))
            c_puct *= (alpha * k + (1.0 - alpha))

        sqrt_N = math.sqrt(self.N)

        for i in range(len(self.children)):
            if self.edge_P[i] <= 1e-8:
                continue

            child = self.children[i]
            n_edge = self.edge_N[i]

            edge_Q = self.Q - fpu_penalty if child is None else -child.Q
            uct = edge_Q + c_puct * self.edge_P[i] * sqrt_N / (1 + n_edge)

            if uct > best_uct:
                best_uct = uct
                best_action = i

        return best_action, self.children[best_action], self.edge_N[best_action]

    def update(self):
        # Terminal/Leaf Node: Q is static U (no children to average)
        if not self.children:
            self.Q = self.U
            self.avg_Q2 = self.U ** 2
            return

        sum_q = 0.0
        sum_q2 = 0.0

        for i in range(len(self.children)):
            n = self.edge_N[i]
            if n > 0:
                if child := self.children[i]:
                    sum_q += child.Q * n
                    sum_q2 += child.avg_Q2 * n

        self.Q = (self.U - sum_q) / self.N
        self.avg_Q2 = (self.U * self.U + sum_q2) / self.N

    @property
    def var(self):
        return max(0.0, self.avg_Q2 - self.Q ** 2)

class MCGS:
    __slots__ = ('c_puct', 'c_fpu', 'policy', 'root', 'nodes_by_hash', 'z_table')

    def __init__(self, policy_value_fn, z_table, c_puct=1.0, c_fpu=0.2):
        self.c_puct = c_puct
        self.c_fpu = c_fpu
        self.policy = policy_value_fn
        self.root = None
        self.nodes_by_hash = defaultdict(TreeNode)
        self.z_table = z_table

    def playout(self, state):
        state = copy.deepcopy(state)
        if self.root is None:
            self.root = self.nodes_by_hash[state.hash]
        node = self.root
        path = []

        # Descent
        while node.is_expanded():
            path.append(node)
            # Use c_fpu=0 at root for exploration, regular c_fpu elsewhere
            eff_fpu = 0.0 if node is self.root else self.c_fpu
            action, child, edge_N = node.select(self.c_puct, eff_fpu)
            state.step(action)

            if child is None:
                child = self.nodes_by_hash[state.hash]
                node.children[action] = child

            node.edge_N[action] += 1

            node = child

        # Leaf Evaluation
        if state.is_terminal():
            node.Q = node.U = -1.0 if state.winner else 0.0
        else:
            node.expand(self.policy, state)

        node.avg_Q2 = node.Q ** 2
        node.N += 1

        # Backprop
        for node in reversed(path):
            node.N += 1
            node.update()

    def reroot(self, state):
        self.root = self.nodes_by_hash[state.hash]

    def get_best_move(self):
        # LCB Selection logic
        root_node = self.root
        if not root_node: return None

        N_min = max(math.ceil(0.1 * root_node.N), 2)

        cands = []
        for act in range(len(root_node.children)):
            child = root_node.children[act]
            # Must check if child exists and meets threshold
            if child and child.N >= N_min:
                cands.append((act, child))

        if not cands:
            # Fallback to Max Visits
            max_visits = -1
            best_action = None
            for act in range(len(root_node.children)):
                edge_N = root_node.edge_N[act]
                if edge_N > max_visits:
                    max_visits = edge_N
                    best_action = act
            return best_action
        else:
            # LCB Metric
            def lcb(item):
                _, child = item
                var = child.var * (child.N / (child.N - 1)) if child.N > 1 else child.var
                idx = max(0, min(child.N, len(self.z_table)) - 1)
                t_val = self.z_table[idx]
                return -child.Q - t_val * math.sqrt(var) / child.N

            best_action, _ = max(cands, key=lcb)
            return best_action

# --- Session State ---
class GameSession:
    """Encapsulate match state, inference objects, and navigation logic."""

    def __init__(self, config: Configuration):
        self.config = config
        device_type = 'cuda' if self.config.use_gpu and torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_type)
        if device_type == 'cpu':
            torch.set_num_threads(4)
        self.net = load_model(self.config.model_path).to(self.device)
        self.net.eval()
        self.z_table = LazyZTable(p=1e-5)
        self.history = []
        self.win_history = []
        self.cursor = 0
        self.mcgs = None
        self._graph = None
        self.start_new_game()

    @property
    def current_game(self):
        return self.history[self.cursor] if 0 <= self.cursor < len(self.history) else None

    def start_new_game(self, height=None, width=None, forbidden_point=None):
        game = ConnectFour(height=height, width=width, forbidden_point=forbidden_point)
        self.history = [game]
        self.win_history = [(0, 0.5)]
        self.cursor = 0
        self._init_mcgs(game)

    def apply_move(self, col):
        current = self.current_game
        if current is None or current.is_terminal():
            return None

        if self.cursor < len(self.history) - 1:
            self.history = self.history[:self.cursor + 1]
            self.win_history = self.win_history[:self.cursor + 1]

        next_game = copy.deepcopy(self.history[-1])
        next_game.step(col)
        self.history.append(next_game)
        self.cursor = len(self.history) - 1
        self.win_history.append((self.cursor, 0.5))
        self.mcgs.reroot(next_game)
        return next_game

    def navigate(self, idx):
        if not (0 <= idx < len(self.history)):
            return False
        self.cursor = idx
        self.mcgs.reroot(self.history[idx])
        return True

    def load_rounds(self, height, width, forbidden_point, rounds):
        base_game = ConnectFour(height=height, width=width, forbidden_point=forbidden_point)
        self.history = [base_game]
        self.win_history = [(0, 0.5)]
        current = base_game
        for idx, move in enumerate(rounds, start=1):
            next_game = copy.deepcopy(current)
            next_game.step(move)
            self.history.append(next_game)
            self.win_history.append((idx, 0.5))
            current = next_game

        self.cursor = len(self.history) - 1
        self._init_mcgs(base_game)

        # Prime evaluations for each node in history
        for idx, game in enumerate(self.history):
            self.mcgs.reroot(game)
            self.mcgs.playout(game)
            if self.mcgs.root and self.mcgs.root.N > 0:
                q = self.mcgs.root.Q
                wr = orange_win_rate(q, game.player)
                self.win_history[idx] = (idx, wr)

        self.mcgs.reroot(self.current_game)

    def ensure_win_history_length(self, idx):
        while len(self.win_history) <= idx:
            self.win_history.append((len(self.win_history), 0.5))

    def set_winrate(self, idx, value):
        self.ensure_win_history_length(idx)
        self.win_history[idx] = (idx, value)

    def _init_mcgs(self, game):
        policy_fn = self._build_policy_fn(game)
        self.mcgs = MCGS(policy_fn, self.z_table, c_puct=self.config.c_puct, c_fpu=self.config.c_fpu)
        self.mcgs.reroot(game)

    def _build_policy_fn(self, game):
        if self.device.type == 'cuda':
            try:
                self._graph = InferenceGraph(self.net, self.device, game.height, game.width)
                return self._graph.policy_value_fn
            except Exception:
                self._graph = None
        cpu = CPUInference(self.net, game.height, game.width)
        return cpu.policy_value_fn

# --- UI Components ---

class PreviewBoardWidget(Widget):
    rows = NumericProperty(10)
    cols = NumericProperty(9)
    forbidden_x = NumericProperty(4)
    forbidden_y = NumericProperty(5) # User Row (0=Top)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        self.bind(rows=self.update_canvas, cols=self.update_canvas)
        self.bind(forbidden_x=self.update_canvas, forbidden_y=self.update_canvas)

    def on_touch_down(self, touch):
        if 'button' in touch.profile and touch.button != 'left':
            return False
        if not self.collide_point(*touch.pos): return False

        off_x, off_y, w, h, cs = compute_board_geometry(self, self.rows, self.cols, 0.95)

        if not (off_x <= touch.x <= off_x + w and off_y <= touch.y <= off_y + h):
            return False

        col = int((touch.x - off_x) / cs)
        row_engine = int((touch.y - off_y) / cs)

        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row_engine, self.rows - 1))
        row = self.rows - 1 - row

        self.forbidden_x = col
        self.forbidden_y = row
        return True

    def update_canvas(self, *args):
        self.canvas.clear()

        off_x, off_y, w, h, cs = compute_board_geometry(self, self.rows, self.cols, 0.95)

        with self.canvas:
            Color(*config.color_board)
            RoundedRectangle(pos=(off_x, off_y), size=(w, h), radius=[4])
            forbidden_r = (self.rows - 1) - self.forbidden_y
            draw_board_grid_and_pieces(self.rows, self.cols, off_x, off_y, cs,
                                       forbidden_point=(forbidden_r, self.forbidden_x))

class Card(BoxLayout):
    def __init__(self, **kwargs):
        self.bg_color = kwargs.pop('bg_color', (0.13, 0.15, 0.18, 1))
        self.radius = kwargs.pop('radius', [10])
        if 'padding' not in kwargs: kwargs['padding'] = 15
        if 'spacing' not in kwargs: kwargs['spacing'] = 10
        super().__init__(**kwargs)
        self.bind(pos=self.update_bg, size=self.update_bg)

    def update_bg(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self.bg_color)
            RoundedRectangle(pos=self.pos, size=self.size, radius=self.radius)

class SpinBox(BoxLayout):
    text = StringProperty('')
    values = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.spacing = 5

        self.btn_down = RoundedButton(text='<', bg_color=(0.25, 0.28, 0.32, 1),
                                      size_hint_x=None, width=35,
                                      font_size='14sp', font_name='Roboto-Bold')
        self.btn_down.bind(on_release=self.decrement)
        self.add_widget(self.btn_down)

        self.label = Label(text=self.text, font_size='16sp', color=(0.9, 0.9, 0.9, 1))
        self.label.bind(pos=self.update_rect, size=self.update_rect)
        self.add_widget(self.label)

        self.btn_up = RoundedButton(text='>', bg_color=(0.25, 0.28, 0.32, 1),
                                    size_hint_x=None, width=35,
                                    font_size='14sp', font_name='Roboto-Bold')
        self.btn_up.bind(on_release=self.increment)
        self.add_widget(self.btn_up)

        self.bind(text=self.on_text_changed)
        self.bind(values=self.on_values_changed)

    def update_rect(self, *args):
        self.label.canvas.before.clear()
        with self.label.canvas.before:
            Color(0.15, 0.17, 0.20, 1)
            RoundedRectangle(pos=self.label.pos, size=self.label.size, radius=[4])

    def on_text_changed(self, *args):
        self.label.text = self.text

    def on_values_changed(self, *args):
        if self.values and self.text not in self.values:
            self.text = self.values[0]

    def increment(self, *args):
        idx = self.values.index(self.text)
        self.text = self.values[(idx + 1) % len(self.values)]

    def decrement(self, *args):
        idx = self.values.index(self.text)
        self.text = self.values[(idx - 1) % len(self.values)]

class OptionSelector(BoxLayout):
    text = StringProperty('')

    def __init__(self, values=(), **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.spacing = 5
        self._buttons = {}

        group_name = f"opt_{id(self)}"

        for val in values:
            val_str = str(val)
            state = 'down' if val_str == self.text else 'normal'
            btn = RoundedToggleButton(text=val_str, group=group_name,
                                      allow_no_selection=False, state=state)
            btn.bind(state=self.on_btn_state)
            self.add_widget(btn)
            self._buttons[val_str] = btn

        self.bind(text=self.on_text_change)

    def on_text_change(self, instance, value):
        for val_str, btn in self._buttons.items():
            btn.state = 'down' if val_str == value else 'normal'

    def on_btn_state(self, instance, value):
        if value == 'down':
            self.text = instance.text

class NewGamePopup(Popup):
    def __init__(self, on_confirm, initial_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.on_confirm = on_confirm
        self.title = "New Game Configuration"
        self.title_size = '22sp'
        self.title_font = 'Roboto-Bold'
        self.separator_height = 0
        self.size_hint = (0.75, 0.75)
        self.background_color = (0.05, 0.06, 0.08, 0.95)

        init_w = '9'
        init_h = '10'
        init_fx = '4'
        init_fy = '5'

        if initial_cfg:
             h, w, (fr, fc) = initial_cfg
             init_w = str(w)
             init_h = str(h)
             init_fx = str(fc)
             # user y = (h - 1) - engine_r
             init_fy = str((h - 1) - fr)

        root = BoxLayout(orientation='vertical', spacing=30, padding=25)

        content = BoxLayout(orientation='horizontal', spacing=30)

        controls = BoxLayout(orientation='vertical', spacing=30, size_hint_x=None, width=360)

        def make_label(text, hint_x):
            l = Label(text=text, color=(0.7,0.7,0.7,1), size_hint_x=hint_x, halign='center', valign='middle')
            l.bind(size=lambda s, _: setattr(s, 'text_size', (s.width-10, s.height)))
            return l

        dim_card = Card(orientation='vertical', size_hint_y=None, height=280, padding=20, spacing=10)

        dim_title = Label(text="Board Size", bold=True, font_size='16sp',
                          size_hint_y=None, height=30, halign='center', valign='middle')
        dim_title.bind(size=lambda s, _: setattr(s, 'text_size', (s.width, s.height)))
        dim_card.add_widget(dim_title)
        dim_card.add_widget(Widget(size_hint_y=None, height=4))

        w_box = BoxLayout(spacing=10)
        w_box.add_widget(make_label("Width", 0.25))
        self.width_input = OptionSelector(text=init_w, values=('9', '10', '11', '12'), size_hint_x=0.75)
        w_box.add_widget(self.width_input)
        dim_card.add_widget(w_box)

        h_box = BoxLayout(spacing=10)
        h_box.add_widget(make_label("Height", 0.25))
        self.height_input = OptionSelector(text=init_h, values=('9', '10', '11', '12'), size_hint_x=0.75)
        h_box.add_widget(self.height_input)
        dim_card.add_widget(h_box)

        dim_card.add_widget(Label(text="The model only supports these board sizes.",
                                  color=(0.5, 0.5, 0.5, 1), font_size='11sp', halign='center'))

        rand_dim_btn = RoundedButton(text="Random Size", size_hint_y=None, height=40, bg_color=(0.3, 0.3, 0.35, 1))
        rand_dim_btn.bind(on_release=self.randomize_dims)
        dim_card.add_widget(rand_dim_btn)

        controls.add_widget(dim_card)

        fp_card = Card(orientation='vertical', size_hint_y=None, height=280, padding=20, spacing=10)

        fp_title = Label(text="Forbidden Point", bold=True, font_size='16sp',
                         size_hint_y=None, height=30, halign='center', valign='middle')
        fp_title.bind(size=lambda s, _: setattr(s, 'text_size', (s.width, s.height)))
        fp_card.add_widget(fp_title)
        fp_card.add_widget(Widget(size_hint_y=None, height=4))

        fpx_box = BoxLayout(spacing=10)
        fpx_box.add_widget(make_label("Column (X)", 0.45))
        self.fp_x = SpinBox(text=init_fx, values=['0'], size_hint_x=0.55)
        fpx_box.add_widget(self.fp_x)
        fp_card.add_widget(fpx_box)

        fpy_box = BoxLayout(spacing=10)
        fpy_box.add_widget(make_label("Row (Y)", 0.45))
        self.fp_y = SpinBox(text=init_fy, values=['0'], size_hint_x=0.55)
        fpy_box.add_widget(self.fp_y)
        fp_card.add_widget(fpy_box)

        fp_card.add_widget(Label(text="Topleft is (0,0). Click preview to set.",
                                 color=(0.5, 0.5, 0.5, 1), font_size='12sp', halign='center'))

        rand_fp_btn = RoundedButton(text="Random Point", size_hint_y=None, height=40, bg_color=(0.3, 0.3, 0.35, 1))
        rand_fp_btn.bind(on_release=self.randomize_fp)
        fp_card.add_widget(rand_fp_btn)

        controls.add_widget(fp_card)

        controls.add_widget(Widget()) # Vertical Spacer
        content.add_widget(controls)

        preview_container = Card(bg_color=(0.1, 0.11, 0.13, 1), radius=[12])
        self.preview = PreviewBoardWidget()

        self.preview.cols = int(init_w)
        self.preview.rows = int(init_h)
        self.preview.forbidden_x = int(init_fx)
        self.preview.forbidden_y = int(init_fy)

        preview_container.add_widget(self.preview)
        content.add_widget(preview_container)

        root.add_widget(content)

        btn_box = BoxLayout(orientation='horizontal', spacing=20, size_hint_y=None, height=55)
        btn_box.add_widget(Widget()) # Left spacer to push buttons right

        cancel_btn = RoundedButton(text="Cancel", bg_color=(0.25, 0.25, 0.25, 1), size_hint_x=None, width=120)
        cancel_btn.bind(on_release=self.dismiss)

        start_btn = RoundedButton(text="Start Game", bg_color=(0.2, 0.65, 0.35, 1), size_hint_x=None, width=160, font_size='16sp', bold=True)
        start_btn.bind(on_release=self.confirm)

        btn_box.add_widget(cancel_btn)
        btn_box.add_widget(start_btn)
        root.add_widget(btn_box)

        self.content = root

        # Bindings
        self.width_input.bind(text=self.on_dim_change)
        self.height_input.bind(text=self.on_dim_change)
        self.fp_x.bind(text=self.on_fp_spinner_change)
        self.fp_y.bind(text=self.on_fp_spinner_change)

        self.preview.bind(forbidden_x=self.on_preview_fp_change)
        self.preview.bind(forbidden_y=self.on_preview_fp_change)

        # Init state
        Clock.schedule_once(self.on_dim_change, 0)

    def randomize_dims(self, instance):
        vals = ['9', '10', '11', '12']
        self.width_input.text = random.choice(vals)
        self.height_input.text = random.choice(vals)

    def randomize_fp(self, instance):
        w = int(self.width_input.text)
        h = int(self.height_input.text)
        self.fp_x.text = str(random.randrange(w))
        self.fp_y.text = str(random.randrange(h))

    def on_dim_change(self, *args):
        w = int(self.width_input.text)
        h = int(self.height_input.text)
        self.preview.cols = w
        self.preview.rows = h

        # Update Spinner ranges
        self.fp_x.values = list(map(str, range(w)))
        self.fp_y.values = list(map(str, range(h)))

        # Clamp preview FP if out of bounds
        self.preview.forbidden_x = min(self.preview.forbidden_x, w - 1)
        self.preview.forbidden_y = min(self.preview.forbidden_y, h - 1)

        self.fp_x.text = str(self.preview.forbidden_x)
        self.fp_y.text = str(self.preview.forbidden_y)

    def on_fp_spinner_change(self, *args):
        x = int(self.fp_x.text)
        y = int(self.fp_y.text)
        self.preview.forbidden_x, self.preview.forbidden_y = x, y

    def on_preview_fp_change(self, *args):
        self.fp_x.text, self.fp_y.text = str(self.preview.forbidden_x), str(self.preview.forbidden_y)

    def confirm(self, instance):
        w = int(self.width_input.text)
        h = int(self.height_input.text)
        fx = min(w - 1, int(self.fp_x.text))
        fy = min(h - 1, int(self.fp_y.text))
        fy = h - 1 - fy
        self.dismiss()
        self.on_confirm(h, w, (fy, fx))

def find_winning_line(game):
    if not game.winner: return []
    h, w = game.height, game.width
    board = game.board
    p = game.winner
    for r in range(h):
        for c in range(w):
            if board[r, c] == p:
                for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
                    if 0 <= r + 3*dr < h and 0 <= c + 3*dc < w:
                        if all(board[r+i*dr, c+i*dc] == p for i in range(1, 4)):
                            return [(r+i*dr, c+i*dc) for i in range(4)]
    return []

class MiniBoardWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.game_snapshot = None
        self.pv_moves = []
        self.winning_line = []
        self.pv_info = None
        self.bind(pos=self.update_canvas, size=self.update_canvas)

    def update(self, game, pv_moves, winning_line=None, pv_info=None):
        self.game_snapshot = game
        self.pv_moves = pv_moves
        self.winning_line = winning_line
        self.pv_info = pv_info
        self.update_canvas()

    def update_canvas(self, *args):
        self.canvas.clear()
        if not self.game_snapshot:
            return

        game = self.game_snapshot

        # Draw Background
        with self.canvas:
            Color(0, 0, 0, 0.3)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[8])

        off_x, off_y, w, h, cs = compute_board_geometry(self, game.height, game.width, 0.9)

        with self.canvas:
            Color(*config.color_board)
            RoundedRectangle(pos=(off_x, off_y), size=(w, h), radius=[4])

            draw_board_grid_and_pieces(game.height, game.width, off_x, off_y, cs,
                                       forbidden_point=game.forbidden_point,
                                       board=game.board)

            # Draw PV Moves
            if self.pv_moves:
                for i, (r, c, p) in enumerate(self.pv_moves):
                    cx = off_x + c * cs
                    cy = off_y + r * cs
                    padding = cs * 0.1
                    sz = cs - 2*padding

                    if p == 1:
                        draw_piece(cx + padding, cy + padding, sz, config.color_p1)
                    else:
                        draw_piece(cx + padding, cy + padding, sz, config.color_p2)

                    label = CoreLabel(text=str(i+1), font_size=cs*0.5, halign='center', valign='middle')
                    label.refresh()
                    if label.texture:
                        t = label.texture
                        Color(0, 0, 0, 0.8)
                        Rectangle(texture=t, pos=(cx + (cs - t.size[0])/2, cy + (cs - t.size[1])/2), size=t.size)

            # Draw Winning Line
            if self.winning_line:
                Color(0.2, 1, 0.2, 0.9)
                pts = []
                for r, c in self.winning_line:
                    cx = off_x + c * cs + cs/2
                    cy = off_y + r * cs + cs/2
                    pts.extend([cx, cy])
                Line(points=pts, width=3, cap='round', joint='round')

            # Title
            lbl_title = CoreLabel(text="Principal Variation", font_size=14, color=(0.7, 0.7, 0.8, 1))
            lbl_title.refresh()
            if lbl_title.texture:
                Color(1, 1, 1, 1)
                Rectangle(texture=lbl_title.texture, pos=(self.x + 10, self.top - 25), size=lbl_title.texture.size)

            if self.pv_info:
                v, wr = self.pv_info
                v_str = f"{v/1000:.1f}k" if v >= 1000 else str(v)
                info_text = f"N={v_str} / {wr*100:.1f}%"

                lbl_info = CoreLabel(text=info_text, font_size=14, color=(0.7, 0.7, 0.8, 1))
                lbl_info.refresh()
                if lbl_info.texture:
                    Color(1, 1, 1, 1)
                    # Align right with 10px padding
                    rx = self.x + self.width - lbl_info.texture.size[0] - 10
                    Rectangle(texture=lbl_info.texture, pos=(rx, self.top - 25), size=lbl_info.texture.size)

class BoardWidget(Widget):
    mouse_col = NumericProperty(-1)
    mouse_row = NumericProperty(-1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        Window.bind(mouse_pos=self.on_mouse_move)

    def get_game(self):
        return App.get_running_app().get_current_game()

    def on_mouse_move(self, window, pos):
        app = App.get_running_app()
        if app.dialog_open:
            if self.mouse_col != -1:
                self.mouse_col = self.mouse_row = -1
                self.update_canvas()
            return

        game = self.get_game()
        if not game or not self.collide_point(*pos):
            if self.mouse_col != -1:
                self.mouse_col = self.mouse_row = -1
                self.update_canvas()
            return

        off_x, off_y, w, h, cs = compute_board_geometry(self, game.height, game.width, 0.9)
        inside_board = off_x <= pos[0] <= off_x + w and off_y <= pos[1] <= off_y + h
        if not inside_board:
            if self.mouse_col != -1:
                self.mouse_col = self.mouse_row = -1
                self.update_canvas()
            return

        col = int((pos[0] - off_x) // cs)
        row = int((pos[1] - off_y) // cs)

        self.mouse_col = col if 0 <= col < game.width and game.top[col] < game.height else -1
        self.mouse_row = row if 0 <= row < game.height else -1
        self.update_canvas()

    def on_touch_down(self, touch):
        app = App.get_running_app()
        if app.dialog_open or ('button' in touch.profile and touch.button != 'left'):
            return super().on_touch_down(touch)

        game = self.get_game()
        if not game or not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)

        col = self.mouse_col
        if (
            col != -1
            and game.top[col] < game.height
            and not game.is_terminal()
            and (not app.is_mode(GameMode.PLAY) or game.player == 1)
        ):
            app.make_move(col)
        return True

    def update_canvas(self, *args):
        self.canvas.clear()
        game = self.get_game()
        if not game: return

        offset_x, offset_y, w, h, cell_size = compute_board_geometry(self, game.height, game.width, 0.9)

        app = App.get_running_app()
        mcgs = app.session.mcgs
        if not mcgs:
            return

        # Determine if PV should be shown (hovering exactly over the next move)
        show_pv = (
            app.is_mode(GameMode.ANALYSIS) and 0 <= self.mouse_col < game.width and
            self.mouse_row == game.top[self.mouse_col] and
            mcgs.root and mcgs.root.is_expanded() and mcgs.root.children[self.mouse_col]
        )
        with self.canvas:
            # Background
            Color(*config.color_board)
            RoundedRectangle(pos=(offset_x, offset_y), size=(w, h), radius=[8])

            # Grid & Pieces
            draw_board_grid_and_pieces(game.height, game.width, offset_x, offset_y, cell_size,
                                       forbidden_point=game.forbidden_point,
                                       board=game.board)

            # Last Move
            if game.last_opp_move:
                lr, lc = game.last_opp_move
                cx = offset_x + lc * cell_size
                cy = offset_y + lr * cell_size

                Color(0, 1, 1, 0.6) # Cyan Glow
                d_size = cell_size * 0.3
                d_off = (cell_size - d_size) / 2
                Ellipse(pos=(cx + d_off, cy + d_off), size=(d_size, d_size))

                Color(1, 1, 1, 0.8) # White Center
                d_size2 = cell_size * 0.15
                d_off2 = (cell_size - d_size2) / 2
                Ellipse(pos=(cx + d_off2, cy + d_off2), size=(d_size2, d_size2))

            # Stats Overlay (Hidden if PV is shown)
            if app.is_mode(GameMode.ANALYSIS) and mcgs.root and mcgs.root.is_expanded() and not show_pv:
                lcb_col = app.get_lcb_move()
                root_node = mcgs.root
                next_player = -game.player
                perspective_player = game.player

                # Find Max Stats for Bold
                max_n = -1
                max_wp = -1.0
                for col in range(len(root_node.children)):
                    edge_N = root_node.edge_N[col]
                    if edge_N == 0: continue
                    if edge_N > max_n: max_n = edge_N
                    child = root_node.children[col]
                    q = child.Q if child else 0.0
                    wp = win_rate_for_player(q, next_player, perspective_player)
                    if wp > max_wp: max_wp = wp

                # Draw Bubbles
                for col in range(len(root_node.children)):
                    edge_N = root_node.edge_N[col]
                    if edge_N == 0: continue
                    child = root_node.children[col]
                    row = game.top[col]
                    if row >= game.height: continue

                    cx = offset_x + col * cell_size
                    cy = offset_y + row * cell_size

                    q_val = child.Q if child else 0.0
                    win_pct = win_rate_for_player(q_val, next_player, perspective_player)

                    # Color scale: Red(0) -> Yellow(0.5) -> Green(1)
                    if win_pct < 0.5:
                        r, g, b = 1.0, 2 * win_pct, 0.2
                    else:
                        r, g, b = 2 * (1 - win_pct), 1.0, 0.2

                    padding = cell_size * 0.15
                    radius = (cell_size - 2*padding) / 2

                    # Bubble
                    Color(r, g, b, 0.85)
                    Ellipse(pos=(cx + padding, cy + padding), size=(cell_size - 2*padding, cell_size - 2*padding))

                    # Rim (LCB Highlight)
                    if col == lcb_col:
                        Color(*config.color_lcb)
                        Line(circle=(cx + cell_size/2, cy + cell_size/2, radius), width=3)
                    else:
                        Color(1, 1, 1, 0.3)
                        Line(circle=(cx + cell_size/2, cy + cell_size/2, radius), width=1)

                    # Text
                    wp_str = f"{win_pct*100:.1f}"
                    if win_pct >= max_wp - 1e-9 and max_wp > -1:
                        wp_str = f"[b]{wp_str}[/b]"

                    if edge_N >= 1000:
                        visits_str = f"{edge_N/1000:.1f}k"
                    else:
                        visits_str = str(edge_N)

                    if edge_N >= max_n and max_n > 0:
                        visits_str = f"[b]{visits_str}[/b]"

                    sz_wp = int(cell_size * 0.25)
                    sz_vis = int(cell_size * 0.2)
                    label = MarkupLabel(text=f"[size={sz_wp}]{wp_str}[/size]\n[size={sz_vis}]{visits_str}[/size]",
                                        halign='center',valign='middle', color=(0,0,0,1))
                    label.refresh()
                    if label.texture:
                        tex = label.texture
                        Color(1, 1, 1, 1)
                        tx = cx + (cell_size - tex.size[0]) / 2
                        ty = cy + (cell_size - tex.size[1]) / 2
                        Rectangle(texture=tex, pos=(tx, ty), size=tex.size)

                        # History Move (Analysis) - Dashed Black Circle
                        if app.is_mode(GameMode.ANALYSIS) and app.session.cursor < len(app.session.history) - 1:
                            next_game = app.session.history[app.session.cursor + 1]
                            if next_game.last_opp_move:
                                lr, lc = next_game.last_opp_move
                                cx = offset_x + lc * cell_size
                                cy = offset_y + lr * cell_size
                                padding = cell_size * 0.15
                                radius = (cell_size - 2*padding) / 2

                                Color(0, 0, 0, 0.8)
                                # High-resolution geometric dashing
                                center_x = cx + cell_size/2
                                center_y = cy + cell_size/2
                                num_dashes = 12
                                step_angle = 2 * math.pi / num_dashes
                                draw_angle = step_angle * 0.6

                                for i in range(num_dashes):
                                    start_rad = i * step_angle
                                    end_rad = start_rad + draw_angle

                                    dash_pts = []
                                    for s in range(5):
                                        t = start_rad + (end_rad - start_rad) * s / 4.0
                                        dash_pts.append(center_x + radius * math.cos(t))
                                        dash_pts.append(center_y + radius * math.sin(t))

                                    Line(points=dash_pts, width=2, cap='round')
            # PV on Hover
            if show_pv:
                pv_moves, _, _ = app.calculate_pv(game, mcgs.root, first_action=self.mouse_col)

                for i, (r, c, p) in enumerate(pv_moves):
                    cx = offset_x + c * cell_size
                    cy = offset_y + r * cell_size
                    padding = cell_size * 0.1
                    sz = cell_size - 2*padding

                    draw_piece(cx + padding, cy + padding, sz, config.color_p1 if p == 1 else config.color_p2)

                    label = CoreLabel(text=str(i+1), font_size=cell_size*0.5, halign='center', valign='middle', font_name='Roboto-Bold')
                    label.refresh()
                    if label.texture:
                        t = label.texture
                        Color(0, 0, 0, 0.8)
                        Rectangle(texture=t, pos=(cx + (cell_size - t.size[0])/2, cy + (cell_size - t.size[1])/2), size=t.size)

            # Mouse Hover
            if self.mouse_col != -1 and not game.is_terminal():
                if self.mouse_col < game.width and game.top[self.mouse_col] < game.height:
                    Color(*config.color_highlight)
                    hx = offset_x + self.mouse_col * cell_size
                    RoundedRectangle(pos=(hx, offset_y), size=(cell_size, h), radius=[4])

class StatsPanel(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 20
        self.spacing = 15
        self.last_idx = 0
        self.last_val = 0.5

        # Mode Switches (Primary)
        mode_layout = BoxLayout(size_hint_y=None, height=45, spacing=15)
        self.btn_play = RoundedToggleButton(text='Play', group='mode', state='down')
        self.btn_analysis = RoundedToggleButton(text='Analysis', group='mode')
        self.btn_play.bind(on_press=self.on_mode_change)
        self.btn_analysis.bind(on_press=self.on_mode_change)
        mode_layout.add_widget(self.btn_play)
        mode_layout.add_widget(self.btn_analysis)
        self.add_widget(mode_layout)

        # Game Actions (Secondary)
        action_layout = BoxLayout(size_hint_y=None, height=40, spacing=15)
        self.btn_new = RoundedButton(text='New Game', bg_color=(0.25, 0.28, 0.32, 1))
        self.btn_new.bind(on_release=App.get_running_app().show_new_game_popup)
        action_layout.add_widget(self.btn_new)

        self.btn_import = RoundedButton(text='Import Game', bg_color=(0.25, 0.28, 0.32, 1))
        self.btn_import.bind(on_release=App.get_running_app().show_load_game_popup)
        action_layout.add_widget(self.btn_import)
        self.add_widget(action_layout)

        # Status
        self.status_label = Label(text="Welcome", font_size='22sp', size_hint_y=None, height=40, bold=True)
        self.add_widget(self.status_label)
        self.info_label = Label(text="", font_size='15sp', size_hint_y=None, height=30, color=(0.6,0.6,0.7,1))
        self.add_widget(self.info_label)

        # Graph
        self.graph_container = BoxLayout(size_hint_y=0.35)
        self.graph = Graph(xlabel='Turn', ylabel='Orange Win %', x_ticks_minor=1,
                           x_ticks_major=5, y_ticks_major=0.2,
                           y_grid_label=True, x_grid_label=True, padding=5,
                           x_grid=True, y_grid=True,
                           xmin=0, xmax=10, ymin=0, ymax=1,
                           background_color=(0,0,0,0), border_color=(0.4,0.4,0.4,0.5))
        self.plot = MeshLinePlot(color=[0.95, 0.75, 0.25, 1]) # Match P1 Color
        self.plot.points = [(0, 0.5)]
        self.graph.add_plot(self.plot)
        self.graph_container.add_widget(self.graph)
        self.graph_container.bind(on_touch_down=self.on_graph_touch)

        # Red Dot Indicator
        with self.graph_container.canvas:
            self.dot_color = Color(1, 0.3, 0.3, 0) # Hidden initially
            self.dot = Ellipse(size=(12, 12))

        self.add_widget(self.graph_container)

        self.graph.bind(pos=self.redraw_dot, size=self.redraw_dot)

        # PV
        self.mini_board = MiniBoardWidget(size_hint=(1, 0.55))
        self.add_widget(self.mini_board)

    def on_mode_change(self, instance):
        app = App.get_running_app()
        if instance == self.btn_play:
            app.set_mode(GameMode.PLAY)
        elif instance == self.btn_analysis:
            app.set_mode(GameMode.ANALYSIS)

    def update_status(self, text, color):
        self.status_label.text = text
        self.status_label.color = color

    def update_info(self, text):
        self.info_label.text = text

    def update_graph(self, history, current_idx):
        self.plot.points = history
        self.graph.xmax = max(10, len(history))

        val = 0.5
        if 0 <= current_idx < len(history):
            val = history[current_idx][1]

        self.last_idx = current_idx
        self.last_val = val
        Clock.schedule_once(lambda dt: self.update_dot(current_idx, val), 0)

    def redraw_dot(self, *args):
        Clock.schedule_once(lambda dt: self.update_dot(self.last_idx, self.last_val), 0)

    def update_dot(self, idx, val):
        # Ensure plot area is calculated
        if not hasattr(self.graph, '_plot_area'): return

        # Graph area relative to window/widget
        # Note: graph._plot_area gives (x, y, w, h) relative to graph widget
        gx = self.graph.x + self.graph._plot_area.x
        gy = self.graph.y + self.graph._plot_area.y
        gw = self.graph._plot_area.width
        gh = self.graph._plot_area.height

        if gw <= 0 or gh <= 0: return

        x_range = self.graph.xmax - self.graph.xmin
        y_range = self.graph.ymax - self.graph.ymin
        if x_range == 0: x_range = 1

        x_rel = (idx - self.graph.xmin) / x_range
        y_rel = (val - self.graph.ymin) / y_range

        cx = gx + x_rel * gw
        cy = gy + y_rel * gh

        self.dot.pos = (cx - 5, cy - 5)
        self.dot_color.a = 1.0

    def on_graph_touch(self, instance, touch):
        if 'button' in touch.profile and touch.button != 'left':
            return False

        app = App.get_running_app()
        if app.is_mode(GameMode.PLAY) or not self.graph.collide_point(*touch.pos):
            return False

        gx = self.graph.x + self.graph._plot_area.x
        gw = self.graph._plot_area.width
        if gw <= 0: return False

        rel_x = (touch.x - gx) / gw

        x_range = self.graph.xmax - self.graph.xmin
        idx = int(round(self.graph.xmin + rel_x * x_range))

        idx = max(0, min(idx, len(app.session.history)-1))
        app.navigate_to(idx)
        return True

# --- Main App ---

class KatacApp(App):
    mode = StringProperty(GameMode.PLAY.value)
    dialog_open = False

    def build(self):
        self.config = config
        Window.clearcolor = self.config.color_bg
        self.title = "Katac4 Explorer"
        Window.bind(on_key_down=self.on_key_down)
        Window.bind(on_drop_file=self.on_file_drop)

        self.session = GameSession(self.config)

        root = BoxLayout(orientation='horizontal')
        self.board_widget = BoardWidget(size_hint=(0.7, 1))
        root.add_widget(self.board_widget)
        self.stats_panel = StatsPanel(size_hint=(0.3, 1))
        root.add_widget(self.stats_panel)

        self.set_mode(GameMode.PLAY)
        self.analyzing = False
        self.playout_speed = 0.0
        self.pv_update_count = 0

        Clock.schedule_interval(self.ai_loop, 1.0/30.0)
        return root

    def show_new_game_popup(self, instance):
        self.dialog_open = True
        self.analyzing = False

        current_game = self.get_current_game()
        initial_cfg = None
        if current_game:
            initial_cfg = (current_game.height, current_game.width, current_game.forbidden_point)

        popup = NewGamePopup(on_confirm=self.start_new_game, initial_cfg=initial_cfg)
        popup.bind(on_dismiss=self.on_popup_dismiss)
        popup.open()

    def show_load_game_popup(self, instance):
        self.analyzing = False
        filechooser.open_file(on_selection=self.on_file_selected, filters=[("JSON Files", "*.json")])

    def on_file_selected(self, selection):
        if selection:
            self.import_game_from_file(selection[0])

    def on_popup_dismiss(self, instance):
        self.dialog_open = False

    def on_file_drop(self, window, file_path, *args):
        file_path = file_path.decode('utf-8') if isinstance(file_path, bytes) else file_path
        if file_path.lower().endswith('.json'):
            self.import_game_from_file(file_path)

    def import_game_from_file(self, filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Extract parameters
            h, w = data['row'], data['col']
            forbidden_point = (h - 1 - data['nox'], data['noy'])
            rounds = data['rounds']

            self.session.load_rounds(h, w, forbidden_point, rounds)
            self.set_mode(GameMode.ANALYSIS)
            self.analyzing = False
            self.pv_update_count = 0
            self.update_gui()

        except Exception as e:
            print(f"Failed to import game: {e}")
            import traceback
            traceback.print_exc()

    def start_new_game(self, height, width, forbidden_point):
        self.session.start_new_game(height=height, width=width, forbidden_point=forbidden_point)
        self.pv_update_count = 0
        self.update_gui()

    def get_current_game(self):
        return self.session.current_game

    def is_mode(self, mode: GameMode) -> bool:
        return self.mode == mode.value

    def set_mode(self, mode: GameMode):
        self.mode = mode.value
        self.analyzing = False

        if hasattr(self, 'stats_panel'):
            if mode is GameMode.PLAY:
                self.stats_panel.btn_play.state = 'down'
                self.stats_panel.btn_analysis.state = 'normal'
                self.stats_panel.graph_container.opacity = 0
                self.stats_panel.mini_board.opacity = 0
            else:
                self.stats_panel.btn_analysis.state = 'down'
                self.stats_panel.btn_play.state = 'normal'
                self.stats_panel.graph_container.opacity = 1
                self.stats_panel.mini_board.opacity = 1
        self.update_gui()

    def navigate_to(self, idx):
        if not (0 <= idx < len(self.session.history)):
            return
        if self.is_mode(GameMode.PLAY):
            self.set_mode(GameMode.ANALYSIS)

        if not self.session.navigate(idx):
            return
        self.pv_update_count = 0
        self.update_gui()

    def on_key_down(self, window, key, scancode, codepoint, modifier):
        if key == 275: # Right
            self.navigate_to(self.session.cursor + 1)
        elif key == 276: # Left
            self.navigate_to(self.session.cursor - 1)
        elif key == 32: # Space
            if self.is_mode(GameMode.PLAY):
                self.set_mode(GameMode.ANALYSIS)
                self.analyzing = True
            elif self.is_mode(GameMode.ANALYSIS):
                self.analyzing = not self.analyzing
        return True

    def make_move(self, col):
        next_game = self.session.apply_move(col)
        if next_game is None:
            return
        self.update_live_winrate()
        self.update_gui()

    def get_lcb_move(self):
        mcgs = getattr(self.session, 'mcgs', None)
        return mcgs.get_best_move() if mcgs else None

    def calculate_pv(self, game, node, first_action=None):
        pv = []
        winning_line = []
        sim_game = copy.deepcopy(game)

        # Greedy Descent
        for _ in range(25):
            if not node.is_expanded(): break
            if sim_game.is_terminal():
                winning_line = find_winning_line(sim_game)
                break

            if first_action is None:
                max_n, best_act = max(zip(node.edge_N, range(len(node.children))))
                if max_n < 50 or best_act is None: break
            else:
                best_act = first_action
                first_action = None

            row = sim_game.top[best_act]
            pv.append((row, best_act, sim_game.player))

            sim_game.step(best_act)
            node = node.children[best_act]
            if sim_game.is_terminal():
                winning_line = find_winning_line(sim_game)
                break

        return pv, winning_line, node

    def update_pv_visualization(self):
        game = self.get_current_game()
        mcgs = getattr(self.session, 'mcgs', None)
        if not game or not mcgs or not mcgs.root:
            return

        pv, winning_line, final_node = self.calculate_pv(game, mcgs.root)

        pv_info = None
        if final_node:
            v = final_node.N
            q = final_node.Q
            final_player = game.player if len(pv) % 2 == 0 else -game.player
            wr = win_rate_for_player(q, final_player, game.player)
            pv_info = (v, wr)

        self.stats_panel.mini_board.update(game, pv, winning_line, pv_info=pv_info)

        # 2. History Backprop
        total_moves = len(self.session.history)
        for i in range(total_moves - 1, -1, -1):
            h_game = self.session.history[i]
            if h_game.hash in mcgs.nodes_by_hash:
                node = mcgs.nodes_by_hash[h_game.hash]
                node.update()

                q = node.Q
                wr = orange_win_rate(q, h_game.player)

                self.session.set_winrate(i, wr)

        self.stats_panel.update_graph(self.session.win_history, self.session.cursor)

    def update_live_winrate(self):
        game = self.get_current_game()
        if not game: return

        mcgs = getattr(self.session, 'mcgs', None)
        root = getattr(mcgs, 'root', None)
        cursor = self.session.cursor
        win_history = self.session.win_history

        wr = 0.5
        if root and root.N:
            wr = orange_win_rate(root.Q, game.player)
        elif game.is_terminal():
            wr = {1: 1.0, -1: 0.0}.get(game.winner, 0.5)
        elif mcgs:
            mcgs.playout(game)
            root = mcgs.root
            if root and root.N:
                wr = orange_win_rate(root.Q, game.player)

        self.session.set_winrate(cursor, wr)
        self.stats_panel.update_graph(win_history, cursor)

    def update_gui(self):
        self.board_widget.update_canvas()
        self.stats_panel.update_graph(self.session.win_history, self.session.cursor)
        self.update_pv_visualization()

    def ai_loop(self, dt):
        game = self.get_current_game()
        if not game: return

        mcgs = getattr(self.session, 'mcgs', None)
        if not mcgs or not mcgs.root:
            return

        root = mcgs.root

        # Logic: Play mode auto, Analysis mode toggle
        should_run = False
        status_txt = ""
        status_col = (1,1,1,1)

        if self.is_mode(GameMode.ANALYSIS):
            if game.is_terminal():
                self.analyzing = False
                status_txt = "Game Over"
                if game.winner == 0:
                    self.stats_panel.update_info("The game ended in a draw")
                else:
                    winner = self.config.player_names[game.winner]
                    self.stats_panel.update_info(f"{winner} wins!")
            else:
                player_name = self.config.player_names[game.player]
                win_rate = win_rate_for_player(root.Q, game.player, game.player)
                status_txt = f"{player_name} to play | Win Rate: {win_rate:.1%}"
                if self.analyzing:
                    should_run = True
                else:
                    self.stats_panel.update_info("Analysis paused" if root.N > 0 else "Press space to toggle analysis")
        else: # GameMode.PLAY
            if game.is_terminal():
                if game.winner == 1: status_txt, status_col = "YOU WIN!", list(self.config.color_p1)
                elif game.winner == -1: status_txt, status_col = "AI WINS!", list(self.config.color_p2)
                else: status_txt = "DRAW"
                self.stats_panel.update_info("View analysis or start a new game")
            elif game.player == -1:
                should_run = True
                status_txt, status_col = "AI THINKING...", list(self.config.color_p2)
            else:
                status_txt, status_col = "YOUR TURN", list(self.config.color_p1)
                self.stats_panel.update_info("Click a column to make your move")

        self.stats_panel.update_status(status_txt, status_col)
        self.update_live_winrate()

        if not should_run:
            return

        root = mcgs.root

        # 25ms per frame
        start_t = time.time()
        target_t = start_t + 0.025
        executed = 0
        while time.time() < target_t and (not self.is_mode(GameMode.PLAY) or root.N < self.config.n_playout):
            for _ in range(5):
                mcgs.playout(game)
                executed += 1

        # Adaptive Sizing
        if executed > 0:
            elapsed = time.time() - start_t
            speed = executed / elapsed
            if self.playout_speed < 10:
                self.playout_speed = speed
            else:
                self.playout_speed = 0.9 * self.playout_speed + 0.1 * speed
        self.stats_panel.update_info(f"Playouts: {root.N} ({round(self.playout_speed)}/s)")

        self.board_widget.update_canvas()

        # Periodic PV
        self.pv_update_count += executed
        if self.pv_update_count >= 100:
            self.update_pv_visualization()
            self.pv_update_count = 0

        # AI Move Trigger
        if self.is_mode(GameMode.PLAY) and root.N >= self.config.n_playout:
            best_action = self.get_lcb_move()
            if best_action is not None:
                self.make_move(best_action)
            self.pv_update_count = 0

if __name__ == '__main__':
    KatacApp().run()

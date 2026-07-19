import math
from collections import defaultdict
from time import time


class TreeNode:
    __slots__ = ('N', 'Q', 'avg_Q2', 'U', 'children', 'edge_P', 'p_explored', 'state_hash')

    def __init__(self):
        self.N = 0
        self.Q = 0
        self.avg_Q2 = 0
        self.U = 0
        self.children = None    # {action: (child_node, edge_N)}
        self.edge_P = None      # {action: P}
        self.p_explored = 0.0   # sum of priors for visited edges
        self.state_hash = None

    def select(self, c_puct, c_fpu):
        if self.N > 2:
            c_puct *= self.cpuct_scaler
        c_puct *= math.sqrt(self.N)
        fpu_Q = self.Q - c_fpu * math.sqrt(self.p_explored)
        best_score = -math.inf
        for action, edge in self.children.items():
            child, edge_N = edge
            edge_Q = -child.Q if child else fpu_Q
            score = edge_Q + c_puct * self.edge_P[action] / (1 + edge_N)
            if score > best_score:
                best_action, best_edge, best_score = action, edge, score
        return best_action, best_edge

    def update(self):
        sum_Q = self.U
        sum_Q2 = self.U * self.U
        for child, edge_N in self.children.values():
            if edge_N > 0:
                sum_Q -= child.Q * edge_N
                sum_Q2 += child.avg_Q2 * edge_N
        self.Q = sum_Q / self.N
        self.avg_Q2 = sum_Q2 / self.N

    @property
    def var(self):
        # prevent negative variance due to numerical errors
        return max(0.0, self.avg_Q2 - self.Q ** 2)

    @property
    def cpuct_scaler(self):
        k = 4 * math.sqrt(self.var) / self.N
        k = min(max(k, 0.5), 1.4)
        child_N = sum(child.N for child, _ in self.children.values() if child)
        alpha = 1.0 / (1 + math.sqrt(child_N / 10000))
        return alpha * k + (1.0 - alpha)


class MCGS:
    __slots__ = ('c_puct', 'c_fpu', 'policy', 'root', 'nodes_by_hash', 'z_table')

    def __init__(self, policy_value_fn, z_table, c_puct=1.0, c_fpu=0.2):
        self.c_puct = c_puct
        self.c_fpu = c_fpu
        self.policy = policy_value_fn
        self.root = None
        self.nodes_by_hash = defaultdict(TreeNode)
        self.z_table = z_table

    def _t_quantile(self, df):
        return self.z_table[0 if df < 1 else min(df, len(self.z_table)) - 1]

    def _playout(self, state):
        state = state.clone()
        node = self.root
        path = []  # does not include leaf
        while node.children:
            path.append(node)
            action, (child, edge_N) = node.select(self.c_puct, self.c_fpu)
            state.step(action)
            if child is None:
                child = self.nodes_by_hash[state.hash]
            if edge_N == 0:
                node.p_explored += node.edge_P[action]
            node.children[action] = (child, edge_N + 1)
            node = child
        if state.is_terminal():
            node.Q = node.U = -1.0 if state.winner else 0.0
        else:
            acts, probs, value = self.policy(state)
            node.edge_P = dict(zip(acts, probs))
            node.Q = node.U = value
            node.children = {action: (None, 0) for action in acts}
        node.avg_Q2 = node.Q ** 2
        node.N += 1
        for node in reversed(path):
            node.N += 1
            node.update()

    def _reroot(self, state):
        if state.hash not in self.nodes_by_hash:
            self.nodes_by_hash.clear()
            self.root = self.nodes_by_hash[state.hash]
            return
        self.root = self.nodes_by_hash[state.hash]
        for hash_value, node in self.nodes_by_hash.items():
            node.state_hash = hash_value
        self.nodes_by_hash.clear()
        def dfs(node):
            self.nodes_by_hash[node.state_hash] = node
            node.state_hash = None
            if not node.children:
                return
            for _, (child, _) in node.children.items():
                if child and child.state_hash:
                    dfs(child)
            node.update()
        dfs(self.root)

    def search(self, game, timeout):
        time_limit = time() + timeout
        self._reroot(game)

        for _ in range(16):
            self._playout(game)

        while time() < time_limit:
            for _ in range(4):
                self._playout(game)

        N_min = max(math.ceil(0.1 * self.root.N), 2)
        cands = [
            (action, child)
            for action, (child, _) in self.root.children.items()
            if child and child.N >= N_min
        ]

        if not cands:
            return max(self.root.children.items(), key=lambda x: x[1][1])[0], float('nan')

        def lcb(edge):
            _, child = edge
            var = child.var * (child.N / (child.N - 1))
            return -child.Q - self._t_quantile(child.N) * math.sqrt(var) / child.N

        edge = max(cands, key=lcb)
        return edge[0], lcb(edge)

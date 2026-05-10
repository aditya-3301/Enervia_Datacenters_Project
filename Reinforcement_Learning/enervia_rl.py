import random
import math
import csv
import os
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_tariff_schedule(path):
    schedule = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule[int(row["hour"])] = (
                float(row["tariff"]),
                row["status"],
                float(row["severity"])
            )
    return schedule


def get_cost_context(hour, schedule, noise=0.0):
    tariff, status, severity = schedule[hour % 24]
    tariff += random.uniform(-noise, noise)
    return max(3.0, tariff), status, severity


class BatteryEnv:
    CAPACITY_KWH = 200.0
    MAX_POWER_KW = 50.0
    EFFICIENCY   = 0.92
    SOC_MIN_PCT  = 0.10
    SOC_MAX_PCT  = 0.95

    def __init__(self, schedule, noise=0.3):
        self.schedule  = schedule
        self.noise     = noise
        self.n_actions = 3
        self.obs_size  = 4
        self.reset()

    def reset(self):
        self.hour    = 0
        self.soc     = self.CAPACITY_KWH * 0.30
        self.history = []
        return self._obs()

    def step(self, action):
        tariff, status, severity = get_cost_context(self.hour, self.schedule, self.noise)
        reward, power_kw, label  = self._apply(action, tariff, status)
        self.history.append({
            "hour":     self.hour,
            "action":   label,
            "soc_pct":  round(self.soc / self.CAPACITY_KWH * 100, 1),
            "tariff":   tariff,
            "status":   status,
            "power_kw": power_kw,
            "reward":   round(reward, 2)
        })
        self.hour += 1
        return self._obs(), reward, self.hour >= 24

    def _obs(self):
        tariff, _, severity = get_cost_context(self.hour % 24, self.schedule)
        return np.array([
            self.hour / 23.0,
            self.soc / self.CAPACITY_KWH,
            (tariff - 4.0) / 6.0,
            severity / 10.0
        ], dtype=np.float32)

    def _apply(self, action, tariff, status):
        soc_min = self.CAPACITY_KWH * self.SOC_MIN_PCT
        soc_max = self.CAPACITY_KWH * self.SOC_MAX_PCT

        if action == 1:
            headroom  = soc_max - self.soc
            charge_kw = min(self.MAX_POWER_KW, headroom / self.EFFICIENCY)
            if charge_kw < 1.0:
                return 0.0, 0.0, "IDLE"
            self.soc += charge_kw * self.EFFICIENCY
            if status == "OFF_PEAK":
                return 0.5, charge_kw, "CHARGE"
            elif status == "SHOULDER":
                return -0.5, charge_kw, "CHARGE"
            else:
                return -3.0, charge_kw, "CHARGE"

        elif action == 2:
            available = self.soc - soc_min
            disch_kw  = min(self.MAX_POWER_KW, available * self.EFFICIENCY)
            if disch_kw < 1.0:
                return 0.0, 0.0, "IDLE"
            self.soc -= disch_kw / self.EFFICIENCY
            if status == "PEAK":
                return (disch_kw * tariff) / 10.0, disch_kw, "DISCHARGE"
            elif status == "SHOULDER":
                return (disch_kw * tariff) / 20.0, disch_kw, "DISCHARGE"
            else:
                return -1.0, disch_kw, "DISCHARGE"

        else:
            if status == "OFF_PEAK" and (self.soc / self.CAPACITY_KWH) < 0.5:
                return -0.2, 0.0, "IDLE"
            return 0.0, 0.0, "IDLE"


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a),
                np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class DQNNet:
    def __init__(self, obs_size=4, n_actions=3, hidden=128, lr=5e-4):
        self.lr = lr
        self.n_actions = n_actions

        def W(i, o): return np.random.randn(i, o) * math.sqrt(2.0 / i)
        def b(o):    return np.zeros((1, o))

        self.W1, self.b1 = W(obs_size, hidden), b(hidden)
        self.W2, self.b2 = W(hidden, hidden),   b(hidden)
        self.W3, self.b3 = W(hidden, n_actions), b(n_actions)

    def _relu(self, x):  return np.maximum(0, x)
    def _drelu(self, x): return (x > 0).astype(float)

    def forward(self, x):
        self.x0 = x
        self.z1 = x @ self.W1 + self.b1;       self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.a2 = self._relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3

    def predict(self, x):
        if x.ndim == 1: x = x[np.newaxis]
        return self.forward(x)

    def train_step(self, states, actions, targets):
        q = self.forward(states)
        grad = np.zeros_like(q)
        for i, (a, t) in enumerate(zip(actions, targets)):
            grad[i, a] = 2 * (q[i, a] - t)
        d3  = grad
        dW3 = self.a2.T @ d3;   db3 = d3.sum(0, keepdims=True)
        d2  = (d3 @ self.W3.T) * self._drelu(self.z2)
        dW2 = self.a1.T @ d2;   db2 = d2.sum(0, keepdims=True)
        d1  = (d2 @ self.W2.T) * self._drelu(self.z1)
        dW1 = self.x0.T @ d1;   db1 = d1.sum(0, keepdims=True)
        for p, g in [(self.W3,dW3),(self.b3,db3),(self.W2,dW2),
                     (self.b2,db2),(self.W1,dW1),(self.b1,db1)]:
            p -= self.lr * np.clip(g / len(states), -1, 1)
        return float(np.mean(grad ** 2))


class DQNAgent:
    def __init__(self, obs_size=4, n_actions=3, gamma=0.95, lr=5e-4,
                 batch_size=64, eps_start=1.0, eps_end=0.05,
                 eps_decay=0.995, target_update=50):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.eps           = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.target_update = target_update
        self.step_count    = 0
        self.online        = DQNNet(obs_size, n_actions, lr=lr)
        self.target        = DQNNet(obs_size, n_actions, lr=lr)
        self.memory        = ReplayBuffer()
        self._sync()

    def _sync(self):
        for attr in ['W1','b1','W2','b2','W3','b3']:
            setattr(self.target, attr, getattr(self.online, attr).copy())

    def act(self, state, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.online.predict(state)))

    def push(self, s, a, r, ns, done):
        self.memory.push(s, a, r, ns, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        s, a, r, ns, d = self.memory.sample(self.batch_size)
        q_next  = self.target.forward(ns).max(axis=1)
        targets = r + self.gamma * q_next * (1 - d)
        loss    = self.online.train_step(s, a, targets)
        self.step_count += 1
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        if self.step_count % self.target_update == 0:
            self._sync()
        return loss


def train(schedule, n_episodes=800, eval_every=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    env   = BatteryEnv(schedule)
    agent = DQNAgent()
    ep_rewards, ep_losses, eval_rewards, eval_episodes = [], [], [], []

    for ep in range(1, n_episodes + 1):
        state   = env.reset()
        total_r = 0.0
        losses  = []
        for _ in range(24):
            action               = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.push(state, action, reward, next_state, float(done))
            loss = agent.learn()
            if loss: losses.append(loss)
            total_r += reward
            state    = next_state
        ep_rewards.append(total_r)
        ep_losses.append(np.mean(losses) if losses else 0)
        if ep % eval_every == 0:
            eval_r = run_episode(env, agent, greedy=True)
            eval_rewards.append(eval_r)
            eval_episodes.append(ep)

    return agent, env, ep_rewards, ep_losses, eval_rewards, eval_episodes


def run_episode(env, agent, greedy=False):
    state   = env.reset()
    total_r = 0.0
    for _ in range(24):
        action           = agent.act(state, greedy=greedy)
        state, reward, _ = env.step(action)
        total_r         += reward
    return total_r


def plot_training(ep_rewards, ep_losses, eval_rewards, eval_episodes, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor("#0b0f14")
    for ax in axes:
        ax.set_facecolor("#131920")
        ax.tick_params(colors="#8a9bb0")
        for sp in ax.spines.values(): sp.set_edgecolor("#252e3a")

    window   = 30
    smoothed = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
    x_smooth = range(window, len(ep_rewards) + 1)

    axes[0].plot(ep_rewards, color="#252e3a", linewidth=0.6, label="Raw")
    axes[0].plot(x_smooth, smoothed, color="#29a65e", linewidth=2, label="MA-30")
    axes[0].scatter(eval_episodes, eval_rewards, color="#4d9fff", s=45, zorder=5, label="Eval")
    axes[0].set_title("Episode Reward", color="#dce7f0", fontsize=12)
    axes[0].set_xlabel("Episode", color="#8a9bb0")
    axes[0].legend(facecolor="#1a2130", labelcolor="#dce7f0", fontsize=9)

    axes[1].plot(ep_losses, color="#9d72f5", linewidth=0.8)
    axes[1].set_title("Training Loss", color="#dce7f0", fontsize=12)
    axes[1].set_xlabel("Episode", color="#8a9bb0")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0b0f14")
    plt.close()


def plot_episode(history, out):
    hours   = [h["hour"]    for h in history]
    soc_pct = [h["soc_pct"] for h in history]
    tariffs = [h["tariff"]  for h in history]
    actions = [h["action"]  for h in history]

    status_color = {"OFF_PEAK": "#29a65e", "SHOULDER": "#dba63a", "PEAK": "#f05050"}
    action_color = {"CHARGE": "#4d9fff", "DISCHARGE": "#29a65e", "IDLE": "#576070"}

    from matplotlib.patches import Patch

    fig  = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#0b0f14")
    gs   = gridspec.GridSpec(3, 1, hspace=0.5)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    for ax in axes:
        ax.set_facecolor("#131920")
        ax.tick_params(colors="#8a9bb0", labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#252e3a")
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(24))

    axes[0].bar(hours, tariffs,
                color=[status_color[h["status"]] for h in history],
                alpha=0.85, width=0.8)
    axes[0].set_title("Electricity Tariff  (₹/kWh)", color="#dce7f0", fontsize=11)
    axes[0].set_ylabel("₹/kWh", color="#8a9bb0", fontsize=9)
    axes[0].legend(handles=[Patch(color=v, label=k) for k, v in status_color.items()],
                   facecolor="#1a2130", labelcolor="#dce7f0", fontsize=8)

    axes[1].fill_between(hours, soc_pct, alpha=0.25, color="#4d9fff")
    axes[1].plot(hours, soc_pct, color="#4d9fff", linewidth=2)
    axes[1].axhline(95, color="#f05050", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[1].axhline(10, color="#f05050", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[1].set_title("Battery State-of-Charge (%)", color="#dce7f0", fontsize=11)
    axes[1].set_ylabel("SoC %", color="#8a9bb0", fontsize=9)
    axes[1].set_ylim(0, 105)

    axes[2].bar(hours, [1] * 24,
                color=[action_color[a] for a in actions],
                alpha=0.85, width=0.8)
    axes[2].legend(handles=[Patch(color=v, label=k) for k, v in action_color.items()],
                   facecolor="#1a2130", labelcolor="#dce7f0", fontsize=8)
    axes[2].set_title("Agent Actions per Hour", color="#dce7f0", fontsize=11)
    axes[2].set_yticks([])
    axes[2].set_xlabel("Hour of Day", color="#8a9bb0", fontsize=9)

    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0b0f14")
    plt.close()


if __name__ == "__main__":
    OUT      = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(OUT, "tariff_schedule.csv")

    schedule = load_tariff_schedule(CSV_PATH)

    agent, env, ep_r, ep_l, eval_r, eval_ep = train(schedule, n_episodes=800)

    plot_training(ep_r, ep_l, eval_r, eval_ep,
                  os.path.join(OUT, "enervia_rl_training.png"))

    run_episode(env, agent, greedy=True)
    plot_episode(env.history, os.path.join(OUT, "enervia_rl_episode.png"))

    charges    = [h for h in env.history if h["action"] == "CHARGE"]
    discharges = [h for h in env.history if h["action"] == "DISCHARGE"]
    avg_c      = np.mean([h["tariff"] for h in charges])    if charges    else 0
    avg_d      = np.mean([h["tariff"] for h in discharges]) if discharges else 0

    print(f"Total reward   : {sum(h['reward'] for h in env.history):.2f}")
    print(f"CHARGE  actions: {len(charges):>3}  avg tariff ₹{avg_c:.2f}")
    print(f"DISCHARGE      : {len(discharges):>3}  avg tariff ₹{avg_d:.2f}")
    print(f"IDLE           : {24 - len(charges) - len(discharges):>3}")
    print(f"Tariff spread  : ₹{avg_d - avg_c:.2f}/kWh")

"""
四足机器人 Gymnasium 环境（给强化学习用）

设计上刻意保持"最小可训练"：
    动作 = 12 个关节的目标角偏移（[-1, 1]，乘 action_scale）
    观测 = 躯干速度 + 重力方向 + 关节角/角速度 + 上一步动作 + 目标速度
    奖励 = 速度跟踪 + 保持直立 + 高度 + 动作平滑 - 能耗
    频率 = 物理 500 Hz，策略 50 Hz（frame_skip=10）

用法：
    from envs.quadruped_env import QuadrupedEnv
    env = QuadrupedEnv()
    obs, info = env.reset()
    obs, r, term, trunc, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import math
import pathlib
import sys

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from quad_utils import (  # noqa: E402
    JOINTS,
    MODEL_PATH,
    N_JOINTS,
    STAND,
    STAND_Z,
    balance_control,
    base_rpy,
    body_velocity,
    dof_adr,
    qpos_adr,
)


class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | pathlib.Path = MODEL_PATH,
        frame_skip: int = 10,
        target_speed: float = 0.4,
        episode_seconds: float = 10.0,
        action_scale: float = 0.45,
        randomize: bool = True,
        use_clock: bool = True,
        clock_period: float = 0.5,
        residual: bool = False,
        residual_scale: float = 0.2,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip     # 0.02 s -> 50 Hz
        self.max_steps = int(episode_seconds / self.dt)
        self.target_speed = target_speed
        self.action_scale = action_scale
        self.randomize = randomize
        self.use_clock = use_clock
        self.clock_period = clock_period
        # residual=True：以阶段 5 的传统控制器为基线，RL 只学一个小的修正量
        self.residual = residual
        self.residual_scale = residual_scale
        self.render_mode = render_mode

        # 关节索引（qpos / qvel 里对应的位置）
        self.qpos_idx = np.array([qpos_adr(self.model, j) for j in JOINTS])
        self.qvel_idx = np.array([dof_adr(self.model, j) for j in JOINTS])

        # 关节限位，用来裁剪目标角
        self.joint_low, self.joint_high = [], []
        for j in JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.joint_low.append(self.model.jnt_range[jid, 0])
            self.joint_high.append(self.model.jnt_range[jid, 1])
        self.joint_low = np.array(self.joint_low)
        self.joint_high = np.array(self.joint_high)

        self.action_space = spaces.Box(-1.0, 1.0, (N_JOINTS,), dtype=np.float32)

        # 3 线速度 + 3 角速度 + 3 重力方向 + 12 关节角 + 12 关节角速度
        # + 12 上一步动作 + 1 目标速度 = 46；启用步态时钟再 + 2
        obs_dim = 3 + 3 + 3 + N_JOINTS + N_JOINTS + N_JOINTS + 1
        obs_dim += 2 if self.use_clock else 0
        self.observation_space = spaces.Box(-50.0, 50.0, (obs_dim,), dtype=np.float32)

        self.prev_action = np.zeros(N_JOINTS, dtype=np.float64)
        self.step_count = 0
        self._renderer = None
        self._viewer = None

    # ------------------------------------------------------------------
    # Gym 接口
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        if self.randomize:
            self.data.qpos[2] += self.np_random.uniform(-0.005, 0.005)
            self.data.qpos[self.qpos_idx] += self.np_random.uniform(-0.05, 0.05, N_JOINTS)
            self.data.qvel[:] = self.np_random.uniform(-0.1, 0.1, self.model.nv)

        mujoco.mj_forward(self.model, self.data)
        self.prev_action[:] = 0.0
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        if self.residual:
            # 基线 = 阶段 5 的传统控制器（正弦步态 + 姿态反馈）
            # RL 只负责输出一个小修正量，所以一开训就已经会走
            baseline = balance_control(self.model, self.data, self.data.time)
            q_target = baseline + action * self.residual_scale
        else:
            # 从零学：动作直接是"站立姿态附近的关节角偏移"
            q_target = STAND + action * self.action_scale

        q_target = np.clip(q_target, self.joint_low, self.joint_high)
        self.data.ctrl[:] = q_target

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        if terminated:
            reward -= 5.0          # 摔倒的额外惩罚，避免"往前扑"占便宜
        truncated = self.step_count >= self.max_steps

        self.prev_action[:] = action

        _, linvel = body_velocity(self.model, self.data)
        roll, pitch, _ = base_rpy(self.data)
        info = {
            "vx": float(linvel[0]),
            "height": float(self.data.qpos[2]),
            "roll": float(roll),
            "pitch": float(pitch),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # 观测 / 奖励 / 终止
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        angvel, linvel = body_velocity(self.model, self.data)

        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, self.data.qpos[3:7])
        gravity = mat.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

        q = self.data.qpos[self.qpos_idx]
        dq = self.data.qvel[self.qvel_idx]

        parts = [
            linvel * 2.0,          # 缩放是为了让 PPO 的网络好训练
            angvel * 0.25,
            gravity,
            q - STAND,
            dq * 0.05,
            self.prev_action,
            [self.target_speed],
        ]

        # 步态时钟：告诉策略"现在处在步态周期的哪一段"。
        # 没有它，PPO 的探索是每步独立的噪声，很难自己发现"周期性摆腿"。
        if self.use_clock:
            phase = 2.0 * math.pi * self.data.time / self.clock_period
            parts.append([math.sin(phase), math.cos(phase)])

        return np.concatenate(parts).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> float:
        _, linvel = body_velocity(self.model, self.data)
        roll, pitch, _ = base_rpy(self.data)
        z = float(self.data.qpos[2])
        dq = self.data.qvel[self.qvel_idx]

        vx = float(linvel[0])

        # 1) 指数项：越接近目标速度分越高（要求"跟得准"）
        tracking = math.exp(-6.0 * (vx - self.target_speed) ** 2)
        # 2) 线性项：只要往前走就比站着强。
        #    上限设为 1.0，超过目标速度不再加分，避免策略为了多拿分而"飙车摔跤"。
        progress = float(np.clip(vx / max(self.target_speed, 1e-3), -0.5, 1.0))

        upright = math.exp(-3.0 * (roll * roll + pitch * pitch))
        smooth = math.exp(-2.0 * float(np.sum((action - self.prev_action) ** 2)))

        reward = 2.0 * tracking + 1.0 * progress + 0.4 * upright + 0.1 * smooth
        reward -= 2.0 * (z - STAND_Z) ** 2
        reward -= 0.0005 * float(np.sum(dq * dq))
        reward -= 0.05 * float(np.sum((action - self.prev_action) ** 2))
        return reward

    def _is_terminated(self) -> bool:
        z = float(self.data.qpos[2])
        roll, pitch, _ = base_rpy(self.data)
        if z < 0.15:
            return True
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            return True
        if abs(float(self.data.qpos[1])) > 2.5:     # 跑出场地
            return True
        return False

    # ------------------------------------------------------------------
    # 可视化（只在 eval 里用）
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

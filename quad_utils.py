"""
四足机器人项目 —— 公共工具

设计原则：
1. 只放"每个阶段都要用"的东西（模型加载、IK、姿态读取、仿真循环）。
2. 不引入任何额外抽象，脚本里能直接看懂的就不封起来。
"""

from __future__ import annotations

import contextlib
import math
import pathlib
import time

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
MODEL_PATH = ROOT / "quadruped_v2.xml"

# ---------------------------------------------------------------------------
# 模型常量（和 quadruped_v2.xml 一一对应）
# ---------------------------------------------------------------------------
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = tuple(f"{leg}_{part}_j" for leg in LEGS for part in ("hip", "thigh", "knee"))
N_JOINTS = len(JOINTS)

L1 = 0.15          # 大腿长度（髋 -> 膝）
L2 = 0.15          # 小腿长度（膝 -> 足心）
FOOT_R = 0.022     # 足端球半径

STAND = np.array([0.0, 0.5, -1.0] * 4, dtype=np.float64)  # 站立关节角（弧度）
STAND_Z = 0.2853   # 站立时躯干中心高度
STAND_H = 0.2633   # 站立时"髋 -> 足心"的垂直距离

HIP_X = {"FL": 0.18, "FR": 0.18, "RL": -0.18, "RR": -0.18}
HIP_Y = {"FL": 0.085, "FR": -0.085, "RL": 0.085, "RR": -0.085}

# 步态相位：trot 是"对角腿一组"，walk 是"四条腿依次"
TROT_OFFSETS = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}
WALK_OFFSETS = {"FL": 0.0, "RR": 0.25, "FR": 0.5, "RL": 0.75}


# ---------------------------------------------------------------------------
# 模型 / 数据
# ---------------------------------------------------------------------------
def load(path: str | pathlib.Path = MODEL_PATH, keyframe: str = "stand"):
    """加载模型，并把状态重置到站立 keyframe。"""
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    reset_to_keyframe(model, data, keyframe)
    return model, data


def reset_to_keyframe(model, data, keyframe: str = "stand"):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if kid < 0:
        raise ValueError(f"模型里没有 keyframe: {keyframe}")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)


def qpos_adr(model, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[jid])


def dof_adr(model, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_dofadr[jid])


def get_joint_angles(model, data) -> np.ndarray:
    return np.array([data.qpos[qpos_adr(model, j)] for j in JOINTS])


def get_joint_vel(model, data) -> np.ndarray:
    return np.array([data.qvel[dof_adr(model, j)] for j in JOINTS])


def leg_angles(model, data, leg: str) -> np.ndarray:
    """取某条腿的 (hip, thigh, knee) 当前角度。"""
    return np.array([data.qpos[qpos_adr(model, f"{leg}_{p}_j")] for p in ("hip", "thigh", "knee")])


# ---------------------------------------------------------------------------
# 数学：单腿逆运动学
# ---------------------------------------------------------------------------
def leg_ik(x: float, z: float, l1: float = L1, l2: float = L2) -> tuple[float, float]:
    """
    足端位置 -> (thigh, knee) 角度。

    参数
    ----
    x : 足端相对髋关节的前后位置（+x 向前，单位 m）
    z : 足端相对髋关节的上下位置（z 通常为负，单位 m）

    备注
    ----
    只用余弦定理 + atan2。膝关节角度取负值（膝盖向后弯，和模型一致）。
    """
    r = math.hypot(x, z)
    # 目标超出腿长时夹紧，避免 acos 出现 nan
    r = min(max(r, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)

    cos_inner = (l1 * l1 + l2 * l2 - r * r) / (2.0 * l1 * l2)
    cos_inner = max(-1.0, min(1.0, cos_inner))
    inner = math.acos(cos_inner)          # 膝盖内角
    knee = -(math.pi - inner)             # 弯折角，负值

    phi = math.atan2(-x, -z)              # 整条腿相对竖直方向的倾角
    beta = math.atan2(l2 * math.sin(knee), l1 + l2 * math.cos(knee))
    thigh = phi - beta
    return thigh, knee


# ---------------------------------------------------------------------------
# 数学：步态足端轨迹
# ---------------------------------------------------------------------------
def foot_targets(
    t: float,
    period: float = 0.5,
    stride: float = 0.12,
    lift: float = 0.05,
    duty: float = 0.55,
    height: float = STAND_H,
    offsets: dict | None = None,
    ramp_time: float = 0.5,
) -> dict[str, tuple[float, float]]:
    """
    给定时间 t，返回四条腿的足端目标 (x, z)（髋坐标系）。

    period : 一个完整步态周期（秒）
    stride : 一个周期内脚相对身体移动的距离（米）—— 决定速度 = stride / period
    lift   : 抬脚高度（米）
    duty   : 支撑相占比（0.5 = 一半时间着地）
    ramp_time : 起步过渡时间（秒），避免第一帧就把脚瞬移到奇怪的位置
    """
    offsets = TROT_OFFSETS if offsets is None else offsets

    # 起步时把步幅/抬腿从 0 平滑放大，避免开机瞬间"抽一下"
    if ramp_time > 0:
        amp = min(1.0, t / ramp_time)
        amp = amp * amp * (3.0 - 2.0 * amp)
    else:
        amp = 1.0

    out: dict[str, tuple[float, float]] = {}

    for leg, off in offsets.items():
        p = (t / period + off) % 1.0
        if p < duty:
            # ---- 支撑相：脚相对身体向后走直线 ----
            s = p / duty
            x = stride * (0.5 - s) * amp
            z = -height
        else:
            # ---- 摆动相：脚向前迈，同时抬起 ----
            s = (p - duty) / (1.0 - duty)
            x = stride * (-0.5 + 0.5 * (1.0 - math.cos(math.pi * s))) * amp
            z = -height + lift * math.sin(math.pi * s) * amp
        out[leg] = (x, z)
    return out


def apply_foot_targets(model, data, targets: dict[str, tuple[float, float]]):
    """
    把四条腿的足端目标写成 12 个执行器目标角。
    髋外摆保持 0（后面姿态闭环阶段会用到它）。
    """
    ctrl = STAND.copy()
    for i, leg in enumerate(LEGS):
        x, z = targets[leg]
        thigh, knee = leg_ik(x, z)
        ctrl[3 * i + 1] = thigh
        ctrl[3 * i + 2] = knee
    data.ctrl[:] = ctrl
    return ctrl


def balance_control(
    model,
    data,
    t: float,
    period: float = 0.5,
    stride: float = 0.12,
    lift: float = 0.05,
    duty: float = 0.55,
    height: float = STAND_H,
    offsets: dict | None = None,
    use_balance: bool = True,
    kp_pitch: float = 0.9,
    kp_roll: float = 0.9,
    kd_pitch: float = 0.03,
    kd_roll: float = 0.03,
    kp_height: float = 0.4,
) -> np.ndarray:
    """
    阶段 5 的完整传统控制器：正弦步态 + 姿态反馈，返回 12 个目标关节角。

    符号约定（dz > 0 = 腿变短）：
        dz = -kp_pitch * pitch * x_hip     # 低头 -> 前腿伸长
           + kp_roll  * roll  * y_hip      # 左侧抬高 -> 左腿缩短
    """
    roll, pitch, _ = base_rpy(data)
    (wx, wy, _), _ = body_velocity(model, data)

    targets = foot_targets(
        t, period=period, stride=stride, lift=lift, duty=duty,
        height=height, offsets=offsets,
    )

    ctrl = STAND.copy()
    for i, leg in enumerate(LEGS):
        x, z = targets[leg]
        if use_balance:
            dz = -kp_pitch * pitch * HIP_X[leg] + kp_roll * roll * HIP_Y[leg]
            dz += -kd_pitch * wy * HIP_X[leg] + kd_roll * wx * HIP_Y[leg]
            dz += kp_height * (float(data.qpos[2]) - STAND_Z)
            z = z + dz

        thigh, knee = leg_ik(x, z)
        ctrl[3 * i + 1] = thigh
        ctrl[3 * i + 2] = knee
    return ctrl


# ---------------------------------------------------------------------------
# 状态读取
# ---------------------------------------------------------------------------
def base_rpy(data) -> tuple[float, float, float]:
    """躯干欧拉角 (roll, pitch, yaw)，单位弧度。pitch > 0 表示"低头"。"""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, data.qpos[3:7])
    R = mat.reshape(3, 3)
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def body_velocity(model, data) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 (角速度, 线速度)，都在【躯干自身坐标系】下，单位 m/s 和 rad/s。
    这就是真机 IMU 能直接测到的东西。
    """
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "root")
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root_id, res, 1)
    return res[:3].copy(), res[3:].copy()


def foot_world_pos(model, data, leg: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site")
    return data.site_xpos[sid].copy()


def foot_in_hip_frame(model, data, leg: str) -> np.ndarray:
    """足端在髋关节坐标系中的位置，用来检查 IK 是否算对。"""
    hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip_body")
    hip_pos = data.xpos[hip_id]          # MuJoCo 3.x：全局刚体位置
    hip_mat = data.xmat[hip_id].reshape(3, 3)
    return hip_mat.T @ (foot_world_pos(model, data, leg) - hip_pos)


# ---------------------------------------------------------------------------
# 仿真循环
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def launch_viewer(model, data, headless: bool = False):
    """打开交互式查看器；没有图形界面时自动退化为无窗口模式。"""
    if headless:
        yield None
        return
    try:
        import mujoco.viewer          # 必须显式导入，否则 mujoco.viewer 不是 mujoco 的属性
        viewer = mujoco.viewer.launch_passive(model, data)
    except Exception as exc:  # 无 DISPLAY / 无 GPU / 没装 glfw 时
        print(f"[提示] 打不开查看器：{type(exc).__name__}: {exc}")
        print("       改为无窗口运行（可以加 --headless 关掉这条提示）。")
        yield None
        return
    try:
        yield viewer
    finally:
        viewer.close()


def run(
    model,
    data,
    duration: float,
    control=None,
    viewer=None,
    realtime: bool = True,
    speed: float = 1.0,
):
    """
    推进仿真 duration 秒。

    control(model, data, t) 每个物理步调用一次（t 单位为秒）。
    viewer 不为 None 时自动同步画面，并按真实时间节流。
    """
    dt = model.opt.timestep
    n = int(round(duration / dt))
    t0 = time.perf_counter()

    for i in range(n):
        t = i * dt
        if control is not None:
            control(model, data, t)
        mujoco.mj_step(model, data)

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            if realtime:
                lag = t0 + (i + 1) * dt / speed - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)


def sleep_forever(viewer, seconds: float = 0.05):
    """保持窗口不关闭，方便手动拖动视角。"""
    while viewer is not None and viewer.is_running():
        viewer.sync()
        time.sleep(seconds)

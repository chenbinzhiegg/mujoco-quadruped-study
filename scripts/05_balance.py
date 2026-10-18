"""
阶段 5：加一个姿态闭环 —— 传统控制真正开始"干活"

运行：
    python scripts/05_balance.py
    python scripts/05_balance.py --no-balance   # 关掉反馈，对比阶段 4

思路（比 LQR / MPC 简单得多，但很有效）：
    trot 的腿长是固定的，所以躯干一歪，某些脚就悬空/砸地。
    解决办法：根据躯干的 roll / pitch，实时改变四条腿的"目标腿长"。

    记 dz > 0 表示"腿变短"，dz < 0 表示"腿变长"，则：

        dz = -kp_pitch * pitch * x_hip     (低头 -> 前腿伸长，把车头撑起来)
           + kp_roll  * roll  * y_hip      (左侧抬高 -> 左腿缩短，把左边放下来)
           - kd * 角速度项                  (抑制左右/前后摆动)

    为什么是"伸长前腿"而不是"缩短前腿"？
    因为腿变长会把躯干推离地面，变短会把躯干拉向脚。
    所以要把姿态"推回去"，方向必须和直觉相反 —— 符号写反就会变成正反馈。
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from quad_utils import (
    STAND_Z,
    TROT_OFFSETS,
    WALK_OFFSETS,
    balance_control,
    base_rpy,
    body_velocity,
    launch_viewer,
    load,
    run,
)

# 反馈增益（先按几何量级取 1.0，再微调）
KP_PITCH = 0.9
KP_ROLL = 0.9
KD_PITCH = 0.03
KD_ROLL = 0.03
KP_HEIGHT = 0.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gait", choices=["trot", "walk"], default="trot")
    parser.add_argument("--period", type=float, default=0.5)
    parser.add_argument("--stride", type=float, default=0.12)
    parser.add_argument("--lift", type=float, default=0.05)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-balance", action="store_true", help="关闭姿态反馈做对比")
    args = parser.parse_args()

    use_balance = not args.no_balance
    if args.gait == "trot":
        offsets, duty = TROT_OFFSETS, 0.55
    else:
        offsets, duty = WALK_OFFSETS, 0.75

    model, data = load()
    x0 = float(data.qpos[0])
    stat = {"vx": [], "z": [], "roll": [], "pitch": [], "fall_t": None}

    print(f"姿态反馈: {'开' if use_balance else '关'}   步态: {args.gait}")
    print(f"{'时间(s)':>8} {'高度':>7} {'vx':>8} {'roll°':>8} {'pitch°':>8}")
    print("-" * 44)

    def control(m, d, t):
        # 正弦步态 + 姿态反馈，公式/符号说明见 quad_utils.balance_control()
        d.ctrl[:] = balance_control(
            m, d, t,
            period=args.period,
            stride=args.stride,
            lift=args.lift,
            duty=duty,
            offsets=offsets,
            use_balance=use_balance,
            kp_pitch=KP_PITCH,
            kp_roll=KP_ROLL,
            kd_pitch=KD_PITCH,
            kd_roll=KD_ROLL,
            kp_height=KP_HEIGHT,
        )

        _, lin = body_velocity(m, d)
        roll, pitch, _ = base_rpy(d)

        _, lin = body_velocity(m, d)
        stat["vx"].append(lin[0])
        stat["z"].append(float(d.qpos[2]))
        stat["roll"].append(roll)
        stat["pitch"].append(pitch)
        if stat["fall_t"] is None and d.qpos[2] < 0.15:
            stat["fall_t"] = t
            print(f"{t:>8.2f} {'--- 摔倒 ---':>26}")
        if abs(t % 1.0) < m.opt.timestep * 0.5:
            print(f"{t:>8.2f} {d.qpos[2]:>7.3f} {lin[0]:>8.3f} "
                  f"{np.degrees(roll):>8.1f} {np.degrees(pitch):>8.1f}")

    with launch_viewer(model, data, headless=args.headless) as viewer:
        run(model, data, args.seconds, control=control, viewer=viewer,
            realtime=viewer is not None)

    dist = float(data.qpos[0] - x0)
    survived = args.seconds if stat["fall_t"] is None else stat["fall_t"]
    print("-" * 44)
    print(f"前向位移        : {dist:+.2f} m")
    print(f"平均前向速度    : {np.mean(stat['vx']):+.3f} m/s")
    print(f"平均高度        : {np.mean(stat['z']):.3f} m")
    print(f"roll / pitch RMS: {np.degrees(np.sqrt(np.mean(np.square(stat['roll'])))):.1f}° / "
          f"{np.degrees(np.sqrt(np.mean(np.square(stat['pitch'])))):.1f}°")
    print(f"存活时间        : {survived:.2f} s / {args.seconds:.0f} s")
    print()
    print("结论：")
    print("  * 只是把'脚的目标高度'跟姿态挂钩，稳定性就明显变好。")
    print("  * 这就是传统控制的典型形态：建模 -> 找误差 -> 加反馈。")
    print("  * 但它仍然需要人手工调增益、手工设计步态，遇到地形/扰动就吃力。")
    print("  * 下一节：让强化学习自己找步态和反馈。")


if __name__ == "__main__":
    main()

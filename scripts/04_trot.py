"""
阶段 4：不训练也能走 —— 正弦步态（开环 trot）

运行：
    python scripts/04_trot.py
    python scripts/04_trot.py --gait walk      # 慢速四足步态，更稳
    python scripts/04_trot.py --headless --seconds 6

核心思想（就三句话）：
    1. 每条腿的脚在"支撑相"相对身体向后划，身体就被推着向前。
    2. 对角的两条腿相位差 180°（FL+RR 一组，FR+RL 一组），这就是 trot。
    3. 脚的轨迹用 4 个参数描述：周期、步幅、抬腿高度、支撑占比。

注意：这是【开环】控制，没有看姿态、没有看速度。
      所以它会走，但走不正、会越走越歪，甚至摔倒 —— 这正是下一节要解决的问题。
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
    apply_foot_targets,
    base_rpy,
    body_velocity,
    foot_targets,
    launch_viewer,
    load,
    run,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gait", choices=["trot", "walk"], default="trot")
    parser.add_argument("--period", type=float, default=0.5, help="步态周期(s)")
    parser.add_argument("--stride", type=float, default=0.12, help="单周期步幅(m)")
    parser.add_argument("--lift", type=float, default=0.05, help="抬腿高度(m)")
    parser.add_argument("--duty", type=float, default=None, help="支撑相占比")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.gait == "trot":
        offsets = TROT_OFFSETS
        duty = 0.55 if args.duty is None else args.duty
    else:
        offsets = WALK_OFFSETS
        duty = 0.75 if args.duty is None else args.duty

    model, data = load()
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])

    stat = {"vx": [], "z": [], "roll": [], "pitch": [], "fall_t": None}

    print(f"步态: {args.gait}  period={args.period}s  理论速度≈{args.stride / args.period:.2f} m/s")
    print(f"{'时间(s)':>8} {'躯干高度':>9} {'body_vx':>9} {'roll(deg)':>10} {'pitch(deg)':>11}")
    print("-" * 54)

    def control(m, d, t):
        targets = foot_targets(
            t,
            period=args.period,
            stride=args.stride,
            lift=args.lift,
            duty=duty,
            offsets=offsets,
        )
        apply_foot_targets(m, d, targets)

        _, lin = body_velocity(m, d)
        roll, pitch, _ = base_rpy(d)
        stat["vx"].append(lin[0])
        stat["z"].append(float(d.qpos[2]))
        stat["roll"].append(roll)
        stat["pitch"].append(pitch)

        if stat["fall_t"] is None and d.qpos[2] < 0.15:
            stat["fall_t"] = t
            print(f"{t:>8.2f} {'--- 摔倒了 ---':>30}")

        if abs(t % 1.0) < m.opt.timestep * 0.5:
            print(f"{t:>8.2f} {d.qpos[2]:>9.3f} {lin[0]:>9.3f} "
                  f"{np.degrees(roll):>10.1f} {np.degrees(pitch):>11.1f}")

    with launch_viewer(model, data, headless=args.headless) as viewer:
        run(model, data, args.seconds, control=control, viewer=viewer,
            realtime=viewer is not None)

    dist = float(np.hypot(data.qpos[0] - x0, data.qpos[1] - y0))
    print("-" * 54)
    print(f"前进距离       : {dist:.2f} m")
    print(f"平均前向速度   : {np.mean(stat['vx']):.3f} m/s")
    print(f"平均躯干高度   : {np.mean(stat['z']):.3f} m（站立 {STAND_Z:.3f}）")
    print(f"横滚/俯仰 标准差: {np.degrees(np.std(stat['roll'])):.1f}° / {np.degrees(np.std(stat['pitch'])):.1f}°")
    if stat["fall_t"] is not None:
        print(f"摔倒时间       : {stat['fall_t']:.2f} s")
    print()
    print("结论：")
    print("  * 纯开环步态能产生前进运动，但没有任何反馈。")
    print("  * 你会看到躯干左右晃、慢慢转圈、最终可能翻倒。")
    print("  * 下一节加一个极简的姿态闭环，先把它扶稳。")


if __name__ == "__main__":
    main()

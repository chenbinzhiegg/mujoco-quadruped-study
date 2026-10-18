"""
阶段 3：单腿逆运动学 —— 让一只脚在空中画圆

运行：
    python scripts/03_leg_ik.py
    python scripts/03_leg_ik.py --headless

为什么有用：
    后面所有"走路"的动作，本质都是"规划脚往哪走"，再翻译成关节角。
    会算 IK，你就不用再关心三个关节怎么配合了。

数学只需要一个公式（余弦定理）：
    已知两段腿长 L1、L2 和脚到髋的距离 r，
    膝盖弯曲角由 cos(内角) = (L1² + L2² - r²) / (2·L1·L2) 决定。
"""

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from quad_utils import (
    STAND,
    STAND_H,
    foot_in_hip_frame,
    launch_viewer,
    leg_ik,
    load,
    run,
)

CENTER = np.array([0.0, -0.24])   # 圆心 (x, z)，单位 m
RADIUS = 0.035                    # 画圆半径
PERIOD = 2.0                      # 转一圈用时


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()

    model, data = load()
    state = {"max_err": 0.0, "t_last": -1.0}

    print(f"只有左前腿 FL 在动，其它三条腿保持站立。")
    print(f"圆心 (x={CENTER[0]:+.2f}, z={CENTER[1]:+.2f})，半径 {RADIUS} m，周期 {PERIOD}s\n")
    print(f"{'时间(s)':>8} {'目标 x':>8} {'目标 z':>8} {'thigh(deg)':>11} {'knee(deg)':>11} {'实际误差(mm)':>13}")
    print("-" * 66)

    def control(m, d, t):
        # 1) 规划足端轨迹
        w = 2.0 * math.pi / PERIOD
        x = CENTER[0] + RADIUS * math.cos(w * t)
        z = CENTER[1] + RADIUS * math.sin(w * t)

        # 2) 逆运动学
        thigh, knee = leg_ik(x, z)

        # 3) 只有 FL 的关节动，其余保持站立
        d.ctrl[:] = STAND
        d.ctrl[1] = thigh
        d.ctrl[2] = knee

        # 4) 检查：仿真里的实际足端位置 vs 目标（取 x、z 两个分量）
        if t > 0.5:
            actual = foot_in_hip_frame(m, d, "FL")
            err = float(np.linalg.norm(actual[[0, 2]] - np.array([x, z])))
            state["max_err"] = max(state["max_err"], err)
            if t - state["t_last"] >= 1.0:
                state["t_last"] = t
                print(f"{t:>8.2f} {x:>8.3f} {z:>8.3f} "
                      f"{math.degrees(thigh):>11.1f} {math.degrees(knee):>11.1f} "
                      f"{err * 1000:>13.1f}")

    with launch_viewer(model, data, headless=args.headless) as viewer:
        run(model, data, args.seconds, control=control, viewer=viewer,
            realtime=viewer is not None)

    print("-" * 66)
    print(f"IK 跟踪最大误差: {state['max_err'] * 1000:.1f} mm")
    print()
    print("结论：")
    print("  * 给一个 (x, z)，IK 直接给出 (thigh, knee)，不用试凑。")
    print("  * 误差主要来自 PD 的柔性和接触，不是公式错。")
    print("  * 把 CENTER 的 z 改成 -0.2633，脚就落在地面上；改成 -0.22 就抬起来了。")
    print(f"  * 站立时髋到足心高度 = {STAND_H:.4f} m。")


if __name__ == "__main__":
    main()

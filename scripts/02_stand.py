"""
阶段 2：让机器人站起来、蹲下去（位置控制 = 最朴素的 PD）

运行：
    python scripts/02_stand.py
    python scripts/02_stand.py --kp 20      # 把 PD 的 P 调小，看它"软"下去
    python scripts/02_stand.py --headless   # 不开窗口，只看数据

看什么：
    1. ctrl 里写的是"目标关节角"，执行器内部用 PD 去追。
    2. 目标角用逆运动学算出来，所以改的是"身体高度"而不是关节角。
    3. kp 太小时机器人会被重力压塌 —— 这就是 PD 刚度的意义。

不需要懂的理论：
    只要知道"误差 = 目标 - 当前，电机出力 = kp * 误差 + kd * 误差变化率"。
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from quad_utils import STAND_H, leg_ik, get_joint_angles, launch_viewer, load, run

H_CROUCH = 0.20   # 蹲下时"髋 -> 足心"的高度


def smoothstep(s: float) -> float:
    s = max(0.0, min(1.0, s))
    return s * s * (3.0 - 2.0 * s)


def height_profile(t: float) -> float:
    """8 秒的剧本：站 -> 蹲 -> 停 -> 起 -> 站"""
    if t < 1.0:
        return STAND_H
    if t < 3.0:
        return STAND_H + (H_CROUCH - STAND_H) * smoothstep((t - 1.0) / 2.0)
    if t < 4.0:
        return H_CROUCH
    if t < 6.0:
        return H_CROUCH + (STAND_H - H_CROUCH) * smoothstep((t - 4.0) / 2.0)
    return STAND_H


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--kp", type=float, default=None, help="覆盖腿部执行器的 kp")
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()

    model, data = load()

    if args.kp is not None:
        for a in range(model.nu):
            model.actuator_gainprm[a, 0] = args.kp
            model.actuator_biasprm[a, 2] = -args.kp
        print(f"[实验] 已把 12 个执行器的 kp 改成 {args.kp}")

    print(f"{'时间(s)':>8} {'目标髋高':>9} {'实际髋高':>9} {'最大跟踪误差(deg)':>18}")
    print("-" * 52)

    state = {"last_print": -1.0, "max_err": 0.0}

    def control(m, d, t):
        h = height_profile(t)
        thigh, knee = leg_ik(0.0, -h)
        d.ctrl[:] = np.tile([0.0, thigh, knee], 4)

        # 调试输出
        err = np.abs(get_joint_angles(m, d) - d.ctrl)
        state["max_err"] = max(state["max_err"], float(err.max()))
        if t - state["last_print"] >= 0.5:
            state["last_print"] = t
            print(f"{t:>8.2f} {h:>9.4f} {d.qpos[2]:>9.4f} {np.degrees(err.max()):>18.2f}")

    with launch_viewer(model, data, headless=args.headless) as viewer:
        run(model, data, args.seconds, control=control, viewer=viewer,
            realtime=viewer is not None)

    print("-" * 52)
    print(f"全程最大关节跟踪误差: {np.degrees(state['max_err']):.2f} deg")
    print(f"结束时躯干高度: {data.qpos[2]:.4f} m（站立目标 {STAND_H + 0.022:.4f} m）")
    print()
    print("结论：")
    print("  * ctrl 是目标角，position 执行器自己完成 PD。")
    print("  * 用 IK 把'身体高度'翻译成三个关节角，比手动调关节角舒服得多。")
    print("  * --kp 调小后误差变大，说明刚度决定了'能不能撑住自己'。")


if __name__ == "__main__":
    main()

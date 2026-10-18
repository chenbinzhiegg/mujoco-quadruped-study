"""
阶段 1：把 XML 和真实机器人对应起来

运行：
    python scripts/01_inspect.py

这一节不需要跑仿真，只是把模型里的结构打印成一张"零件对照表"。
看懂这张表，你就知道以后改参数该改哪里。
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco

from quad_utils import load


def main():
    model, _ = load()

    print("\n【1】刚体（body）—— 相当于机械结构件")
    print(f"{'id':>3}  {'名称':<16} {'父级':<16} {'质量(kg)':>9}")
    print("-" * 50)
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "(无名)"
        parent = model.body_parentid[b]
        pname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) or "world"
        print(f"{b:>3}  {name:<16} {pname:<16} {model.body_mass[b]:>9.3f}")

    print("\n【2】关节（joint）—— 相当于电机 + 编码器")
    print(f"{'id':>3}  {'名称':<14} {'类型':<7} {'qpos下标':>8} {'范围(deg)':>18}")
    print("-" * 62)
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "(无名)"
        jtype = mujoco.mjtJoint(model.jnt_type[j]).name
        qadr = model.jnt_qposadr[j]
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j] * 180.0 / 3.141592653589793
            rng = f"[{lo:+.1f}, {hi:+.1f}]"
        else:
            rng = "无限位"
        print(f"{j:>3}  {name:<14} {jtype:<7} {qadr:>8} {rng:>18}")

    print("\n【3】执行器（actuator）—— 相当于电机驱动器里的 PD 环")
    print(f"{'id':>3}  {'名称':<14} {'控制关节':<14} {'kp':>7} {'kd(kv)':>8} {'力矩上限':>10}")
    print("-" * 68)
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "(无名)"
        jid = model.actuator_trnid[a, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "?"
        kp = model.actuator_gainprm[a, 0]
        kd = -model.actuator_biasprm[a, 2]   # position 执行器: bias = [0, -kp, -kv]
        lo, hi = model.actuator_forcerange[a]
        print(f"{a:>3}  {name:<14} {jname:<14} {kp:>7.1f} {kd:>8.1f} [{lo:+.0f}, {hi:+.0f}]")

    print("\n【4】传感器（sensor）—— 相当于真机上的 IMU / 编码器 / 足底开关")
    print(f"{'id':>3}  {'名称':<16} {'类型':<14}")
    print("-" * 40)
    for s in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s) or "(无名)"
        stype = mujoco.mjtSensor(model.sensor_type[s]).name
        print(f"{s:>3}  {name:<16} {stype:<14}")

    print("\n【5】关键数字")
    shin_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FL_shin_body")
    print(f"  单腿长度 L1 = {abs(model.body_pos[shin_body, 2]):.3f} m（髋到膝）")
    print(f"  站立髋高 = 0.15 + 0.15 的折线，垂直投影 0.2633 m")
    print(f"  躯干尺寸 = 0.44 x 0.20 x 0.09 m，质量 6 kg")
    print()
    print("对照记忆：")
    print("  body      -> 结构件（躯干、大腿、小腿）")
    print("  joint     -> 电机轴 + 编码器（能转多少度、转多快）")
    print("  actuator  -> 驱动器（这里用 position，相当于内置 PD）")
    print("  sensor    -> IMU、编码器、足底开关")
    print()


if __name__ == "__main__":
    main()

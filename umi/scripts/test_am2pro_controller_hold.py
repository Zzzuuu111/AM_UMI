"""Start the AM2Pro UMI controller, read state, then stop without a trajectory."""

import pathlib
import sys
import time


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from umi.real_world.am2pro_interpolation_controller import (  # noqa: E402
    AM2ProInterpolationController,
)


def main():
    # start() enables torque and writes each motor's *current* position as the
    # hold target.  This script never calls servoL, servoJ, schedule_waypoint,
    # or schedule_gripper.
    controller = AM2ProInterpolationController(
        robot_usb_port="/dev/ttyACM0",
        frequency=50,
        receive_latency=0.005,
        ik_backend="ros2_dh",
        flip_joints=["wrist_roll"],
        gripper_width_min=0.0,
        gripper_width_max=0.09,
        gripper_servo_closed=97.1,
        gripper_servo_open=1.4,
        server_python="/home/zzzjh/anaconda3/envs/AM_UMI/bin/python",
        verbose=True,
    )
    try:
        controller.start()
        time.sleep(1.0)
        state = controller.get_state()
        if state is None:
            raise RuntimeError("controller started but no robot state arrived")
        print("AM2PRO_CONTROLLER_HOLD_OK")
        print("tcp_pose:", state["ActualTCPPose"].round(5).tolist())
        print("joint_positions:", state["ActualQ"].round(3).tolist())
        print(f"gripper_width_m: {state['gripper_position']:.5f}")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()

base_link_name = "base_link"
foot_link_name = ".*_wheel"

# fmt: off
leg_joint_names = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
]
wheel_joint_names = [
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
]

hipx_joint_names = [
    "fl_hipx_joint", "fr_hipx_joint", "hl_hipx_joint", "hr_hipx_joint",
]

hipy_joint_names = [
    "fl_hipy_joint", "fr_hipy_joint", "hl_hipy_joint", "hr_hipy_joint",
]

knee_joint_names = [
    "fl_knee_joint", "fr_knee_joint", "hl_knee_joint", "hr_knee_joint",
]

arm_joint_names = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
]

gripper_joint_names = [
    "joint7", "joint8",
]

joint_names = leg_joint_names + wheel_joint_names
nav2_path_dataset_dir: str = "/home/sutd/Downloads/Pavan/m3_ros2_ws/src/nav_rl_bridge/rl_path_dataset/aws_warehouse"
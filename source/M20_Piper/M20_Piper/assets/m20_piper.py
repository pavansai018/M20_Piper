import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
from M20_Piper.assets import M20_PIPER_ASSET_DIR


M20_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(M20_PIPER_ASSET_DIR, 'M20_Piper.urdf'),
        fix_base=False,
        copy_from_source=False,
        activate_contact_sensors=True,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),

        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0,
                damping=0,
            ),
        ),
        
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.475,),
        joint_pos={
            '.*hipx_joint': 0.0,
            'f[l,r]_hipy_joint': -0.70,
            'h[l,r]_hipy_joint': 0.70,
            'f[l,r]_knee_joint': 1.4,
            'h[l,r]_knee_joint': -1.4,
            '.*_wheel_joint': 0.0,
            'joint[1-8]':0.0,
        },
        joint_vel={'.*': 0.0,},

    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        'joint': DelayedPDActuatorCfg(
            joint_names_expr=['.*hipx_joint', '.*hipy_joint', '.*knee_joint'],
            effort_limit=76.4,
            velocity_limit=22.4,
            stiffness=80.0,
            damping=8.0,  # increased from 2.0; near-critical for passive stance (no RL leg control)
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
        'wheel': DelayedPDActuatorCfg(
            joint_names_expr=['.*_wheel_joint'],
            effort_limit=21.6,
            velocity_limit=79.3,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
            armature=0.00243216,
            min_delay=0,
            max_delay=1,
        ),
        'arm': DelayedPDActuatorCfg(
            joint_names_expr=['joint[1-6]'],
            effort_limit=100,
            velocity_limit=3,
            stiffness=180.0,
            damping=45.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
        'gripper': DelayedPDActuatorCfg(
            joint_names_expr=['joint[7-8]'],
            effort_limit=10.0,
            velocity_limit=1.0,
            stiffness=80,
            damping=8.0,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
    },
)
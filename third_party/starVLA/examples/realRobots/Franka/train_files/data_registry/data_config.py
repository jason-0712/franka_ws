"""Franka benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig — Franka Delta EEF
# ---------------------------------------------------------------------------
class SingleFrankaRobotiqDeltaEefDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.base_view", "video.ego_view"]
    state_keys = ["state.eef_position", "state.eef_rotation"]
    action_keys = ["action.delta_eef_position", "action.delta_eef_rotation", "action.gripper_close"]
    # Per-key dims for PolicyNormProcessor (3+3+1 = 7-D total)
    action_key_dims = {"action.delta_eef_position": 3, "action.delta_eef_rotation": 3, "action.gripper_close": 1}
    state_key_dims  = {"state.eef_position": 3, "state.eef_rotation": 3}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={"state.eef_position": "min_max", "state.eef_rotation": "min_max"},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_eef_position": "min_max",
                    "action.delta_eef_rotation": "min_max",
                    "action.gripper_close": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — Franka Delta Joints (sim)
# ---------------------------------------------------------------------------
class SingleFrankaRobotiqDeltaJointsDataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = ["video.base_view", "video.ego_view"]
    state_keys = ["state.joints"]
    action_keys = ["action.delta_joints", "action.gripper_close"]
    # Per-key dims for PolicyNormProcessor (7+1 = 8-D total)
    action_key_dims = {"action.delta_joints": 7, "action.gripper_close": 1}
    state_key_dims  = {"state.joints": 7}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={"state.joints": "min_max"}),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={"action.delta_joints": "min_max", "action.gripper_close": "binary"},
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — Quest3 Franka Delta End-Effector Pose
# ---------------------------------------------------------------------------
class Quest3FrankaDeltaEefDataConfig:
    """Quest3 teleop data recorded as 7D Cartesian delta end-pose actions."""

    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = ["video.primary_image"]
    state_keys = ["state.eef_position", "state.eef_rotation", "state.gripper"]
    action_keys = ["action.delta_eef_position", "action.delta_eef_rotation", "action.gripper"]
    # Per-key dims for PolicyNormProcessor (3+3+1 = 7-D total)
    action_key_dims = {
        "action.delta_eef_position": 3,
        "action.delta_eef_rotation": 3,
        "action.gripper": 1,
    }
    state_key_dims = {
        "state.eef_position": 3,
        "state.eef_rotation": 3,
        "state.gripper": 1,
    }
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "q99",
                    "state.eef_rotation": "q99",
                    "state.gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_eef_position": "q99",
                    "action.delta_eef_rotation": "q99",
                    "action.gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — Quest3 Franka Dual-Camera Delta End-Effector Pose
# ---------------------------------------------------------------------------
class Quest3FrankaDualCamDeltaEefDataConfig(Quest3FrankaDeltaEefDataConfig):
    """Quest3 delta-EFF data with third-person and wrist camera views."""

    video_keys = ["video.primary_image", "video.wrist_image"]


# ---------------------------------------------------------------------------
# DataConfig — CRISP Franka Absolute Joint Targets
# ---------------------------------------------------------------------------
class CrispFrankaAbsJointsDataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = ["video.primary_image"]
    state_keys = ["state.joints", "state.gripper"]
    action_keys = ["action.target_joints", "action.gripper"]
    # Per-key dims for PolicyNormProcessor (7+1 = 8-D total)
    action_key_dims = {"action.target_joints": 7, "action.gripper": 1}
    state_key_dims = {"state.joints": 7, "state.gripper": 1}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joints": "q99",
                    "state.gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.target_joints": "q99",
                    "action.gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — CRISP Franka Delta Joint Targets
# ---------------------------------------------------------------------------
class CrispFrankaDeltaJointsDataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = ["video.primary_image"]
    state_keys = ["state.joints", "state.gripper"]
    action_keys = ["action.delta_joints", "action.gripper"]
    # Per-key dims for PolicyNormProcessor (7+1 = 8-D total)
    action_key_dims = {"action.delta_joints": 7, "action.gripper": 1}
    state_key_dims = {"state.joints": 7, "state.gripper": 1}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joints": "q99",
                    "state.gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_joints": "q99",
                    "action.gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — SO101
# ---------------------------------------------------------------------------
class SO101Config:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.primary_image", "video.wrist_image"]
    state_keys = [
        "state.shoulder_pan.pos", "state.shoulder_lift.pos", "state.elbow_flex.pos",
        "state.wrist_flex.pos", "state.wrist_roll.pos", "state.gripper.pos",
    ]
    action_keys = [
        "action.shoulder_pan.pos", "action.shoulder_lift.pos", "action.elbow_flex.pos",
        "action.wrist_flex.pos", "action.wrist_roll.pos", "action.gripper.pos",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={k: "min_max" for k in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={k: "min_max" for k in self.action_keys},
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "custom_robot_config": SingleFrankaRobotiqDeltaEefDataConfig(),
    "demo_sim_franka_delta_joints": SingleFrankaRobotiqDeltaJointsDataConfig(),
    "quest3_franka_delta_eef": Quest3FrankaDeltaEefDataConfig(),
    "quest3_franka_dualcam_delta_eef": Quest3FrankaDualCamDeltaEefDataConfig(),
    "crisp_franka_abs_joints": CrispFrankaAbsJointsDataConfig(),
    "crisp_franka_delta_joints": CrispFrankaDeltaJointsDataConfig(),
    "SO101": SO101Config(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}

DATASET_NAMED_MIXTURES = {
    "custom_dataset": [("custom_dataset_name", 1.0, "custom_robot_config")],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],
    "demo_sim_pick_place": [("sim_pick_place", 1.0, "demo_sim_franka_delta_joints")],
    "quest3_franka_dualcam_pickplace_50eps": [
        (
            "quest3_franka_dualcam_pickplace_50eps",
            1.0,
            "quest3_franka_dualcam_delta_eef",
        ),
    ],
    "quest3_franka_dualcam_pickplace_74eps": [
        (
            "quest3_franka_dualcam_pickplace_74eps",
            1.0,
            "quest3_franka_dualcam_delta_eef",
        ),
    ],
    "quest3_franka_pick_cube_place_box_40eps": [
        ("quest3_franka_tele_0021", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0024", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0025", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0026", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0027", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0028", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0030", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0031", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0032", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0033", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0034", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0035", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0036", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0037", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0038", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0039", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0040", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0041", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0042", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0043", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0044", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0045", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0046", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0047", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0048", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0049", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0050", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0051", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0052", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0053", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0054", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0055", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0056", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0057", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0058", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0059", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0060", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0061", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0062", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0063", 1.0, "quest3_franka_delta_eef"),
    ],
    "quest3_franka_pick_cube_place_box_debug": [
        ("quest3_franka_tele_0063", 1.0, "quest3_franka_delta_eef"),
    ],
    "quest3_franka_pick_cube_place_box_100eps_all_qwen": [
        ("quest3_franka_tele_0021", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0024", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0025", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0026", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0027", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0028", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0030", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0031", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0032", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0033", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0034", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0035", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0036", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0037", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0038", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0039", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0040", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0041", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0042", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0043", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0044", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0045", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0046", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0047", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0048", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0049", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0050", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0051", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0052", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0053", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0054", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0055", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0056", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0057", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0058", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0059", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0060", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0061", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0062", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0063", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0069", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0071", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0072", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0075", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0076", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0077", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0078", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0079", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0081", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0082", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0084", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0085", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0086", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0087", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0089", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0090", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0092", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0094", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0095", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0097", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0100", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0105", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0106", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0107", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0108", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0109", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0110", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0111", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0112", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0117", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0118", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0119", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0120", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0125", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0126", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0127", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0128", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0131", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0132", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0133", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0134", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0135", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0136", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0137", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0138", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0139", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0140", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0141", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0142", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0150", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0151", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0152", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0153", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0154", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0157", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0158", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0159", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0160", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0161", 1.0, "quest3_franka_delta_eef"),
        ("quest3_franka_tele_0162", 1.0, "quest3_franka_delta_eef"),
    ],
    "crisp_franka_pick_cube_place_bowl_20eps": [
        ("franka_test_135", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_136", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_137", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_139", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_140", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_141", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_144", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_146", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_148", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_149", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_150", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_151", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_152", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_153", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_154", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_155", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_156", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_157", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_158", 1.0, "crisp_franka_abs_joints"),
        ("franka_test_161", 1.0, "crisp_franka_abs_joints"),
    ],
    "crisp_franka_pick_cube_place_bowl_debug": [
        ("franka_test_161", 1.0, "crisp_franka_abs_joints"),
    ],
    "crisp_franka_pick_cube_place_bowl_delta_20eps": [
        ("franka_test_135", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_136", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_137", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_139", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_140", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_141", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_144", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_146", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_148", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_149", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_150", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_151", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_152", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_153", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_154", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_155", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_156", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_157", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_158", 1.0, "crisp_franka_delta_joints"),
        ("franka_test_161", 1.0, "crisp_franka_delta_joints"),
    ],
    "crisp_franka_pick_cube_place_bowl_delta_debug": [
        ("franka_test_161", 1.0, "crisp_franka_delta_joints"),
    ],
    "SO101_pick": [("pick_dataset_name", 1.0, "SO101")],
}

QUEST3_FRANKA_LATERAL11 = {
    "quest3_franka_tele_0150",
    "quest3_franka_tele_0151",
    "quest3_franka_tele_0152",
    "quest3_franka_tele_0153",
    "quest3_franka_tele_0154",
    "quest3_franka_tele_0157",
    "quest3_franka_tele_0158",
    "quest3_franka_tele_0159",
    "quest3_franka_tele_0160",
    "quest3_franka_tele_0161",
    "quest3_franka_tele_0162",
}

DATASET_NAMED_MIXTURES["quest3_franka_pick_cube_place_box_100eps_lateral11x3_weighted"] = [
    (dataset_name, 3.0 if dataset_name in QUEST3_FRANKA_LATERAL11 else weight, robot_type)
    for dataset_name, weight, robot_type in DATASET_NAMED_MIXTURES[
        "quest3_franka_pick_cube_place_box_100eps_all_qwen"
    ]
]

# Replay94 is the successful 74-episode dataset plus 10 front and 10 back
# demonstrations.  The policy server needs this name to reconstruct the
# training-time normalization contract when loading Replay94 checkpoints.
DATASET_NAMED_MIXTURES["quest3_franka_dualcam_replay_94eps_v1"] = [
    (
        "quest3_franka_dualcam_replay_94eps_v1",
        1.0,
        "quest3_franka_dualcam_delta_eef",
    ),
]

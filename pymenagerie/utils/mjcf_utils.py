from typing import List

import numpy as np
from dm_control import mjcf

from pymenagerie import types


def safe_find_all(
    root: types.MjcfRootElement,
    feature_name: str,
    immediate_children_only: bool = False,
    exclude_attachments: bool = False,
) -> List[mjcf.Element]:
    """Find all given elements or throw an error if none are found."""
    features = root.find_all(feature_name, immediate_children_only, exclude_attachments)
    if not features:
        raise ValueError(f"No {feature_name} found in the MJCF model.")
    return features


def safe_find(
    root: types.MjcfRootElement,
    namespace: str,
    identifier: str,
) -> mjcf.Element:
    """Find the given element or throw an error if not found."""
    feature = root.find(namespace, identifier)
    if feature is None:
        raise ValueError(f"{namespace} with the specified {identifier} not found.")
    return feature


def attach_hand_to_arm(
    arm_mjcf: types.MjcfRootElement,
    hand_mjcf: types.MjcfRootElement,
) -> None:
    physics = mjcf.Physics.from_mjcf_model(hand_mjcf)

    attachment_site = arm_mjcf.find("site", "attachment_site")
    if attachment_site is None:
        raise ValueError("No attachment site found in the arm model.")

    # Expand the ctrl and qpos keyframes to account for the new hand DoFs.
    arm_key = arm_mjcf.find("key", "home")
    if arm_key is not None:
        hand_key = hand_mjcf.find("key", "home")
        if hand_key is None:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, np.zeros(physics.model.nu)])
            arm_key.qpos = np.concatenate([arm_key.qpos, np.zeros(physics.model.nq)])
        else:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, hand_key.ctrl])
            arm_key.qpos = np.concatenate([arm_key.qpos, hand_key.qpos])

    attachment_site.attach(hand_mjcf)


def get_critical_damping_from_stiffness(
    stiffness: float, joint_name: str, model: types.MjcfRootElement
) -> float:
    """Compute the critical damping coefficient for a given stiffness.

    Args:
        stiffness: The stiffness coefficient.
        joint_name: The name of the joint to compute the critical damping for.
        model: The MJCF model.

    Returns:
        The critical damping coefficient.
    """
    physics = mjcf.Physics.from_mjcf_model(model)
    joint_id = physics.named.model.jnt_qposadr[joint_name]
    joint_mass = physics.model.dof_M0[joint_id]
    return 2 * np.sqrt(joint_mass * stiffness)


def compensate_gravity(model: types.MjcfRootElement) -> None:
    """Applies gravity compensation to the all bodies in the model.

    Args:
        model: The MJCF model.

    Raises:
        ValueError: If the MuJoCo version is less than 2.3.1.
    """
    for body in model.find_all("body"):
        # A value of 1.0 creates an upward force equal to the body’s weight and
        # compensates for gravity exactly.
        body.gravcomp = 1.0

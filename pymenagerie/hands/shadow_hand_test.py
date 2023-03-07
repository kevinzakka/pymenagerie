"""Tests for shadow_hand.py."""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf

from pymenagerie.hands import shadow_hand
from pymenagerie.hands import shadow_hand_constants as consts


def _get_env() -> composer.Environment:
    robot = shadow_hand.ShadowHand()
    task = composer.NullTask(root_entity=robot)
    return composer.Environment(task=task, strip_singleton_obs_buffer_dim=True)


class ShadowHandTest(parameterized.TestCase):
    def test_compiles_and_steps(self) -> None:
        robot = shadow_hand.ShadowHand()
        physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
        physics.step()

    def test_set_name(self) -> None:
        robot = shadow_hand.ShadowHand(name="larry")
        self.assertEqual(robot.name, "larry")
        self.assertEqual(robot.mjcf_model.model, "larry")

    def test_joints(self) -> None:
        robot = shadow_hand.ShadowHand()
        for joint in robot.joints:
            self.assertEqual(joint.tag, "joint")
        self.assertLen(robot.joints, consts.NQ)

    def test_actuators(self) -> None:
        robot = shadow_hand.ShadowHand()
        for actuator in robot.actuators:
            self.assertEqual(actuator.tag, "position")
        self.assertLen(robot.actuators, consts.NU)


class ShadowHandObservableTest(parameterized.TestCase):
    @parameterized.parameters(
        [
            "root_body",
        ]
    )
    def test_get_element_property(self, name: str) -> None:
        attribute_value = getattr(shadow_hand.ShadowHand(), name)
        self.assertIsInstance(attribute_value, mjcf.Element)

    @parameterized.parameters(
        [
            "actuators",
            "joints",
            "joint_torque_sensors",
        ]
    )
    def test_get_element_tuple_property(self, name: str) -> None:
        attribute_value = getattr(shadow_hand.ShadowHand(), name)
        self.assertNotEmpty(attribute_value)
        for element in attribute_value:
            self.assertIsInstance(element, mjcf.Element)

    @parameterized.parameters(
        [
            "joints_pos",
            "joints_pos_cos_sin",
            "joints_vel",
            "joints_torque",
            "position",
        ]
    )
    def test_evaluate_observable(self, name: str) -> None:
        env = _get_env()
        physics = env.physics
        observable = getattr(env.task.root_entity.observables, name)
        observation = observable(physics)
        self.assertIsInstance(observation, (float, np.ndarray))


if __name__ == "__main__":
    absltest.main()

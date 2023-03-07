"""Shadow hand composer class."""

from typing import Optional, Sequence

from dm_control import composer
from dm_control import mjcf

from pymenagerie import types
from pymenagerie.hands import base
from pymenagerie.hands import shadow_hand_constants as consts
from pymenagerie.utils import mjcf_utils

_RESTRICTED_WRJ2_RANGE: tuple[float, float] = (-0.174533, 0.174533)


class ShadowHand(base.Hand):
    """A Shadow Hand E3M5."""

    def _build(
        self,
        name: Optional[str] = None,
        side: base.HandSide = base.HandSide.RIGHT,
        primitive_fingertip_collisions: bool = False,
        restrict_wrist_yaw_range: bool = False,
    ) -> None:
        """Initializes a ShadowHand.

        Args:
            name: Name of the hand. Used as a prefix in the MJCF name attributes.
            side: Which side (left or right) to model.
            primitive_fingertip_collisions: Whether to use capsule approximations for
                the fingertip colliders or the true meshes. Using primitive colliders
                speeds up the simulation.
            restrict_wrist_yaw_range: Whether to restrict the range of the wrist yaw
                joint.
        """
        if side == base.HandSide.RIGHT:
            xml_file = consts.RIGHT_SHADOW_HAND_XML
            if name is None:
                name = "right_shadow_hand"
        elif side == base.HandSide.LEFT:
            xml_file = consts.LEFT_SHADOW_HAND_XML
            if name is None:
                name = "left_shadow_hand"

        self._hand_side = side
        self._mjcf_root = mjcf.from_path(str(xml_file))
        self._mjcf_root.model = name

        if restrict_wrist_yaw_range:
            joint = mjcf_utils.safe_find(self._mjcf_root, "joint", "WRJ2")
            joint.range = _RESTRICTED_WRJ2_RANGE
            actuator = mjcf_utils.safe_find(self._mjcf_root, "actuator", "A_WRJ2")
            actuator.ctrlrange = _RESTRICTED_WRJ2_RANGE

        self._parse_mjcf_elements()
        self._add_mjcf_elements()

        if primitive_fingertip_collisions:
            for geom in self._mjcf_root.find_all("geom"):
                if (
                    geom.dclass.dclass == "plastic_collision"
                    and geom.mesh is not None
                    and geom.mesh.name is not None
                    and geom.mesh.name.endswith("distal_pst")
                ):
                    geom.type = "capsule"

        self._action_spec = None

    def _parse_mjcf_elements(self) -> None:
        joints = mjcf_utils.safe_find_all(self._mjcf_root, "joint")
        actuators = mjcf_utils.safe_find_all(self._mjcf_root, "actuator")
        self._joints = tuple(joints)
        self._actuators = tuple(actuators)

    def _add_mjcf_elements(self) -> None:
        # Add joint torque sensors.
        joint_torque_sensors = []
        for joint_elem in self._joints:
            site_elem = joint_elem.parent.add(
                "site",
                name=joint_elem.name + "_site",
                size=(0.001, 0.001, 0.001),
                type="box",
                rgba=(0, 1, 0, 1),
                group=composer.SENSOR_SITES_GROUP,
            )
            torque_sensor_elem = joint_elem.root.sensor.add(
                "torque",
                site=site_elem,
                name=joint_elem.name + "_torque",
            )
            joint_torque_sensors.append(torque_sensor_elem)
        self._joint_torque_sensors = tuple(joint_torque_sensors)

    # Accessors.

    @property
    def hand_side(self) -> base.HandSide:
        return self._hand_side

    @property
    def mjcf_model(self) -> types.MjcfRootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._mjcf_root.model

    @composer.cached_property
    def root_body(self) -> types.MjcfElement:
        name = "rh_forearm" if self._hand_side == base.HandSide.RIGHT else "lh_forearm"
        return mjcf_utils.safe_find(self._mjcf_root, "body", name)

    @property
    def joints(self) -> Sequence[types.MjcfElement]:
        return self._joints

    @property
    def actuators(self) -> Sequence[types.MjcfElement]:
        return self._actuators

    @property
    def joint_torque_sensors(self) -> Sequence[types.MjcfElement]:
        return self._joint_torque_sensors

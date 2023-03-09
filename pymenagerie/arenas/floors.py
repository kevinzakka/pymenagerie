"""Floor arenas."""

from mujoco_utils import composer_utils

# This specifies the spacing between the grid subdivisions of the plane for rendering
# purposes.
_GROUNDPLANE_QUAD_SIZE = 0.05


class CheckeredFloor(composer_utils.Arena):
    """An arena with a checkered ground plane."""

    def _build(
        self,
        name: str = "floor",
        size: tuple[float, float] = (0, 0),
        reflectance: float = 0.2,
        rgb1: tuple[float, float, float] = (0.2, 0.3, 0.4),
        rgb2: tuple[float, float, float] = (0.1, 0.2, 0.3),
        markrgb: tuple[float, float, float] = (0.8, 0.8, 0.8),
    ) -> None:
        super()._build(name=name)

        self._size = size

        # Add skybox and groundplane textures.
        self._mjcf_root.asset.add(
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1=(0.3, 0.5, 0.7),
            rgb2=(0, 0, 0),
            width=512,
            height=3072,
        )
        self._mjcf_root.asset.add(
            "texture",
            type="2d",
            name="groundplane",
            builtin="checker",
            mark="edge",
            rgb1=rgb1,
            rgb2=rgb2,
            markrgb=markrgb,
            width=300,
            height=300,
        )
        self._mjcf_root.asset.add(
            "material",
            name="groundplane",
            texture="groundplane",
            texuniform=True,
            texrepeat=(5, 5),
            reflectance=reflectance,
        )

        # Build groundplane.
        self._ground_geom = self._mjcf_root.worldbody.add(
            "geom",
            type="plane",
            name="groundplane",
            material="groundplane",
            size=list(size) + [_GROUNDPLANE_QUAD_SIZE],
            condim=3,
        )

    @property
    def ground_geoms(self):
        return (self._ground_geom,)

    @property
    def size(self) -> tuple[float, float]:
        return self._size

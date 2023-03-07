"""Tests for floors.py."""

from absl.testing import absltest
from dm_control import mjcf

from pymenagerie.arenas import floors


class CheckeredFloorTest(absltest.TestCase):
    def test_compiles_and_steps(self) -> None:
        arena = floors.CheckeredFloor()
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        physics.step()


if __name__ == "__main__":
    absltest.main()

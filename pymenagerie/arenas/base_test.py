"""Tests for base.py."""

from absl.testing import absltest
from dm_control import mjcf

from pymenagerie.arenas import base as base_arena


class ArenaTest(absltest.TestCase):
    def test_compiles_and_steps(self) -> None:
        arena = base_arena.Arena()
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        physics.step()


if __name__ == "__main__":
    absltest.main()

"""CPU coverage of batched in-hand placement contacts and model rebinding."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from abc_sim.task_eval.inhand_transfer import InhandTransferOtherSideEvaluator
from abc_sim.task_specs import get_task_spec

_SPEC = get_task_spec("inhand transfer the item to other side")


def make_model(*, extra_joint=False):
    extra = (
        '<body pos="3 0 0"><joint type="slide"/><geom type="sphere" size=".01" contype="0" conaffinity="0"/></body>'
        if extra_joint
        else ""
    )
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="table_plane" type="plane" size="1 1 .1"/>'
        + extra
        + '<body name="object" pos="0 -.2 .019"><freejoint name="task_object_joint"/>'
        '<geom name="object_geom" type="sphere" size=".02" mass=".1"/></body>'
        '<body name="left_link_left_finger" pos="2 0 0"><freejoint name="left_finger"/>'
        '<geom name="left_finger_geom" type="sphere" size=".01" mass=".1"/></body>'
        '<body name="right_link_right_finger" pos="-2 0 0"><freejoint name="right_finger"/>'
        '<geom name="right_finger_geom" type="sphere" size=".01" mass=".1"/></body>'
        "</worldbody></mujoco>"
    )


class InhandContactsTest(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.evaluator = self.make_evaluator(self.model)

    @staticmethod
    def configure(evaluator, model):
        evaluator.configure_from_randomization(
            model,
            SimpleNamespace(
                metadata={"side": "left"},
                object_states={"task_object_joint": {"pos": [0, 0.2, 0.02]}},
            ),
        )

    def make_evaluator(self, model):
        evaluator = InhandTransferOtherSideEvaluator(model=model, spec=_SPEC)
        self.configure(evaluator, model)
        return evaluator

    def qpos_batch(self, model):
        batch = np.repeat(model.qpos0[None], 6, axis=0).astype(np.float32)
        obj = model.joint("task_object_joint").qposadr[0]
        batch[1, obj + 2] = 0.08
        batch[2, obj + 1] = 0.2
        left = model.joint("left_finger").qposadr[0]
        right = model.joint("right_finger").qposadr[0]
        batch[3, left : left + 3] = [0, -0.2, 0.04]
        batch[4, right : right + 3] = [0, -0.2, 0.04]
        batch[5, obj] = 1.1
        return batch

    def test_batched_contacts_match_full_forward_for_every_world(self):
        qpos = self.qpos_batch(self.model)
        for batch in (qpos, qpos[::-1]):
            with self.subTest(reversed=batch is not qpos):
                self.evaluator.reset(nworld=len(batch))
                result = self.evaluator.evaluate_qpos_batch(batch)
                expected = []
                for world in batch:
                    data = mujoco.MjData(self.model)
                    data.qpos[:] = world
                    mujoco.mj_forward(self.model, data)
                    reference = self.make_evaluator(self.model)
                    expected.append(reference.evaluate_model_data(self.model, data))
                np.testing.assert_array_equal(result.success, [r.scalar_success() for r in expected])
                np.testing.assert_array_equal(result.reward, [r.scalar_reward() for r in expected])
                for key in ("object_table_contact", "object_gripper_contact", "object_placed", "ever_success"):
                    np.testing.assert_array_equal(result.metrics[key], [r.metrics[key][0] for r in expected])
        np.testing.assert_array_equal(result.success[::-1], [True, False, False, False, False, False])
        np.testing.assert_array_equal(
            result.metrics["object_table_contact"][::-1], [True, False, True, True, True, True]
        )
        np.testing.assert_array_equal(
            result.metrics["object_gripper_contact"][::-1], [False, False, False, True, True, False]
        )

    def test_repeated_batches_reuse_contact_data_without_full_dynamics(self):
        qpos = self.qpos_batch(self.model)
        with (
            mock.patch(
                "abc_sim.task_eval.inhand_transfer.mujoco.MjData", side_effect=AssertionError("new scratch data")
            ),
            mock.patch(
                "abc_sim.task_eval.inhand_transfer.mujoco.mj_forward", side_effect=AssertionError("full dynamics")
            ),
        ):
            self.evaluator.evaluate_qpos_batch(qpos)
            self.evaluator.reset(nworld=1)
            result = self.evaluator.evaluate_qpos_batch(qpos[:1])
        self.assertTrue(result.scalar_success())
        self.assertFalse(result.metrics["object_gripper_contact"][0])

    def test_rebinding_replaces_scratch_data_for_new_model_layout(self):
        old_scratch = self.evaluator._contact_data
        self.evaluator.evaluate_qpos_batch(self.qpos_batch(self.model))
        model = make_model(extra_joint=True)
        self.configure(self.evaluator, model)
        self.assertIsNot(self.evaluator._contact_data, old_scratch)
        self.assertEqual(self.evaluator._contact_data.qpos.shape, (model.nq,))
        result = self.evaluator.evaluate_qpos_batch(self.qpos_batch(model))
        np.testing.assert_array_equal(result.success, [True, False, False, False, False, False])
        np.testing.assert_array_equal(
            result.metrics["object_gripper_contact"], [False, False, False, True, True, False]
        )

    def test_obsolete_height_option_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "max_resting_height_m"):
            InhandTransferOtherSideEvaluator(model=self.model, spec=_SPEC, max_resting_height_m=0.2)


if __name__ == "__main__":
    unittest.main()

"""CPU grasp regressions for counting and sorting, using the authored scenes.

Prepare assets with ``python prepare.py --sim-task count_into_opaque_box``,
``--sim-task nuts_bolts_sorting``, and ``--sim-task lego_blocks_sorting``.
No camera renderer or learned policy is used.
"""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import abc_sim


class GripperContactsTest(unittest.TestCase):
    def _make_env(self, task: str, *, counting_sphere: bool = False):
        scene = abc_sim.get_task_scene_xml(task)
        root = ET.parse(scene).getroot()
        meshdir = scene.parent / root.find("compiler").get("meshdir")
        missing = [
            mesh.get("file")
            for mesh in root.findall("./asset/mesh")
            if mesh.get("file") and not (meshdir / mesh.get("file")).is_file()
        ]
        if missing:
            self.skipTest(f"Prepare task assets: python prepare.py --sim-task {task}")
        if counting_sphere:
            # A supported counting-item shape; isolate its small contact patch.
            sphere = root.find(".//geom[@name='count_item_1_geom']")
            sphere.set("type", "sphere")
            sphere.set("size", "0.021")
            sphere.set("mass", "0.016")
        env = abc_sim.make_env(
            task=task,
            scene_xml_string=ET.tostring(root, encoding="unicode"),
            render_cameras=False,
            enable_task_randomizer=not counting_sphere,
            chunk_dim=1,
            control_decimation=1,
        )
        self.addCleanup(env.close)
        env.reset(seed=7, randomize=not counting_sphere)
        return env

    def _check_grasps(self, env, body_name: str, close_seconds: float = 1.5) -> None:
        model = env.model
        gravity = model.opt.gravity.copy()
        self.addCleanup(np.copyto, model.opt.gravity, gravity)
        pads = {
            geom
            for geom in range(model.ngeom)
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
            and model.body(int(model.geom_bodyid[geom])).name.endswith(
                ("_lf_down", "_rf_down")
            )
        }
        self.assertEqual(len(pads), 8)
        for arm in ("left", "right"):
            pad_materials = set()
            for offset in (0.017, 0.030, 0.044):
                with self.subTest(part=body_name, arm=arm, offset=offset):
                    pad_materials.update(
                        self._check_lift(
                            model,
                            env.data,
                            body_name,
                            arm,
                            offset,
                            close_seconds,
                            gravity,
                            pads,
                        )
                    )
            # Some offsets engage the fingertip spheres. Each arm must also exercise
            # the pads and use their higher-priority contact material during a lift.
            with self.subTest(part=body_name, arm=arm, check="resolved_material"):
                self.assertTrue(
                    pad_materials, f"{arm} grasps did not exercise the finger pads"
                )
                for solref, solimp, friction in pad_materials:
                    np.testing.assert_allclose(solref, [0.004, 1])
                    np.testing.assert_allclose(solimp, [0.998, 0.998, 0.001, 0.5, 2])
                    np.testing.assert_allclose(friction, [4, 4, 0.1, 0.01, 0.01])

    def _check_lift(
        self,
        model: mujoco.MjModel,
        reset_data: mujoco.MjData,
        body_name: str,
        arm: str,
        offset: float,
        close_seconds: float,
        gravity: np.ndarray,
        pads: set[int],
    ) -> set:
        model.opt.gravity[:] = 0
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, reset_data)
        body = model.body(body_name).id
        joint = int(model.body_jntadr[body])
        address = int(model.jnt_qposadr[joint])
        part_geoms = set(np.flatnonzero(model.geom_bodyid == body))
        finger_geoms = {
            geom
            for geom in range(model.ngeom)
            if model.body(int(model.geom_bodyid[geom])).name.startswith(
                (f"{arm}_lf_", f"{arm}_rf_")
            )
        }
        # Remove other loose objects from this isolated grasp without changing contacts.
        for other in range(model.njnt):
            if model.jnt_type[other] == mujoco.mjtJoint.mjJNT_FREE and other != joint:
                other_address = int(model.jnt_qposadr[other])
                data.qpos[other_address : other_address + 3] = [10 + other, 10, 10]
        for hand in ("left", "right"):
            data.joint(hand + "_left_finger").qpos[0] = 0.0475
            data.joint(hand + "_right_finger").qpos[0] = -0.0475
        for actuator in range(model.nu):
            actuator_joint = int(model.actuator_trnid[actuator, 0])
            data.ctrl[actuator] = data.qpos[model.jnt_qposadr[actuator_joint]]
        mujoco.mj_forward(model, data)
        site_id = model.site(arm + "_grasp_site").id
        site = data.site(site_id)
        rotation = site.xmat.reshape(3, 3).copy()
        grasp_local = np.array([offset, 0, -0.020])
        initial_point = site.xpos + rotation @ grasp_local
        data.qpos[address : address + 3] = (
            initial_point - rotation @ model.body_ipos[body]
        )
        mujoco.mju_mat2Quat(data.qpos[address + 3 : address + 7], rotation.ravel())
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            if set(contact.geom) & part_geoms:
                self.assertGreaterEqual(
                    contact.dist, -1e-5, "Grasp starts in penetration"
                )

        contact_count = 0
        max_penetration = 0.0
        pad_materials = set()

        def step() -> None:
            nonlocal contact_count, max_penetration
            mujoco.mj_step(model, data)
            for contact in data.contact:
                pair = set(contact.geom)
                if pair & part_geoms and pair & finger_geoms:
                    contact_count += 1
                    max_penetration = max(max_penetration, -float(contact.dist))
                    if pair & pads:
                        pad_materials.add(
                            (
                                tuple(contact.solref),
                                tuple(contact.solimp),
                                tuple(contact.friction),
                            )
                        )

        dt = model.opt.timestep
        gripper = model.actuator(arm + "_gripper").id
        for n in range(round(close_seconds / dt)):
            data.ctrl[gripper] = 0.0475 * max(0, 1 - n * dt / 0.5)
            step()
        mujoco.mj_forward(model, data)
        start_site = data.site(site_id).xpos.copy()
        target_position = start_site + [0, 0, 0.080]
        target_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(target_quaternion, data.site(site_id).xmat)
        joints = [model.joint(f"{arm}_joint{i}").id for i in range(1, 7)]
        qpos_indices = model.jnt_qposadr[joints]
        dof_indices = model.jnt_dofadr[joints]
        actuators = [model.actuator(f"{arm}_joint{i}").id for i in range(1, 7)]
        kinematics = mujoco.MjData(model)
        mujoco.mj_copyData(kinematics, model, data)
        # Solve a single 8 cm lift while retaining the grasp orientation.
        for _ in range(200):
            mujoco.mj_forward(model, kinematics)
            site = kinematics.site(site_id)
            quaternion, angular_error = np.empty(4), np.empty(3)
            mujoco.mju_mat2Quat(quaternion, site.xmat)
            mujoco.mju_subQuat(angular_error, target_quaternion, quaternion)
            angular_error = site.xmat.reshape(3, 3) @ angular_error
            error = np.r_[target_position - site.xpos, angular_error]
            if np.linalg.norm(error) < 1e-5:
                break
            jac_position = np.zeros((3, model.nv))
            jac_rotation = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, kinematics, jac_position, jac_rotation, site_id)
            jacobian = np.vstack(
                [jac_position[:, dof_indices], jac_rotation[:, dof_indices]]
            )
            kinematics.qpos[qpos_indices] += jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + np.eye(6) * 1e-5, error
            )
        self.assertLess(np.linalg.norm(error), 1e-5, "Lift target is unreachable")
        goal = kinematics.qpos[qpos_indices].copy()
        self.assertTrue(np.all(goal >= model.jnt_range[joints, 0]))
        self.assertTrue(np.all(goal <= model.jnt_range[joints, 1]))
        before = data.qpos[qpos_indices].copy()
        model.opt.gravity[:] = gravity
        for n in range(round(2.5 / dt)):
            data.ctrl[actuators] = before + (goal - before) * min(1, (n + 1) * dt / 1.5)
            step()
        mujoco.mj_forward(model, data)
        site = data.site(site_id)
        retention_error = np.linalg.norm(
            site.xmat.reshape(3, 3).T @ (data.xipos[body] - site.xpos) - grasp_local
        )
        part_lift = data.xipos[body, 2] - initial_point[2]
        tcp_lift = site.xpos[2] - start_site[2]
        measurements = {
            "contacts": contact_count,
            "penetration_mm": max_penetration * 1000,
            "retention_error_mm": retention_error * 1000,
            "part_lift_mm": part_lift * 1000,
            "tcp_lift_mm": tcp_lift * 1000,
        }
        self.assertTrue(np.isfinite(data.qpos).all(), measurements)
        self.assertGreater(contact_count, 0, measurements)
        self.assertLess(max_penetration, 0.0015, measurements)
        self.assertLess(retention_error, 0.005, measurements)
        self.assertGreater(part_lift, 0.070, measurements)
        self.assertGreater(tcp_lift, 0.070, measurements)
        return pad_materials

    def test_counting_sphere_survives_gravity_loaded_lift(self) -> None:
        env = self._make_env("count_into_opaque_box", counting_sphere=True)
        self._check_grasps(env, "count_item_1", close_seconds=2.0)

    def test_nut_survives_gravity_loaded_lift(self) -> None:
        self._check_grasps(self._make_env("nuts_bolts_sorting"), "nut_1")

    def test_bolt_survives_gravity_loaded_lift(self) -> None:
        self._check_grasps(self._make_env("nuts_bolts_sorting"), "bolt_1")

    def test_lego_survives_gravity_loaded_lift(self) -> None:
        self._check_grasps(self._make_env("lego_blocks_sorting"), "red_lego_1")


if __name__ == "__main__":
    unittest.main()

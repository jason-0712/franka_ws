import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "third_party" / "starVLA"


class VendoredStarVLARuntimeTest(unittest.TestCase):
    def test_required_policy_server_files_exist(self):
        required = [
            "deployment/model_server/server_policy.py",
            "deployment/model_server/policy_wrapper.py",
            "deployment/model_server/policy_norm_processor.py",
            "deployment/model_server/tools/websocket_policy_server.py",
            "starVLA/model/framework/VLM4A/QwenGR00T.py",
            "starVLA/model/modules/action_model/GR00T_ActionHeader.py",
            "starVLA/model/modules/vlm/QWen2_5.py",
            "starVLA/model/modules/vlm/QWen3.py",
            "examples/realRobots/Franka/train_files/data_registry/data_config.py",
            "LICENSE",
        ]
        for relative in required:
            with self.subTest(path=relative):
                self.assertTrue((RUNTIME / relative).is_file())

    def test_python_sources_parse(self):
        for source in RUNTIME.rglob("*.py"):
            with self.subTest(path=source.relative_to(RUNTIME)):
                ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    def test_franka_inference_registry_is_present(self):
        registry = (
            RUNTIME
            / "examples/realRobots/Franka/train_files/data_registry/data_config.py"
        ).read_text(encoding="utf-8")
        for marker in (
            "quest3_franka_dualcam_pickplace_74eps",
            "quest3_franka_pick_cube_place_box_100eps_all_qwen",
            "quest3_franka_delta_eef",
            "quest3_franka_dualcam_delta_eef",
            "crisp_franka_abs_joints",
            "crisp_franka_delta_joints",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, registry)

    def test_experimental_and_large_artifacts_are_excluded(self):
        forbidden_fragments = ("spatial_forcing", "vggt", "sam2", "rlinf")
        forbidden_suffixes = (
            ".pt",
            ".pth",
            ".safetensors",
            ".ckpt",
            ".parquet",
            ".mp4",
        )
        for path in RUNTIME.rglob("*"):
            if not path.is_file():
                continue
            relative = str(path.relative_to(RUNTIME)).lower()
            if path.name != "README.md":
                for fragment in forbidden_fragments:
                    with self.subTest(path=relative, fragment=fragment):
                        self.assertNotIn(fragment, relative)
                        if path.suffix == ".py":
                            self.assertNotIn(
                                fragment,
                                path.read_text(encoding="utf-8").lower(),
                            )
            self.assertNotIn(path.suffix.lower(), forbidden_suffixes)
            self.assertLess(path.stat().st_size, 10 * 1024 * 1024)

    def test_request_local_inference_seed_is_retained(self):
        wrapper = (
            RUNTIME / "deployment/model_server/policy_wrapper.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"supports_inference_seed": True', wrapper)
        self.assertIn("torch.random.fork_rng", wrapper)


if __name__ == "__main__":
    unittest.main()

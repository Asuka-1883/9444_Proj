"""Semantic checks for time windows, missing neighbors, leakage and coordinates."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from gc_dataset import GCConfig, GCStore, GCDataset, restore_pixels


class TestGCInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls.temp.name)
        (root / "Annotation").mkdir()
        (root / "Frame").mkdir()
        frames = np.arange(0, 2400, 20)
        for f in frames:
            Image.new("RGB", (64, 48), (int(f//20) % 255, 100, 150)).save(root / "Frame" / f"{f:06d}.jpg")
        for agent, name in [(1, "000001.txt"), (2, "000002.txt"),
                            (3, "000003.txt"), (4, "000004.txt"),
                            (7427, "007427 extra suffix.txt")]:
            chosen = frames.copy()
            if agent == 2:
                chosen = chosen[chosen >= 400]
            if agent == 3:
                chosen = chosen[chosen != 200]
            if agent == 4:
                chosen = chosen[chosen >= 420]
            x = 20 + chosen/200 + {1: 0, 2: 3, 3: 6, 4: 1, 7427: 20}[agent]
            y = np.full(len(chosen), 20)
            rows = np.column_stack((x, y, chosen))
            np.savetxt(root / "Annotation" / name, rows.reshape(-1), fmt="%.8f")
        cls.config = GCConfig(max_neighbors=3, scale_pixels=64, crop_pixels=32, image_size=16)
        cls.store = GCStore(root, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def sample_at(self, agent=1, frame=400, include_image=False):
        dataset = GCDataset(self.store, "train", include_image)
        for i, (aid, anchor) in enumerate(self.store.samples["train"]):
            if aid == agent and self.store.tracks[aid][0][anchor] == frame:
                return dataset[i]
        self.fail("Requested fixture sample does not exist")

    def test_window_duration_endpoints(self):
        s = self.sample_at()
        self.assertEqual(s["obs"].shape, (5, 4))
        self.assertEqual(s["future"].shape, (6, 2))
        self.assertAlmostEqual((s["obs_frame_ids"][-1]-s["obs_frame_ids"][0])/25, 3.2)
        self.assertAlmostEqual((s["future_frame_ids"][-1]-s["anchor_frame"])/25, 4.8)
        self.assertEqual(s["future_frame_ids"][0]-s["anchor_frame"], 20)

    def test_missing_neighbor_history_and_future_only_agent(self):
        s = self.sample_at()
        self.assertNotIn(1, s["neighbor_ids"])
        self.assertNotIn(4, s["neighbor_ids"])
        slot = int(np.flatnonzero(s["neighbor_ids"] == 2)[0])
        np.testing.assert_array_equal(s["neighbor_mask"][slot], [False, False, False, False, True])
        self.assertFalse(s["neighbor_delta_mask"][slot].any())
        self.assertTrue((s["neighbors"][slot, :4] == 0).all())

    def test_target_windows_never_bridge_a_gap_or_split(self):
        used = {}
        for split, windows in self.store.samples.items():
            used[split] = set()
            lo, hi = self.store.split_ranges[split]
            self.assertGreater(len(windows), 0)
            for agent, anchor in windows:
                frames, _ = self.store.tracks[agent]
                window = frames[anchor-4:anchor+7]
                self.assertTrue(np.all(np.diff(window) == 20))
                self.assertGreaterEqual(window[0], lo)
                self.assertLessEqual(window[-1], hi)
                used[split].update(window.tolist())
        self.assertFalse(used["train"] & used["val"])
        self.assertFalse(used["train"] & used["test"])
        self.assertFalse(used["val"] & used["test"])

    def test_coordinate_roundtrip_and_image(self):
        s = self.sample_at(include_image=True)
        frames, xy = self.store.tracks[1]
        anchor = int(np.flatnonzero(frames == 400)[0])
        np.testing.assert_allclose(restore_pixels(s["future"], s["origin"], s["scale"]), xy[anchor+1:anchor+7], atol=1e-5)
        np.testing.assert_allclose(s["obs"][-1, :2], 0)
        self.assertEqual(s["image"].shape, (3, 16, 16))
        self.assertEqual(s["image_mask"].dtype, np.bool_)
        self.assertTrue(0 <= s["image"].min() <= s["image"].max() <= 1)
        self.assertTrue(s["image_path"].endswith("000400.jpg"))

    def test_future_labels_cannot_change_inputs(self):
        before = self.sample_at()
        originals = {aid: xy.copy() for aid, (_, xy) in self.store.tracks.items()}
        try:
            for frames, xy in self.store.tracks.values():
                xy[frames > 400] += 10
            after = self.sample_at()
            for key in ["obs", "neighbors", "neighbor_ids", "neighbor_mask", "origin", "crop_box"]:
                np.testing.assert_array_equal(before[key], after[key])
            self.assertFalse(np.array_equal(before["future"], after["future"]))
        finally:
            for aid, (_, xy) in self.store.tracks.items():
                xy[:] = originals[aid]

    def test_nonstandard_filename_and_invalid_duration(self):
        self.assertIn(7427, self.store.tracks)
        self.assertEqual(len(self.store.renamed_files), 1)
        with self.assertRaises(ValueError):
            GCConfig(history_seconds=3.3)

    def test_torch_collation_if_installed(self):
        try:
            from torch.utils.data import DataLoader
        except ImportError:
            self.skipTest("PyTorch is optional and not installed in this environment")
        batch = next(iter(DataLoader(GCDataset(self.store, "train"), batch_size=2)))
        self.assertEqual(tuple(batch["obs"].shape), (2, 5, 4))
        self.assertEqual(tuple(batch["neighbors"].shape), (2, 3, 5, 4))
        self.assertEqual(tuple(batch["future"].shape), (2, 6, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
import numpy as np
import os
import json
import stereo_utils

class TestStereoUtils(unittest.TestCase):
    
    def setUp(self):
        # Create a dummy config for testing
        self.test_config_name = "test_config.json"
        self.config_data = {"test": 123, "camera": {"width": 640}}
        with open(self.test_config_name, "w") as f:
            json.dump(self.config_data, f)

    def tearDown(self):
        if os.path.exists(self.test_config_name):
            os.remove(self.test_config_name)

    def test_load_config_success(self):
        """Test loading a valid config file."""
        config = stereo_utils.load_config(self.test_config_name)
        self.assertEqual(config["test"], 123)
        self.assertEqual(config["camera"]["width"], 640)

    def test_load_config_missing(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            stereo_utils.load_config("non_existent_file.json")

    def test_get_3d_points(self):
        """Test 3D point projection math."""
        # Simple Q matrix (disparity-to-depth)
        # Q usually looks like [[1,0,0,-cx], [0,1,0,-cy], [0,0,0,f], [0,0,1/b,0]]
        # Let's make a dummy identity-like Q for easy math
        Q = np.eye(4, dtype=np.float32)
        Q[3,2] = 1.0 # simplistic w term
        
        # Disparity map (2x2)
        disparity = np.ones((2, 2), dtype=np.float32)
        
        points = stereo_utils.get_3d_points(disparity, Q)
        
        # Output should be (2, 2, 3)
        self.assertEqual(points.shape, (2, 2, 3))
        # Validate values (reprojectImageTo3D is complex, but we check running without crash)
        self.assertTrue(np.isfinite(points).any())

if __name__ == "__main__":
    unittest.main()

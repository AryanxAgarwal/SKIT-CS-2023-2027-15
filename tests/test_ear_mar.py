import unittest
import numpy as np

from landmarks import LandmarkExtractor


class TestEAR(unittest.TestCase):

    def test_ear_formula(self):
        """
        Test the mathematical EAR formula:

        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """

        points = np.array([
            [0.0, 0.0],    # p1
            [1.0, 2.0],    # p2
            [3.0, 2.0],    # p3
            [4.0, 0.0],    # p4
            [3.0, -2.0],   # p5
            [1.0, -2.0],   # p6
        ])

        indices = [0, 1, 2, 3, 4, 5]

        ear = LandmarkExtractor._eye_aspect_ratio(points, indices)

        # vertical distances = 4 + 4
        # horizontal distance = 4
        # EAR = 8 / (2 * 4) = 1
        expected = 1.0

        self.assertAlmostEqual(ear, expected, places=6)


    def test_ear_scale_invariance(self):
        """
        Scaling all landmark coordinates should not change EAR.
        """

        points = np.array([
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 2.0],
            [4.0, 0.0],
            [3.0, -2.0],
            [1.0, -2.0],
        ])

        scaled_points = points * 5

        indices = [0, 1, 2, 3, 4, 5]

        ear_original = LandmarkExtractor._eye_aspect_ratio(
            points, indices
        )

        ear_scaled = LandmarkExtractor._eye_aspect_ratio(
            scaled_points, indices
        )

        self.assertAlmostEqual(
            ear_original,
            ear_scaled,
            places=6
        )


    def test_ear_closed_eye_lower_than_open_eye(self):
        """
        A geometrically closed eye should produce a smaller EAR
        than an open eye.
        """

        open_eye = np.array([
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 2.0],
            [4.0, 0.0],
            [3.0, -2.0],
            [1.0, -2.0],
        ])

        closed_eye = np.array([
            [0.0, 0.0],
            [1.0, 0.2],
            [3.0, 0.2],
            [4.0, 0.0],
            [3.0, -0.2],
            [1.0, -0.2],
        ])

        indices = [0, 1, 2, 3, 4, 5]

        open_ear = LandmarkExtractor._eye_aspect_ratio(
            open_eye, indices
        )

        closed_ear = LandmarkExtractor._eye_aspect_ratio(
            closed_eye, indices
        )

        self.assertGreater(open_ear, closed_ear)


class TestMAR(unittest.TestCase):

    def test_mar_formula(self):
        """
        Test the mathematical MAR formula implemented in landmarks.py.
        """

        points = np.array([
            [0.0, 0.0],    # p1
            [1.0, 1.0],    # p2
            [2.0, 1.0],    # p3
            [3.0, 1.0],    # p4
            [4.0, 0.0],    # p5
            [3.0, -1.0],   # p6
            [2.0, -1.0],   # p7
            [1.0, -1.0],   # p8
        ])

        indices = list(range(8))

        mar = LandmarkExtractor._mouth_aspect_ratio(
            points, indices
        )

        # vertical = 2 + 2 + 2 = 6
        # horizontal = 3 * 4 = 12
        # MAR = 6 / 12 = 0.5
        expected = 0.5

        self.assertAlmostEqual(mar, expected, places=6)


    def test_mar_scale_invariance(self):
        """
        Scaling the mouth coordinates should not change MAR.
        """

        points = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 0.0],
            [3.0, -1.0],
            [2.0, -1.0],
            [1.0, -1.0],
        ])

        scaled_points = points * 10

        indices = list(range(8))

        mar_original = LandmarkExtractor._mouth_aspect_ratio(
            points, indices
        )

        mar_scaled = LandmarkExtractor._mouth_aspect_ratio(
            scaled_points, indices
        )

        self.assertAlmostEqual(
            mar_original,
            mar_scaled,
            places=6
        )


    def test_mar_open_mouth_greater_than_closed_mouth(self):
        """
        A geometrically open mouth should have a greater MAR
        than a closed mouth.
        """

        open_mouth = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 0.0],
            [3.0, -1.0],
            [2.0, -1.0],
            [1.0, -1.0],
        ])

        closed_mouth = np.array([
            [0.0, 0.0],
            [1.0, 0.1],
            [2.0, 0.1],
            [3.0, 0.1],
            [4.0, 0.0],
            [3.0, -0.1],
            [2.0, -0.1],
            [1.0, -0.1],
        ])

        indices = list(range(8))

        open_mar = LandmarkExtractor._mouth_aspect_ratio(
            open_mouth, indices
        )

        closed_mar = LandmarkExtractor._mouth_aspect_ratio(
            closed_mouth, indices
        )

        self.assertGreater(open_mar, closed_mar)


if __name__ == "__main__":
    unittest.main()
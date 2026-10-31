import unittest
import numpy as np
from metrology.postprocess import match_instances, validate_eye, pixel_distance


def detection(box, name='dog'):
    return {'bbox': box, 'class_name': name}


class PostprocessTests(unittest.TestCase):
    def test_one_to_one(self):
        a = [detection([0,0,10,10]), detection([8,0,18,10])]
        self.assertEqual(match_instances(a, [detection([0,0,10,10])]), {0: 0})

    def test_species_and_bear_alias(self):
        self.assertEqual(match_instances([detection([0,0,10,10], 'bear')],
                                         [detection([0,0,10,10], 'brown bear')]), {0: 0})
        self.assertEqual(match_instances([detection([0,0,10,10])],
                                         [detection([0,0,10,10], 'cat')]), {})

    def test_disjoint_rejected_even_without_class_gate(self):
        self.assertEqual(match_instances([detection([0,0,10,10])],
                         [detection([20,20,30,30], 'animal')], same_class=False), {})

    def test_eye_reasons(self):
        mask = np.ones((10,10), dtype=bool)
        self.assertEqual(validate_eye([3,4], .9, mask, [mask], 0)['status'], 'valid')
        self.assertEqual(validate_eye([3,4], .1, mask, [mask], 0)['status'], 'low_confidence')
        self.assertEqual(validate_eye([10,4], .9, mask, [mask], 0)['status'], 'outside_image')
        self.assertEqual(validate_eye([3,4], .9, mask, [mask,mask], 0)['status'], 'ambiguous_mask')
        mask[4,3] = False
        self.assertEqual(validate_eye([3,4], .9, mask, [mask], 0)['status'], 'outside_mask')
        self.assertEqual(validate_eye([float('nan'),4], .9, mask, [mask], 0)['status'], 'invalid_coordinates')

    def test_distance_requires_valid_points(self):
        self.assertEqual(pixel_distance([0,0], [3,4]), 5.0)
        self.assertIsNone(pixel_distance(None, [3,4]))

    def test_assignment_is_global_optimum(self):
        from metrology.postprocess import _assignment
        # Greedy would choose row0/col0 and lose row1's only strong match.
        self.assertEqual(_assignment(-np.array([[.9,.8,0,0],[.85,0,0,0]])), {1:0,0:1})

    def test_empty_detections(self):
        self.assertEqual(match_instances([], []), {})

    def test_missing_confidence_is_not_valid(self):
        mask = np.ones((5,5), dtype=bool)
        self.assertEqual(validate_eye([2,2], None, mask, [mask], 0)['status'], 'missing_confidence')

    def test_iou_threshold_is_applied_before_assignment(self):
        self.assertEqual(match_instances([detection([0,0,10,10])],
                                         [detection([9,0,19,10])], min_iou=.3), {})

"""Real pipeline with only expensive neural inference replaced."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
import torch
from metrology.pipeline import run


class PoseEyesTests(unittest.TestCase):
    def execute(self, right_conf=.8, include_pose=True):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        image=root/'input.png'
        cv2.imwrite(str(image),np.zeros((12,12,3),dtype=np.uint8))
        box=SimpleNamespace(cls=torch.tensor([0]),conf=torch.tensor([.9]),
                            xyxy=torch.tensor([[0.,0.,12.,12.]]))
        seg=SimpleNamespace(boxes=[box],names={0:'dog'},keypoints=None,
                            masks=SimpleNamespace(data=torch.ones((1,12,12))))
        pose=SimpleNamespace(boxes=[box] if include_pose else [],names={0:'dog'},
            keypoints=SimpleNamespace(xy=torch.tensor([[[3.,4.],[6.,8.]]+[[11.,11.]]*15]),
                                      conf=torch.tensor([[.9,right_conf]+[.99]*15])))
        def load(path):
            name=Path(path).name
            if name=='animal_seg.pt': prediction=seg
            elif name=='animal_pose_v2.pt': prediction=pose
            else: raise AssertionError('Old Eye model must not be loaded')
            return SimpleNamespace(predict=lambda *args,**kwargs:[prediction])
        with patch('metrology.pipeline.load_model',side_effect=load):
            result=run(image,root/'result',device='cpu')
        self.assertEqual(json.loads((root/'result/result.json').read_text()),result)
        return result

    def test_measure_uses_first_two_pose_points(self):
        r=self.execute()
        a=r['animals'][0]
        self.assertEqual(a['eye_points'],[[3.,4.],[6.,8.]])
        self.assertEqual(a['inter_eye_distance_px'],5.)
        self.assertEqual(a['eye_match']['source'],'animal_pose')
        self.assertNotIn('eyes',r['models'])

    def test_low_pose_confidence_does_not_fallback_to_old_eye_model(self):
        a=self.execute(right_conf=.1)['animals'][0]
        self.assertEqual(a['eyes']['right']['status'],'low_confidence')
        self.assertIsNone(a['inter_eye_distance_px'])

    def test_unmatched_pose_is_unmeasurable(self):
        a=self.execute(include_pose=False)['animals'][0]
        self.assertEqual(a['eyes']['left']['status'],'unmatched')
        self.assertIsNone(a['inter_eye_distance_px'])

import unittest
from unittest.mock import patch
from metrology.runtime import select_device


class RuntimeTests(unittest.TestCase):
    # Probe is the hardware boundary; selection and fallback remain real.
    def test_auto_prefers_cuda(self):
        with patch('metrology.runtime.probe_device', return_value=None):
            self.assertEqual(select_device()['device'], 'cuda')

    def test_auto_uses_mps_when_cuda_fails(self):
        def probe(device):
            if device == 'cuda':
                raise RuntimeError('unavailable')
        with patch('metrology.runtime.probe_device', side_effect=probe):
            result = select_device()
        self.assertEqual(result['device'], 'mps')
        self.assertIn('cuda', result['fallback_reasons'][0])

    def test_failed_mps_falls_back_to_cpu(self):
        def probe(device):
            if device != 'cpu':
                raise RuntimeError('unavailable')
        with patch('metrology.runtime.probe_device', side_effect=probe):
            self.assertEqual(select_device('mps')['device'], 'cpu')

    def test_explicit_cpu(self):
        with patch('metrology.runtime.probe_device', return_value=None):
            self.assertEqual(select_device('cpu')['device'], 'cpu')

    def test_strict_mps_does_not_silently_fallback(self):
        with patch('metrology.runtime.probe_device', side_effect=RuntimeError('unavailable')):
            with self.assertRaises(RuntimeError):
                select_device('mps', allow_cpu_fallback=False)

    def test_reject_invalid_device(self):
        with self.assertRaises(ValueError):
            select_device('mpz')

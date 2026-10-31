"""Device selection shared by native and container entry points."""
import torch


def probe_device(device: str) -> None:
    """Require availability and a completed, finite tensor operation."""
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    x = torch.ones((2, 2), device=device)
    result = x @ x
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError('Device probe produced non-finite values')


def select_device(requested: str = 'auto', allow_cpu_fallback: bool = True) -> dict:
    if requested not in {'auto', 'cuda', 'mps', 'cpu'}:
        raise ValueError(f'Unsupported device: {requested}')
    candidates = ['cuda', 'mps', 'cpu'] if requested == 'auto' else [requested]
    if requested in {'cuda', 'mps'} and allow_cpu_fallback:
        candidates.append('cpu')
    reasons = []
    for device in candidates:
        try:
            probe_device(device)
        except RuntimeError as exc:
            reasons.append(f'{device}: {exc}')
        else:
            return {'requested_device': requested, 'device': device,
                    'fallback_reasons': reasons}
    raise RuntimeError('No usable execution device: ' + '; '.join(reasons))

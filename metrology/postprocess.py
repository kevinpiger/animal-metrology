"""Geometry and conservative association; no inference or file side effects."""
import math
import numpy as np


def box_iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area = lambda x: max(0., x[2]-x[0]) * max(0., x[3]-x[1])
    union = area(a) + area(b) - intersection
    return intersection / union if union > 0 else 0.


def _assignment(cost):
    """Rectangular Hungarian minimization, rows <= columns; O(n^2 m)."""
    n, m = cost.shape
    u, v = np.zeros(n+1), np.zeros(m+1)
    p, way = np.zeros(m+1, dtype=int), np.zeros(m+1, dtype=int)
    for i in range(1, n+1):
        p[0] = i
        j0 = 0
        minimum, used = np.full(m+1, np.inf), np.zeros(m+1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, m+1):
                if not used[j]:
                    current = cost[i0-1,j-1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return {int(p[j]-1): j-1 for j in range(1,m+1) if p[j]}


def match_instances(animals, candidates, min_iou=.3, same_class=True):
    if not 0 < min_iou <= 1:
        raise ValueError('min_iou must be in (0, 1]')
    if not animals or not candidates:
        return {}
    aliases = {'black bear': 'bear', 'brown bear': 'bear', 'polar bear': 'bear', 'panda': 'bear'}
    canonical = lambda name: aliases.get(name, name)
    # Zero-cost dummy columns permit every animal to remain unmatched.
    scores = np.zeros((len(animals), len(candidates)+len(animals)))
    for i, animal in enumerate(animals):
        for j, candidate in enumerate(candidates):
            if same_class and canonical(animal['class_name']) != canonical(candidate['class_name']):
                continue
            overlap = box_iou(animal['bbox'], candidate['bbox'])
            if overlap >= min_iou:
                scores[i,j] = overlap
    return {i:j for i,j in _assignment(-scores).items()
            if j < len(candidates) and scores[i,j] > 0}


def validate_eye(point, confidence, mask, all_masks, animal_index, threshold=.5):
    status = 'valid'
    if point is None or len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
        return {'point': None, 'confidence': None, 'status': 'invalid_coordinates'}
    point = [float(v) for v in point]
    confidence = float(confidence) if confidence is not None else None
    if confidence is None or not math.isfinite(confidence):
        return {'point': point, 'confidence': None, 'status': 'missing_confidence'}
    x, y = point
    h, w = mask.shape
    if confidence < threshold:
        status = 'low_confidence'
    elif not (0 <= x < w and 0 <= y < h):
        status = 'outside_image'
    elif not mask[int(y),int(x)]:
        status = 'outside_mask'
    elif any(other[int(y),int(x)] for j, other in enumerate(all_masks) if j != animal_index):
        status = 'ambiguous_mask'
    return {'point': point, 'confidence': confidence, 'status': status}


def pixel_distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1]) if a is not None and b is not None else None

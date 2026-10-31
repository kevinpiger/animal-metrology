"""Segmentation and Animal Pose inference with per-image association and pixel measurements."""
from pathlib import Path
from itertools import combinations
import argparse
import json
import os
import cv2
from ultralytics import YOLO
from functools import lru_cache
from types import SimpleNamespace
import csv
from .runtime import select_device
from .postprocess import box_iou, match_instances, validate_eye, pixel_distance


def detections(prediction):
    rows = []
    if prediction.boxes is None:
        return rows
    for i, box in enumerate(prediction.boxes):
        cls = int(box.cls.item())
        row = {'bbox': box.xyxy[0].cpu().tolist(), 'class': cls,
               'class_name': prediction.names[cls], 'confidence': float(box.conf.item())}
        if prediction.keypoints is not None:
            row['keypoints'] = prediction.keypoints.xy[i].cpu().tolist()
            conf = prediction.keypoints.conf
            row['keypoint_confidence'] = conf[i].cpu().tolist() if conf is not None else None
        rows.append(row)
    return rows


def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError('Must be between 0 and 1')
    return number


@lru_cache(maxsize=2)
def load_model(path):
    return YOLO(path)


def run(image, out, *, device='auto', conf=.25, eye_conf=.5, match_iou=.3):
    root = Path(__file__).resolve().parents[1]
    a = SimpleNamespace(image=str(image), out=str(out), device=device, strict_device=False,
                        conf=conf, eye_conf=eye_conf, match_iou=match_iou,
                        pose=str(root/'models/animal_pose.pt'),
                        seg=str(root/'models/animal_seg.pt'))
    if a.match_iou == 0:
        raise ValueError('match_iou must be greater than zero')
    remote = bool(os.getenv('SEG_SERVICE_URL') and os.getenv('POSE_SERVICE_URL'))
    for weight in (() if remote else (a.pose, a.seg)):
        if not Path(weight).is_file():
            raise FileNotFoundError(f'Model weight not found: {weight}')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new output directory: {out}')
    img = cv2.imread(a.image)
    if img is None:
        raise FileNotFoundError(a.image)
    predictions = {}
    if remote:
        from .remote import predict_remote
        devices = {}
        for name in ('seg', 'pose'):
            predictions[name], devices[name] = predict_remote(os.environ[name.upper() + '_SERVICE_URL'], img, a.device, a.conf)
        runtime = {'requested_device': a.device, 'device': devices['seg']['device'], 'services': devices}
    else:
        runtime = select_device(a.device, allow_cpu_fallback=not a.strict_device)
        for name, weight in [('seg', a.seg), ('pose', a.pose)]:
            predictions[name] = load_model(weight).predict(
                img, device=runtime['device'], conf=a.conf, retina_masks=True, verbose=False)[0]
    seg = predictions['seg']
    animals = detections(seg)
    poses = detections(predictions['pose'])
    eyes = poses  # AP-10K indices 0/1 are anatomical left/right eyes.
    masks = []
    if animals and seg.masks is None:
        raise RuntimeError('Segmentation detections have no masks')
    if seg.masks is not None:
        for mask in seg.masks.data.cpu().numpy():
            masks.append(cv2.resize(mask, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_NEAREST) > .5)
    if len(masks) != len(animals):
        raise RuntimeError('Mask and detection counts differ')
    pose_matches = match_instances(animals, poses, a.match_iou)
    eye_matches = pose_matches  # Reuse the same species-aware, one-to-one instance association.
    result = {'schema_version': 3, 'eye_source': 'animal_pose', 'eye_keypoint_indices': {'left': 0, 'right': 1}, 'image': str(Path(a.image).resolve()),
              'image_size': {'width':img.shape[1], 'height':img.shape[0]},
              'runtime': runtime, 'models': {k: (os.environ[k.upper()+'_SERVICE_URL'] if remote else str(Path(getattr(a,k)).resolve())) for k in predictions},
              'thresholds': {'detection': a.conf, 'eye': a.eye_conf, 'match_iou': a.match_iou},
              'association_status': 'heuristic_not_accuracy_validated',
              'distance_unit': 'pixel', 'animals': [], 'right_eye_distances': [],
              'unmatched_pose_indices': sorted(set(range(len(poses)))-set(pose_matches.values())),
              'unmatched_eye_indices': sorted(set(range(len(eyes)))-set(eye_matches.values()))}
    vis = img.copy()
    out.mkdir(parents=True, exist_ok=True)

    def save_image(path, image):
        if not cv2.imwrite(str(path), image):
            raise OSError(f'Cannot write image: {path}')

    for i, animal in enumerate(animals):
        animal.update({'id': i, 'mask': f'animal_{i}_mask.png', 'pose': None,
                       'eyes': {}, 'eye_points': None, 'inter_eye_distance_px': None})
        if i in pose_matches:
            j = pose_matches[i]
            animal['pose'] = dict(poses[j], candidate_index=j,
                                  match_iou=box_iou(animal['bbox'],poses[j]['bbox']))
        eye_index = eye_matches.get(i)
        animal['eye_match'] = {'status': 'unmatched' if eye_index is None else 'matched',
                               'candidate_index': eye_index, 'source': 'animal_pose',
                               'match_iou': None if eye_index is None else box_iou(animal['bbox'], eyes[eye_index]['bbox'])}
        valid_points = []
        for k, name in enumerate(['left', 'right']):
            if eye_index is None:
                info = {'point': None, 'confidence': None, 'status': 'unmatched'}
            else:
                candidate = eyes[eye_index]
                points, confidence = candidate.get('keypoints', []), candidate.get('keypoint_confidence')
                info = validate_eye(points[k] if len(points)>k else None,
                                    confidence[k] if confidence is not None and len(confidence)>k else None,
                                    masks[i], masks, i, a.eye_conf)
            animal['eyes'][name] = info
            point = info['point'] if info['status'] == 'valid' else None
            valid_points.append(point)
            if point is not None:
                xy = tuple(int(v) for v in point)
                cv2.circle(vis, xy, 4, (79,123,30), -1)
                cv2.rectangle(vis, (xy[0]-8,xy[1]-8), (xy[0]+8,xy[1]+8), (79,123,30), 1)
                cv2.putText(vis, name[0].upper(), xy, cv2.FONT_HERSHEY_SIMPLEX, .5, (79,123,30), 1)
        animal['inter_eye_distance_px'] = pixel_distance(*valid_points)
        if all(point is not None for point in valid_points):
            animal['eye_points'] = valid_points
        color = (40+35*(i%5),200-25*(i%5),50+30*(i%5))
        contours,_ = cv2.findContours(masks[i].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis,contours,-1,color,2)
        x,y = (int(v) for v in animal['bbox'][:2])
        cv2.putText(vis,f"#{i} {animal['class_name']}",(x,max(15,y)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
        save_image(out/animal['mask'], masks[i].astype('uint8')*255)
        result['animals'].append(animal)
    for first, second in combinations(animals,2):
        points = [animal['eyes']['right']['point'] if animal['eyes']['right']['status']=='valid' else None
                  for animal in (first,second)]
        distance = pixel_distance(*points)
        result['right_eye_distances'].append({'animal_ids':[first['id'],second['id']],
                                              'distance_px':distance,
                                              'status':'valid' if distance is not None else 'missing_valid_right_eye'})
    save_image(out/'overlay.jpg',vis)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(f'Saved {len(animals)} animals to {out}; pose matches={len(pose_matches)}, eye matches={len(eye_matches)}')

    csv_rows = []
    for animal in animals:
        distance = animal['inter_eye_distance_px']
        reason = '; '.join(f"{side}:{animal['eyes'][side]['status']}" for side in ('left','right')
                           if animal['eyes'][side]['status'] != 'valid')
        csv_rows.append(['inter_eye', animal['id'], '', distance,
                         'valid' if distance is not None else 'unavailable', reason])
    for pair in result['right_eye_distances']:
        csv_rows.append(['right_eye_pair', *pair['animal_ids'], pair['distance_px'],
                         pair['status'], '' if pair['status']=='valid' else pair['status']])
    with (out/'measurements.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['measurement_type','animal_id_a','animal_id_b','distance_px','status','reason'])
        writer.writerows(csv_rows)
    return result

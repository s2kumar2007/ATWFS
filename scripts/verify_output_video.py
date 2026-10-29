"""Objective video audit using raw-frame reference detections and saved run telemetry.

Reference detections are comparisons, not ground truth. Missing sidecars are errors.
No inference from overlay colors or historical measurements is substituted for masks.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import cv2
import numpy as np


def box_iou(a, b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]))
    y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=x*y
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return inter/union if union else 0.0


def match_boxes(predicted, reference, threshold=0.3):
    candidates=sorted(((box_iou(a,b),i,j) for i,a in enumerate(predicted) for j,b in enumerate(reference)),reverse=True)
    used_p,used_r=set(),set()
    for score,i,j in candidates:
        if score >= threshold and i not in used_p and j not in used_r:
            used_p.add(i); used_r.add(j)
    return len(used_p), [i for i in range(len(predicted)) if i not in used_p]


def mask_statistics(mask, hood_top=0.9, horizon=0.25):
    h,w=mask.shape
    end=max(1,min(h,int(hood_top*h)))
    start=max(0,end-max(1,int(0.05*h)))
    strip=mask[start:end,int(.3*w):max(int(.3*w)+1,int(.7*w))]>0
    return {'road_strip_coverage':float(strip.mean()),
            'mask_above_horizon_fraction':float(np.count_nonzero(mask[:int(horizon*h)]))/max(1,np.count_nonzero(mask))}


def verify(run_dir, raw_source, reference_weights, out_dir, reference_stride=1):
    run_dir,out_dir=Path(run_dir),Path(out_dir)
    rows=[json.loads(line) for line in (run_dir/'telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
    if not rows: raise ValueError('No telemetry frames available')
    if not Path(reference_weights).is_file(): raise FileNotFoundError('Independent reference detector weights missing')
    from ultralytics import YOLO
    reference=YOLO(str(reference_weights))
    cap=cv2.VideoCapture(str(raw_source))
    if not cap.isOpened(): raise ValueError('Cannot open raw video')
    annotated=cv2.VideoCapture(str(run_dir/'annotated.mp4'))
    out_dir.mkdir(parents=True,exist_ok=True)
    angle=np.array([r['angle'] for r in rows])
    latency=np.array([r['processing_ms'] for r in rows])
    tp=npred=nref=empty=label_overlap=offscreen=fb_mismatch=unexplained_stop=sign_ok=sign_n=ghosts=0
    ghost_tracks=[]
    mask_stats=[]; actual_horizon_stats=[]; flicker=[]; previous=None; thumbs=[]
    contacts=set(np.linspace(0,len(rows)-1,12).astype(int).tolist())
    raw_index=-1
    for r in rows:
        idx=r['frame']
        while raw_index < idx:
            ret,frame=cap.read(); raw_index+=1
            if not ret: raise ValueError('Raw video ended before telemetry')
        h,w=frame.shape[:2]
        if idx % reference_stride == 0:
            result=reference.predict(frame,conf=.15,imgsz=640,device='cpu',classes=[0,1,2,3,5,7,16,17,18,19],verbose=False)[0]
            refs=result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
            matched,unmatched=match_boxes(r['boxes'],refs)
            tp+=matched;npred+=len(r['boxes']);nref+=len(refs)
            empty+=int(not r['boxes'] and bool(refs))
            new_tracks=[]
            remaining=list(ghost_tracks)
            for i in unmatched:
                box=r['boxes'][i]
                best=max(range(len(remaining)),key=lambda j:box_iou(box,remaining[j][0]),default=None)
                streak=1
                if best is not None and box_iou(box,remaining[best][0])>=.3:
                    old=remaining.pop(best);streak=old[1]+1
                if streak==5: ghosts+=1
                new_tracks.append((box,streak))
            ghost_tracks=new_tracks
        rects=r['labels']
        offscreen+=sum(int(a[0]<0 or a[1]<0 or a[2]>w or a[3]>h) for a in rects)
        label_overlap+=sum(int(box_iou(a,b)>0) for i,a in enumerate(rects) for b in rects[i+1:])
        fb_mismatch+=int((r['active']=='FALLBACK') != (r['trust']<.5))
        unexplained_stop+=int(r['speed']==0 and not r['obstacle'])
        path=r['path']
        if len(path)>1:
            bend=path[-1][0]-path[0][0]
            if abs(bend)>1 and abs(r['angle'])>.5:
                sign_n+=1;sign_ok+=int(bend*r['angle']>0)
        mask_path=run_dir/'masks'/f'{idx:06d}.png'
        if mask_path.is_file():
            mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)>0
            mask_stats.append(mask_statistics(mask,hood_top=r.get('hood_top',.9)))
            if r.get('horizon_fraction') is not None:
                actual_horizon_stats.append(mask_statistics(mask,hood_top=r.get('hood_top',.9),horizon=r['horizon_fraction']))
            if previous is not None and previous.shape==mask.shape:
                union=np.logical_or(previous,mask).sum()
                flicker.append(float(np.logical_and(previous,mask).sum()/union) if union else 1.)
            previous=mask
        if idx in contacts:
            annotated.set(cv2.CAP_PROP_POS_FRAMES,idx)
            ok,ann=annotated.read()
            if ok: thumbs.append(cv2.resize(ann,(424,239)))
    cap.release();annotated.release()
    result={
        'frames':len(rows),'perception':'real student ONNX and pipeline detector; independent YOLO reference is not ground truth',
        'reference_detector':Path(reference_weights).name,'reference_confidence':.15,'reference_stride':reference_stride,
        'precision_iou_0_3':tp/npred if npred else None,'recall_iou_0_3':tp/nref if nref else None,
        'matched_boxes':tp,'pipeline_boxes_compared':npred,'reference_boxes':nref,
        'ghost_tracks_5_consecutive_frames':ghosts if reference_stride==1 else 'NOT RUN: reference sampled; cannot establish consecutive raw frames',
        'zero_pipeline_boxes_with_reference':empty,'offscreen_labels':offscreen,'overlapping_label_pairs':label_overlap,
        'fallback_threshold_mismatches':fb_mismatch,'speed_zero_without_obstacle_frames':unexplained_stop,
        'steering_ge15_pct':float(np.mean(abs(angle)>=15)*100),'steering_ge20_pct':float(np.mean(abs(angle)>=20)*100),
        'steering_ge29_5_pct':float(np.mean(abs(angle)>=29.5)*100),'max_steering_change_deg':float(np.max(abs(np.diff(angle)))) if len(angle)>1 else 0.,
        'drawn_path_sign_agreement':sign_ok/sign_n if sign_n else None,'drawn_path_sign_frames':sign_n,
        'pipeline_mean_ms':float(latency.mean()),'pipeline_p95_ms':float(np.percentile(latency,95)),
        'pipeline_frames_over_150ms':int(np.sum(latency>150)),
        'fps_timeline':[{'frame':r['frame'],'fps':r['fps']} for r in rows],
        'mode_switches':sum(a['mode']!=b['mode'] for a,b in zip(rows,rows[1:])),
        'estimator_switches':sum(a['active']!=b['active'] for a,b in zip(rows,rows[1:])),
        'mask_sampling_stride':10,'mask_sample_count':len(mask_stats),
        'road_strip_definition':'central 40 percent, 5 percent height strip above detected hood apex (or y=0.9 if no hood); estimated horizon when available, nominal y=0.25 reported separately; not GT road accuracy',
        'road_strip_coverage_mean':float(np.mean([m['road_strip_coverage'] for m in mask_stats])) if mask_stats else None,
        'mask_above_estimated_horizon_mean':float(np.mean([m['mask_above_horizon_fraction'] for m in actual_horizon_stats])) if actual_horizon_stats else None,
        'estimated_horizon_mask_samples':len(actual_horizon_stats),
        'mask_above_nominal_horizon_mean':float(np.mean([m['mask_above_horizon_fraction'] for m in mask_stats])) if mask_stats else None,
        'sampled_mask_iou_mean':float(np.mean(flicker)) if flicker else None,'sampled_mask_iou_min':float(min(flicker)) if flicker else None,
        'consecutive_mask_iou_mean':float(np.mean([r['consecutive_mask_iou'] for r in rows if r.get('consecutive_mask_iou') is not None])) if any(r.get('consecutive_mask_iou') is not None for r in rows) else None,
        'consecutive_mask_iou_min':min((r['consecutive_mask_iou'] for r in rows if r.get('consecutive_mask_iou') is not None),default=None),
    }
    if len(thumbs)==12:
        sheet=np.vstack([np.hstack(thumbs[i:i+4]) for i in range(0,12,4)])
        cv2.imwrite(str(out_dir/'contact_sheet.jpg'),sheet)
    (out_dir/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='fps_timeline'},indent=2))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',required=True)
    ap.add_argument('--source',default=os.environ.get('ATWFS_TEST_VIDEO'))
    ap.add_argument('--reference-weights',default='yolov8s.pt')
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--reference-stride',type=int,default=1)
    a=ap.parse_args()
    if not a.source: ap.error('Set ATWFS_TEST_VIDEO or --source')
    if a.reference_stride<1: ap.error('--reference-stride must be positive')
    verify(a.run_dir,a.source,a.reference_weights,a.out_dir,a.reference_stride)


if __name__=='__main__':
    try: main()
    except (ValueError,OSError,RuntimeError,ImportError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        sys.exit(2)

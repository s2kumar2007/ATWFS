"""Objective verifier metric regressions; no detector downloads."""
import numpy as np
from scripts.verify_output_video import box_iou, match_boxes, mask_statistics


def test_independent_matching_is_one_to_one():
    box=[0,0,10,10]
    assert box_iou(box,box)==1
    assert box_iou(box,[20,20,30,30])==0
    matched,unmatched=match_boxes([box,box],[box])
    assert matched==1 and len(unmatched)==1


def test_mask_metrics_use_mask_pixels():
    mask=np.zeros((100,100),np.uint8)
    mask[85:90,30:70]=1
    stats=mask_statistics(mask)
    assert stats['road_strip_coverage']==1
    assert stats['mask_above_horizon_fraction']==0


def test_numpy_detector_telemetry_serialization():
    import json
    from inference.live import _numpy_json
    encoded=json.dumps({'boxes':[[np.int64(1),np.int32(2)]],'speed':np.float32(.5)},default=_numpy_json)
    assert json.loads(encoded)=={'boxes':[[1,2]],'speed':.5}


def test_label_placement_stays_inside_image_and_avoids_overlap():
    from inference.live import place_label_rect,is_overlap
    occupied=[[0,0,380,260]]
    for x,y in [(840,2),(-20,-10),(800,470),(400,200)]:
        rect=place_label_rect(x,y,120,15,848,478,occupied)
        assert rect is not None
        assert 0<=rect[0]<rect[2]<=848 and 0<=rect[1]<rect[3]<=478
        assert not any(is_overlap(rect,old) for old in occupied)
        occupied.append(rect)


def test_estimated_horizon_metric_is_not_fixed_nominal():
    mask=np.zeros((100,100),np.uint8);mask[30:40,:]=1
    assert mask_statistics(mask,horizon=.25)['mask_above_horizon_fraction']==0
    assert mask_statistics(mask,horizon=.5)['mask_above_horizon_fraction']==1


def test_run_summary_counts_rendered_mode_transitions(tmp_path,monkeypatch):
    import csv
    import torch
    from data.synthetic import write_synthetic_video
    from inference.live import LivePipeline,run_video
    from inference.detector import NullDetector
    from models.unet_mobilenet import build_model
    from utils.config import smoke_config
    cfg=smoke_config()
    source=write_synthetic_video(str(tmp_path/'input.mp4'),4,(160,120))
    original=LivePipeline.process_frame
    def alternating(self,frame,frame_idx=0,**kwargs):
        result=original(self,frame,frame_idx=frame_idx,**kwargs)
        result.active='DL' if frame_idx%2==0 else 'FALLBACK'
        return result
    monkeypatch.setattr(LivePipeline,'process_frame',alternating)
    features=tmp_path/'features.csv'
    result=run_video(cfg,source,str(tmp_path/'out.mp4'),model=build_model(cfg['model']).to(torch.device('cpu')),detector=NullDetector(),max_frames=4,log_features=str(features),hood_polygon='none',profile='cpu')
    with features.open() as stream:
        modes=[row['active'] for row in csv.DictReader(stream)]
    assert modes==['DL','FALLBACK','DL','FALLBACK']
    assert result['estimator_switches']==3

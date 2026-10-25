"""Seeded negative-control faults outside the ATWFS decision/supervisor algorithms."""
import math
import random
from inference.detector import ObstacleDetector

class MissedDetections(ObstacleDetector):
    """Independent Bernoulli omissions on detector outputs BEFORE LivePipeline uses them."""
    def __init__(self,backend,rate,seed):
        if not math.isfinite(rate) or not 0<=rate<=1:raise ValueError('Missed-detection rate must be in [0,1].')
        self.backend=backend;self.rate=rate;self.rng=random.Random(seed)
        self.seen=0;self.missed=0
    def detect(self,bgr,frame_idx=0):
        detections=self.backend.detect(bgr,frame_idx);kept=[]
        for d in detections:
            self.seen+=1
            if self.rate>0 and self.rng.random()<self.rate:self.missed+=1
            else:kept.append(d)
        return kept
    def stop(self):self.backend.stop()
    def stats(self):
        return dict(nominal_miss_rate=self.rate,detector_boxes_seen=self.seen,detector_boxes_missed=self.missed,
                    observed_miss_rate=self.missed/self.seen if self.seen else None)

def in_dropout(t,trigger_time,duration):
    """Blackout begins at scenario trigger; plant and metric sensors keep running."""
    return trigger_time is not None and duration>0 and trigger_time-1e-9<=t<trigger_time+duration-1e-9

"""Small measured simulation metrics helpers."""
import math
import numpy as np
def iou(mask,ground_truth):
 a=np.asarray(mask)>0;b=np.asarray(ground_truth)>0;u=np.logical_or(a,b).sum();return float(np.logical_and(a,b).sum()/u) if u else 1.
def stopping_distance(v,t_react=0.5,decel=4.):return max(0.,v)*t_react+max(0.,v)**2/(2*max(.01,decel))
def ttc(distance,closing_speed):return math.inf if closing_speed<=0 else max(0.,distance/closing_speed)

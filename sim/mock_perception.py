"""Pixel-only synthetic front ends for portable mocks; downstream ATWFS code stays real.
These are test doubles, not trained-model accuracy or CARLA domain-gap evidence.
"""
import cv2
import numpy as np
import torch
from inference.detector import Detection, ObstacleDetector

class SyntheticSegmentation(torch.nn.Module):
    def forward(self, x):
        mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1)
        std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
        rgb=(x*std+mean)*255
        spread=rgb.amax(1,keepdim=True)-rgb.amin(1,keepdim=True)
        road=(spread<20)&(rgb.mean(1,keepdim=True)>65)
        score=torch.where(road,5.,-5.)
        logits=torch.cat([-score,score],1)
        return {'seg_logits':logits,'alpha':torch.ones_like(logits)*30.}

class PixelDetector(ObstacleDetector):
    def __init__(self): self.prev={}
    def detect(self,bgr,frame_idx=0):
        found=[]
        for kind,color,cls in [('person',(30,30,220),0),('vehicle',(190,70,20),2)]:
            mask=np.all(np.abs(bgr.astype(np.int16)-np.array(color))<15,axis=2).astype(np.uint8)
            count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask)
            for j,(x,y,w,h,area) in enumerate(stats[1:]):
                if area<8:continue
                key=(kind,j);old=self.prev.get(key,h);self.prev[key]=h
                found.append(Detection(int(x),int(y),int(x+w),int(y+h),cls,kind,.99,kind=kind,
                                       approach_rate=float(h/max(1,old)),tracker_id=cls*100+j))
        return found

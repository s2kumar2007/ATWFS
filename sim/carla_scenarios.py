"""Seeded trigger definitions shared by real and mock scenario runners."""
import random
SCENARIOS=('lane_follow','pedestrian_dash','lead_brake','blocked_lane','cut_in')
def make_scenarios(seed=0,runs=1):
 r=random.Random(seed);return {n:[dict(start=0.,trigger_distance=round(r.uniform(8,20),2),speed=round(r.uniform(1,2.5),2),seed=seed+i) for i in range(runs)] for n in SCENARIOS}
def triggered(distance,trigger_distance):return distance<=trigger_distance

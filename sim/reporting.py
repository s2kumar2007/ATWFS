"""Full scalar metric aggregation; p95 is pooled from raw frame latencies when available."""
from __future__ import annotations
import ast
import csv
import json
import math
from pathlib import Path
import numpy as np

SUM_KEYS={'frames','processed_frames','commands','collisions','target_collisions','sensor_dropout_ticks','detector_boxes_seen','detector_boxes_missed','false_stop_ticks','false_stop_events','false_stop_eligible_ticks','supervisor_override_ticks','iou_samples'}
SKIP={'scenario','run','seed','parameters','latency_samples_ms','supervisor_override_reasons'}

def full_summary(records,expected):
    rows=[]
    for name in dict.fromkeys(r['scenario'] for r in records):
        group=[r for r in records if r['scenario']==name]
        collisions=sum(r['collisions'] for r in group)
        row=dict(scenario=name,runs=len(group),collisions=collisions,status='PASS' if len(group)==expected and collisions==0 else 'FAIL')
        keys=list(dict.fromkeys(k for r in group for k in r if k not in SKIP))
        for key in keys:
            if key=='collisions':continue
            values=[r[key] for r in group if r.get(key) is not None]
            if key in SUM_KEYS:row[key]=sum(values) if values else None
            elif key.startswith('min_distance') or key in ('min_ttc_s','min_target_ttc_s'):row[key]=min(values) if values else None
            elif values and isinstance(values[0],bool):row[key+'_runs']=sum(values)
            elif values and isinstance(values[0],str):row[key]='; '.join(dict.fromkeys(values))
            else:row[key]=sum(values)/len(values) if values else None
        samples=[v for r in group for v in r.get('latency_samples_ms',[])]
        if samples:
            row['latency_mean_ms']=float(np.mean(samples));row['latency_ms']=row['latency_mean_ms']
            row['latency_p95_ms']=float(np.percentile(samples,95));row['pipeline_fps']=1000/row['latency_mean_ms'] if row['latency_mean_ms'] else None
            row['latency_percentile_source']='pooled raw inference-frame latencies'
        else:row['latency_percentile_source']='raw unavailable; aggregate of run statistics'
        eligible=sum(r.get('false_stop_eligible_ticks',0) for r in group)
        ticks=sum(r.get('false_stop_ticks',round((r.get('false_stop_rate') or 0)*r.get('false_stop_eligible_ticks',0))) for r in group)
        row['false_stop_ticks']=ticks;row['false_stop_rate']=ticks/eligible if eligible else None
        boxes=sum(r.get('detector_boxes_seen',0) for r in group)
        if boxes:row['observed_miss_rate']=sum(r.get('detector_boxes_missed',0) for r in group)/boxes
        row['stopping_distance_samples']=sum(r.get('stopping_distance_m') is not None for r in group)
        row['reaction_time_samples']=sum(r.get('reaction_time_s') is not None for r in group)
        reasons={}
        for r in group:
            for reason,n in r.get('supervisor_override_reasons',{}).items():reasons[reason]=reasons.get(reason,0)+n
        row['supervisor_override_reasons']=json.dumps(reasons,sort_keys=True)
        rows.append(row)
    return rows

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def historical_records(folder):
    folder=Path(folder);data=json.loads((folder/'metrics.json').read_text());records=data['runs'];groups=[]
    for line in (folder/'run_log.txt').read_text(encoding='utf-8').splitlines():
        marker='INFO sim.carla_bridge tick '
        if marker not in line:continue
        sample=ast.literal_eval(line.split(marker,1)[1])
        if sample['tick']==0:groups.append([])
        if sample.get('latency_ms') is not None:groups[-1].append(sample['latency_ms'])
    if len(groups)!=len(records):raise ValueError('Historical latency log/run count mismatch; do not invent pooled p95.')
    for r,samples in zip(records,groups):
        if len(samples)!=r['frames']:raise ValueError('Historical raw latency count mismatch.')
        if abs(float(np.mean(samples))-r['latency_mean_ms'])>1e-6:raise ValueError('Historical latency samples do not match stored means.')
        r['latency_samples_ms']=samples
    return data,records

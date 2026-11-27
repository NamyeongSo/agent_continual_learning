# SPDX-License-Identifier: Apache-2.0
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile

p=Path(__file__).resolve().parent
def rows(f):
    return list(csv.DictReader((p/f).open()))
rank=rows('task_quartiles.csv')
blocks=defaultdict(Counter)
initial = {}
for r in rank:
    key = r['model'],r['mode'],r['initial_pct']
    blocks[key][r['quartile']]+=1
    initial[r['model'],r['mode'],r['task']]=key
g=defaultdict(list)
for r in rows('task_seed_gains.csv'):
    key=initial[r['model'],r['mode'],r['task']]
    block=blocks[key]
    for q,n in block.items():
        k=tuple(r[x] for x in ['model','mode','arm','variant','strategy'])+(q,)
        g[k].append((n/sum(block.values()),float(r['independent_gain_pp']),float(r['accumulated_gain_pp'])))
stats=rows('quartile_gains.csv')
results=[]
for row in stats:
    k=tuple(row[x] for x in ['model','mode','arm','variant','strategy','quartile'])
    group=g[k]
    weight=sum(x[0] for x in group)
    assert abs(weight-int(row['tasks'])*int(row['seeds']))<1e-9
    r={k: v for k,v in row.items()}
    for i,name in ((1,'independent'),(2,'accumulated')):
        value=sum(x[0]*x[i] for x in group)/weight
        r[f'{name}_tie_weighted_gain_pp']=value
        r[f'{name}_tie_weighting_shift_pp']=value-float(row[f'{name}_gain_pp'])
    results.append(r)
with (p/'tie_weighting_sensitivity.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
for m in ('qwen38','qwen36'):
    for q in ('Q1','Q2','Q3','Q4'):
        for variant,labels in [('ADD',('best','worst')),('advantage',('best_advantage','worst_advantage'))]:
            d={k:[r for r in stats if r['model']==m and r['quartile']==q and r['variant']==variant
                  and r['strategy']==k] for k in labels}
            if q=='Q1':
                print(m,q,variant,[(k,round(mean(float(r['independent_gain_pp']) for r in rs),2),
                                      round(mean(float(r['accumulated_gain_pp']) for r in rs),2)) for k,rs in d.items()])
    shifts=[abs(r['accumulated_tie_weighting_shift_pp']) for r in results if r['model']==m and r['variant']!='AUD']
    print(m,'largest tie shift:',round(max(shifts),2))
archive=p/'ace_initial_pct_quartiles_20261120.zip'
with ZipFile(archive,'w',compression=ZIP_DEFLATED) as z:
    for f in sorted(p.iterdir()):
        if f.suffix in ('.csv','.json','.py','.png','.md') and f.name!='notion_receipt.json':
            z.write(f,f.name)
print(archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest())

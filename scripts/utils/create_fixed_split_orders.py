#!/usr/bin/env python3
"""Export per-seed permutations of frozen partitions without selecting tasks."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def sha(value):
    return hashlib.sha256(value).hexdigest()


def export(manifest_path, seeds):
    raw=manifest_path.read_bytes()
    manifest=json.loads(raw)
    result={'schema_version':2,'split_version':'fixed_balanced_v1_ordered',
        'split_seed':manifest['split_seed'],'order_seeds':seeds,
        'seed_scope':'within_split_order_only',
        'canonical_manifest':str(manifest_path.resolve()),'canonical_manifest_sha256':sha(raw),
        'generator_sha256':sha(Path(__file__).read_bytes()),
        'policy':'Task membership is frozen. Each order seed only permutes train/val/test independently within each benchmark.',
        'runtime_instruction':'Use benchmark_seed entries; experiment seed denotes data-order seed. Optimizer/model settings are separate.'}
    validation={'passed':True,'canonical_manifest_sha256':sha(raw),'order_seeds':seeds,'benchmarks':{}}
    for benchmark,section in manifest['benchmarks'].items():
        hashes={}
        for seed in seeds:
            view={'benchmark_kwargs':copy.deepcopy(section['benchmark_kwargs']),
                'split_counts':copy.deepcopy(section['split_counts'])}
            for split in ['train','val','test']:
                ids=copy.deepcopy(section[split])
                derived=int(sha(f'order-v1:{seed}:{benchmark}:{split}'.encode())[:16],16)
                random.Random(derived).shuffle(ids)
                assert len(ids)==len(set(ids)) and set(ids)==set(section[split])
                view[split]=ids
            result[f'{benchmark}_{seed}']=view
            hashes[str(seed)]={s:sha(canonical(view[s]).encode()) for s in ['train','val','test']}
        for split in ['train','val','test']:
            assert len({hashes[str(seed)][split] for seed in seeds})==len(seeds)
        validation['benchmarks'][benchmark]={'membership_identical_across_seeds':True,
            'ordered_lists_differ_across_seeds':True,'ordered_sha256':hashes,
            'membership_sha256':{s:sha(canonical(sorted(section[s])).encode()) for s in ['train','val','test']}}
    return result,validation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if len(args.seeds)!=len(set(args.seeds)) or len(args.seeds)<2:
        raise ValueError('Provide at least two distinct order seeds')
    views,validation=export(args.manifest,args.seeds)
    data=(json.dumps(views,ensure_ascii=False,indent=2)+'\n').encode()
    validation['ordered_manifest_sha256']=sha(data)
    contents={args.output:data,args.output.with_suffix('.validation.json'):(json.dumps(validation,ensure_ascii=False,indent=2)+'\n').encode()}
    for path,content in contents.items():
        if path.exists() and path.read_bytes()!=content:
            raise ValueError(f'Refusing to overwrite different frozen orders: {path}')
    for path,content in contents.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(content)
    print(json.dumps({'output':str(args.output),'passed':True,'membership_changed':False,'order_seeds':args.seeds},ensure_ascii=False))


if __name__=='__main__':
    main()

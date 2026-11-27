#!/usr/bin/env python3
"""Freeze common scenario/domain/category-balanced task splits for all run seeds.

Reads only benchmark metadata; never runs environments or uses model outcomes.
Requires numpy/scipy for Terminal-Bench's joint category/difficulty allocation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import tomllib

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

SPLITS = ('train', 'val', 'test')
PRIMARY = ('GorillaFileSystem', 'VehicleControlAPI', 'TradingBot', 'TravelAPI')
DOMAIN_NAMES = dict(zip(PRIMARY, ('파일시스템', '차량제어', '주식거래', '여행예약')))
BFCL_QUOTAS = {'train': (13, 13, 12, 12), 'val': (12, 13, 13, 12), 'test': (12, 12, 13, 13)}
TB_SIZES = (30, 30, 29)
TB_DIFFICULTY = ('easy', 'medium', 'hard')
TB_DIFFICULTY_TARGETS = ((1, 19, 10), (2, 18, 10), (1, 18, 10))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def rng(seed, namespace):
    return random.Random(int(digest(f'{seed}:{namespace}'.encode())[:16], 16))


def immutable_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Refusing to overwrite different frozen data: {path}')
        return
    path.write_bytes(content)


def load_jsonl_or_array(path):
    text = path.read_text()
    return json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]


def ordered(ids, seed, namespace):
    ids = sorted(ids)
    rng(seed, namespace).shuffle(ids)
    return ids


def bfcl_split(path, seed):
    entries = load_jsonl_or_array(path)
    groups = defaultdict(list)
    metadata = {}
    for row in entries:
        task = row['id']
        primary = set(row['involved_classes']) & set(PRIMARY)
        if len(primary) != 1 or task in metadata:
            raise ValueError(f'BFCL domain or ID ambiguity: {task}')
        domain = primary.pop()
        groups[domain].append(task)
        metadata[task] = {'primary_domain': domain, 'involved_classes': row['involved_classes'],
                          'user_turns': len(row['question'])}
    if {d: len(groups[d]) for d in PRIMARY} != {d: 50 for d in PRIMARY}:
        raise ValueError('BFCL source pool differs from expected four domains ×50')
    result = {s: [] for s in SPLITS}
    for di, domain in enumerate(PRIMARY):
        ids = ordered(groups[domain], seed, f'bfcl-domain:{domain}')
        offset = 0
        for split in SPLITS:
            count = BFCL_QUOTAS[split][di]
            result[split].extend(ids[offset:offset+count])
            offset += count
    for split in SPLITS:
        result[split] = ordered(result[split], seed, f'bfcl-order:{split}')
    return result, metadata, {'file': str(path), 'sha256': digest(path.read_bytes()), 'pool_size': len(entries)}


def appworld_split(root, seed):
    dataset = root/'datasets/test_challenge.txt'
    ids = [line.strip().split(':')[0] for line in dataset.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate AppWorld source task ID')
    metadata, scenarios, file_hashes = {}, defaultdict(list), []
    for task in sorted(ids):
        scenario, variant = task.rsplit('_', 1)
        path = root/'tasks'/task/'ground_truth/metadata.json'
        meta = json.loads(path.read_text())
        difficulty = int(meta['difficulty'])
        if difficulty not in (1, 2, 3):
            raise ValueError(f'Unexpected AppWorld difficulty: {task}')
        metadata[task] = {'scenario_id': scenario, 'variant': int(variant), 'difficulty': difficulty,
            'num_apps': meta['num_apps'], 'num_apis': meta['num_apis'],
            'num_solution_code_lines': meta['num_solution_code_lines']}
        scenarios[scenario].append(task)
        file_hashes.append((f'tasks/{task}/ground_truth/metadata.json', digest(path.read_bytes())))
    by_difficulty = defaultdict(list)
    for scenario, tasks in sorted(scenarios.items()):
        if {metadata[t]['variant'] for t in tasks} != {1, 2, 3} or len(tasks) != 3:
            raise ValueError(f'AppWorld scenario must have three distinct variants: {scenario}')
        values = {metadata[t]['difficulty'] for t in tasks}
        if len(values) != 1:
            raise ValueError(f'AppWorld scenario difficulty varies: {scenario}')
        by_difficulty[values.pop()].append(scenario)
    if len(scenarios) != 139:
        raise ValueError('AppWorld source scenario pool differs from expected139')
    # Largest-remainder allocation of50 scenarios to source difficulty ratios.
    ideal = {d: 50*len(by_difficulty[d])/len(scenarios) for d in (1, 2, 3)}
    quotas = {d: math.floor(ideal[d]) for d in ideal}
    for d in sorted(ideal, key=lambda d: (-(ideal[d]-quotas[d]), d))[:50-sum(quotas.values())]:
        quotas[d] += 1
    result, selected = {s: [] for s in SPLITS}, []
    for difficulty in (1, 2, 3):
        chosen = ordered(by_difficulty[difficulty], seed, f'appworld-scenarios:{difficulty}')[:quotas[difficulty]]
        selected.extend(chosen)
        variants = ordered([1, 2, 3], seed, f'appworld-variants:{difficulty}')
        # Counterbalanced cyclic permutations: variant indices are not difficulty.
        for i, scenario in enumerate(chosen):
            for si, split in enumerate(SPLITS):
                result[split].append(f'{scenario}_{variants[(i+si)%3]}')
    for split in SPLITS:
        result[split] = ordered(result[split], seed, f'appworld-order:{split}')
    extra = {'scenario_ids': sorted(selected), 'difficulty_scenario_counts': quotas,
             'scenario_assignment': 'One distinct variant per split; variant suffix counts counterbalanced.'}
    provenance = {'dataset_file': str(dataset), 'dataset_sha256': digest(dataset.read_bytes()),
        'metadata_aggregate_sha256': digest(canonical(file_hashes).encode()),
        'pool_tasks': len(ids), 'pool_scenarios': len(scenarios),
        'pool_scenarios_by_difficulty': {d: len(v) for d, v in by_difficulty.items()}}
    return result, metadata, extra, provenance


def terminal_split(root, seed):
    files = sorted(root.glob('*/task.toml'))
    records = [(f.parent.name, tomllib.loads(f.read_text())['metadata']) for f in files]
    n = len(records)
    if n != 89:
        raise ValueError('Expected all89 pinned Terminal-Bench2.0 tasks')
    if Counter(m['difficulty'] for _, m in records) != Counter(easy=4, medium=55, hard=30):
        raise ValueError('Terminal-Bench difficulty pool changed')
    categories = sorted({m['category'] for _, m in records})
    category_sizes = Counter(m['category'] for _,m in records)
    singleton_indices = [i for i,(_,m) in enumerate(records) if category_sizes[m['category']] == 1]
    singleton_ideal = [len(singleton_indices)*size/n for size in TB_SIZES]
    singleton_targets = [math.floor(v) for v in singleton_ideal]
    for si in sorted(range(3),key=lambda s: (-(singleton_ideal[s]-singleton_targets[s]),s))[:len(singleton_indices)-sum(singleton_targets)]:
        singleton_targets[si] += 1
    cells = sorted({(m['category'], m['difficulty']) for _, m in records})
    p = 3*n
    variables = p+3*len(cells)
    constraint_rows, lower, upper = [], [], []

    def constrain(coefficients, lo, hi):
        constraint_rows.append(coefficients); lower.append(lo); upper.append(hi)

    def count(indices, lo, hi):
        constrain({j: 1 for j in indices}, lo, hi)

    for i in range(n):
        count([3*i+s for s in range(3)], 1, 1)
    for s, size in enumerate(TB_SIZES):
        count([3*i+s for i in range(n)], size, size)
        # Rare categories must not all collect in one split. All nonsingleton
        # categories have at least3 tasks in this pinned pool, so this also
        # balances the number of represented categories across the splits.
        count([3*i+s for i in singleton_indices],singleton_targets[s],singleton_targets[s])
        for di, difficulty in enumerate(TB_DIFFICULTY):
            target = TB_DIFFICULTY_TARGETS[s][di]
            count([3*i+s for i, (_, m) in enumerate(records) if m['difficulty'] == difficulty], target, target)
        for category in categories:
            indices = [3*i+s for i, (_, m) in enumerate(records) if m['category'] == category]
            # Every category's three counts differ by at most one.
            count(indices, len(indices)//3, (len(indices)+2)//3)
        for ci, cell in enumerate(cells):
            indices = [3*i+s for i, (_, m) in enumerate(records) if (m['category'],m['difficulty']) == cell]
            target = len(indices)*size/n
            z = p+3*ci+s
            constrain({**{j: 1 for j in indices}, z: -1}, -np.inf, target)
            constrain({**{j: -1 for j in indices}, z: -1}, -np.inf, -target)
    matrix = lil_matrix((len(constraint_rows), variables))
    for i, coefficients in enumerate(constraint_rows):
        for j, value in coefficients.items():
            matrix[i, j] = value
    cost = np.ones(variables)
    randomizer = rng(seed, 'terminal-tie-break')
    cost[:p] = [randomizer.random()*1e-6 for _ in range(p)]
    integrality = np.zeros(variables); integrality[:p] = 1
    high = np.full(variables, np.inf); high[:p] = 1
    solved = milp(cost, integrality=integrality, bounds=Bounds(np.zeros(variables),high),
        constraints=LinearConstraint(matrix.tocsr(),lower,upper), options={'time_limit':30, 'mip_rel_gap':0.0})
    if not solved.success or solved.x is None:
        raise RuntimeError(f'Terminal-Bench allocation not proven optimal: {solved.message}')
    assignment = np.rint(solved.x[:p]).astype(int).reshape(n,3)
    if not np.all(assignment.sum(axis=1) == 1):
        raise RuntimeError('Nonintegral Terminal-Bench assignment')
    result = {s: [] for s in SPLITS}
    metadata = {}
    for i, (task, m) in enumerate(records):
        result[SPLITS[int(np.argmax(assignment[i]))]].append(task)
        metadata[task] = {'category':m['category'], 'difficulty':m['difficulty'], 'tags':m.get('tags',[])}
    for split in SPLITS:
        result[split] = ordered(result[split], seed, f'terminal-order:{split}')
    solver_info = {'method':'scipy.optimize.milp/HiGHS', 'optimal':True, 'mip_gap':float(solved.mip_gap),
        'hard_constraints':['each task once','sizes30/30/29','difficulty targets','each category max split count difference1','singleton categories distributed2/2/1'],
        'soft_objective':'minimize category×difficulty cell absolute deviations from pool proportions; deterministic seed-based tie-break',
        'objective_with_tiebreak':float(solved.fun)}
    provenance = {'task_root':str(root), 'task_toml_aggregate_sha256':digest(canonical([(f.parent.name,digest(f.read_bytes())) for f in files]).encode()), 'pool_size':n}
    return result, metadata, solver_info, provenance


def split_stats(ids, metadata, field):
    return dict(sorted(Counter(metadata[t].get(field) for t in ids).items(),key=lambda p:str(p[0])))


def validate(manifest, all_metadata):
    result = {}
    for benchmark, section in manifest['benchmarks'].items():
        used = set()
        for split in SPLITS:
            ids = section[split]
            if len(ids) != section['split_counts'][split] or len(ids) != len(set(ids)) or used & set(ids):
                raise ValueError(f'Split size/duplicate/overlap: {benchmark}:{split}')
            if not set(ids) <= set(all_metadata[benchmark]):
                raise ValueError(f'Unknown task ID: {benchmark}:{split}')
            used.update(ids)
        result[benchmark] = {'split_counts':section['split_counts'], 'selected_tasks':len(used),
            'unused_tasks':len(all_metadata[benchmark])-len(used), 'task_id_overlap':False,
            'membership_sha256':{s:digest(canonical(sorted(section[s])).encode()) for s in SPLITS},
            'ordered_ids_sha256':{s:digest(canonical(section[s]).encode()) for s in SPLITS}}
    aw = manifest['benchmarks']['appworld']
    scenario_sets = [{t.rsplit('_',1)[0] for t in aw[s]} for s in SPLITS]
    if not scenario_sets[0] == scenario_sets[1] == scenario_sets[2] or len(scenario_sets[0]) != 50:
        raise ValueError('AppWorld must share exactly50 scenarios across splits')
    if any(len([t for s in SPLITS for t in aw[s] if t.rsplit('_',1)[0] == sc]) != 3 for sc in scenario_sets[0]):
        raise ValueError('AppWorld variant assignment invalid')
    aw_difficulties = [split_stats(aw[s],all_metadata['appworld'],'difficulty') for s in SPLITS]
    if aw_difficulties[0] != aw_difficulties[1] or aw_difficulties[1] != aw_difficulties[2]:
        raise ValueError('AppWorld difficulty imbalance')
    result['appworld'].update(scenarios_per_split=50,difficulty_counts=aw_difficulties[0],
        variants_by_split={s:split_stats(aw[s],all_metadata['appworld'],'variant') for s in SPLITS})
    bfcl = manifest['benchmarks']['bfcl']
    for si, s in enumerate(SPLITS):
        counts = split_stats(bfcl[s],all_metadata['bfcl'],'primary_domain')
        if sorted(counts.values()) != [12,12,13,13]:
            raise ValueError('BFCL domain imbalance')
    tb = manifest['benchmarks']['terminalbench2']
    categories = sorted({m['category'] for m in all_metadata['terminalbench2'].values()})
    result['terminalbench2']['category_counts'] = {}
    for category in categories:
        counts = [sum(all_metadata['terminalbench2'][t]['category'] == category for t in tb[s]) for s in SPLITS]
        if max(counts)-min(counts) > 1:
            raise ValueError(f'Terminal category imbalance: {category}')
        result['terminalbench2']['category_counts'][category] = dict(zip(SPLITS,counts))
    for si, split in enumerate(SPLITS):
        counts = split_stats(tb[split],all_metadata['terminalbench2'],'difficulty')
        if [counts[d] for d in TB_DIFFICULTY] != list(TB_DIFFICULTY_TARGETS[si]):
            raise ValueError('Terminal difficulty target mismatch')
    represented = {s:len({all_metadata['terminalbench2'][t]['category'] for t in tb[s]}) for s in SPLITS}
    if max(represented.values())-min(represented.values()) > 1:
        raise ValueError('Terminal category coverage imbalance')
    result['terminalbench2']['represented_categories'] = represented
    return {'passed':True, 'shared_for_experiment_seeds':[42,43,44], 'benchmarks':result}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def report(manifest, all_metadata, validation):
    bm = manifest['benchmarks']
    text = ['**공통 fixed balanced split v1**',
        f"분할 seed={manifest['split_seed']}. 동일 task ID와 manifest 순서를 모든 방법 및 실험 seeds42/43/44에서 공유합니다.",
        table(['벤치마크','Train','Val','Test','미사용'], [[b,*[len(bm[b][s]) for s in SPLITS],validation['benchmarks'][b]['unused_tasks']] for b in bm]),
        '**BFCL: multi_turn_base**', '주요 도메인별 무작위 추출 후12/13개 할당. 세 split의 task ID는 서로 겹치지 않습니다.',
        table(['도메인','Train','Val','Test','미사용'], [[DOMAIN_NAMES[d],*[sum(all_metadata['bfcl'][t]['primary_domain']==d for t in bm['bfcl'][s]) for s in SPLITS],50-sum(all_metadata['bfcl'][t]['primary_domain']==d for s in SPLITS for t in bm['bfcl'][s])] for d in PRIMARY]),
        '**AppWorld: test_challenge**',
        '139개 원본 시나리오에서 난이도 비율에 맞춰50개를 선정했습니다. 시나리오마다 변형1/2/3을 서로 다른 split에 하나씩 배정했습니다. 각 split은 같은50개 시나리오를 포함하며 task ID는 서로 겹치지 않습니다. task ID suffix는 난이도가 아니라 변형 번호입니다.',
        table(['난이도','원본 시나리오','선정 시나리오','Train','Val','Test'], [[d,manifest['sources']['appworld']['pool_scenarios_by_difficulty'][d],bm['appworld']['difficulty_scenario_counts'][d],*[sum(all_metadata['appworld'][t]['difficulty']==d for t in bm['appworld'][s]) for s in SPLITS]] for d in [1,2,3]]),
        table(['변형 번호','Train','Val','Test'], [[v,*[sum(all_metadata['appworld'][t]['variant']==v for t in bm['appworld'][s]) for s in SPLITS]] for v in [1,2,3]]),
        '**Terminal-Bench2.0**',
        '89개 task 전체를30/30/29개로 배정했습니다. 모든16개 category의 split 간 task 개수 차이는 최대1입니다. task가1개뿐인5개 category도2/2/1개씩 분산해 category 다양성을 맞췄습니다. 아래 난이도 목표를 동시에 만족시키며 category×difficulty 조합의 비율 편차도 최소화했습니다. 난이도는 데이터셋의 작성자 메타데이터 기준입니다.',
        table(['난이도','원본 개수/비율','Train','Val','Test'], [[d,f"{sum(m['difficulty']==d for m in all_metadata['terminalbench2'].values())} ({100*sum(m['difficulty']==d for m in all_metadata['terminalbench2'].values())/89:.2f}%)",*[f"{sum(all_metadata['terminalbench2'][t]['difficulty']==d for t in bm['terminalbench2'][s])} ({100*sum(all_metadata['terminalbench2'][t]['difficulty']==d for t in bm['terminalbench2'][s])/len(bm['terminalbench2'][s]):.2f}%)" for s in SPLITS]] for d in TB_DIFFICULTY]),
        table(['Category','Train','Val','Test','총계'], [[cat,*[counts[s] for s in SPLITS],sum(counts.values())] for cat,counts in validation['benchmarks']['terminalbench2']['category_counts'].items()]),
        '**재현 및 검증**',
        '현재 실험의 성공률/완료 여부는 분할에 사용하지 않았습니다. 생성된 manifest는 기존 select_task_split_sections의 공통 benchmarks 형식이므로 실험 seed와 무관하게 동일 구성을 읽습니다. 원본 fingerprint, 정책, 선택 시나리오, 검증 hash를 함께 저장했습니다. train-as-val 실험은 이 원본 분할을 유지한 채 학습 중 검증 입력만train으로 연결하면 됩니다.']
    return '\n\n'.join(text)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bfcl-data',type=Path,required=True)
    parser.add_argument('--appworld-data',type=Path,default=Path.home()/'.exgentic/appworld/data')
    parser.add_argument('--terminal-root',type=Path,default=Path.home()/'.exgentic/benchmarks/terminalbench2/terminal-bench-2')
    parser.add_argument('--split-seed',type=int,default=42)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    bfcl,bmeta,bsource = bfcl_split(args.bfcl_data,args.split_seed)
    aw,ameta,aextra,asource = appworld_split(args.appworld_data,args.split_seed)
    tb,tmeta,solver,tsource = terminal_split(args.terminal_root,args.split_seed)
    all_metadata = {'bfcl':bmeta,'appworld':ameta,'terminalbench2':tmeta}
    manifest = {'schema_version':2,'split_version':'fixed_balanced_v1','split_seed':args.split_seed,'seed':args.split_seed,
        'experiment_seeds':[42,43,44], 'shared_membership_across_seeds':True,
        'policy':'Frozen shared split; BFCL primary-domain balance; AppWorld50 common scenarios one variant/split; TB category+difficulty balance.',
        'sources':{'bfcl':bsource,'appworld':asource,'terminalbench2':tsource},'terminal_allocation_solver':solver,
        'generator_sha256':digest(Path(__file__).read_bytes()), 'benchmarks':{}}
    for b, splits, kwargs in [('bfcl',bfcl,{'subset':'multi_turn_base'}),('appworld',aw,{'subset':'test_challenge'}),('terminalbench2',tb,{'subset':'2.0'})]:
        manifest['benchmarks'][b]={'benchmark_kwargs':kwargs,'split_counts':{s:len(splits[s]) for s in SPLITS},**splits}
    manifest['benchmarks']['appworld'].update(aextra)
    validation = validate(manifest,all_metadata)
    output = args.output
    serialized = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    # Preflight conflicts before writing any frozen output.
    validation['manifest_sha256'] = digest(serialized)
    outputs = {output:serialized,
        output.with_suffix('.validation.json'):(json.dumps(validation,ensure_ascii=False,indent=2)+'\n').encode(),
        output.with_suffix('.report.md'):report(manifest,all_metadata,validation).encode()}
    import io
    stream=io.StringIO(newline='')
    fields=['benchmark','task_id','split','scenario_id','variant','difficulty','primary_domain','category','user_turns','num_apps','num_apis','num_solution_code_lines','involved_classes','tags']
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for benchmark, records in all_metadata.items():
        lookup={task:s for s in SPLITS for task in manifest['benchmarks'][benchmark][s]}
        for task, meta in sorted(records.items()):
            row={'benchmark':benchmark,'task_id':task,'split':lookup.get(task,'unused'),**meta}
            for key in ['involved_classes','tags']:
                if key in row:row[key]=json.dumps(row[key],ensure_ascii=False)
            writer.writerow(row)
    outputs[output.with_suffix('.metadata.csv')]=stream.getvalue().encode()
    for path, content in outputs.items():
        if path.exists() and path.read_bytes()!=content:
            raise ValueError(f'Refusing to overwrite different frozen data: {path}')
    for path,content in outputs.items():immutable_write(path,content)
    print(json.dumps({'manifest':str(output),'validation':validation,'terminal_solver':solver},ensure_ascii=False))


if __name__=='__main__':
    main()

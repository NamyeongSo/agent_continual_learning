from types import SimpleNamespace
import json
import random

import pytest

from exgentic.agents.premise.pipeline import PremisePipeline, write_json
from exgentic.agents.premise.selection import select_beam


@pytest.mark.parametrize('past', [False, True])
def test_removing_alignment_preserves_all_official_score_tie_breakers(past):
    rng = random.Random(42)
    for _ in range(60):
        candidates = [dict(id=f'candidate_{i}', valid=True, current_alignment=1,
                           past_alignment=1 if past else None,
                           current_official_score=rng.choice([0, .25, .5, .75, 1]),
                           past_official_score=rng.choice([0, .25, .5, .75, 1]) if past else None)
                      for i in [12, 2, 1, 10]]
        expected, _ = select_beam(candidates)
        stripped = [{k: v for k, v in item.items() if 'alignment' not in k} for item in candidates]
        actual, details = select_beam(stripped, alignment_scoring=False)
        assert [x['id'] for x in actual] == [x['id'] for x in expected]
        assert details['alignment_scoring'] is False
        assert 'alignment_fronts' not in details


def test_no_alignment_selects_benchmark_performance_despite_conflicting_alignment():
    candidates = [dict(id=f'candidate_{i}', valid=True, current_alignment=alignment,
                       past_alignment=alignment, current_official_score=score, past_official_score=score)
                  for i, alignment, score in [(1, 1, 0), (2, 1, .1), (3, 0, 1), (4, 0, .8)]]
    assert [x['id'] for x in select_beam(candidates)[0]] == ['candidate_2', 'candidate_1']
    assert [x['id'] for x in select_beam(candidates, alignment_scoring=False)[0]] == ['candidate_3', 'candidate_4']


def test_full_pipeline_keeps_replay_and_memory_without_any_alignment_calls(tmp_path):
    manifest = tmp_path / 'split.json'
    write_json(manifest, {'bfcl_42': {'train': ['one', 'two']}})
    options = dict(output=tmp_path / 'run', manifest=manifest, benchmarks=['bfcl'], mode='isolated',
                   seed=42, count=None, past_count=3, model='openai/test', base_url='http://localhost/v1')
    pipeline = PremisePipeline(**options, alignment_scoring=False)
    calls = []

    def run_agent(**kwargs):
        calls.append(kwargs)
        return dict(benchmark='bfcl', task_id=kwargs['task_id'], actions=[{'action': {}}], score=.5, details={})

    pipeline._run_agent = run_agent
    pipeline.llm = SimpleNamespace(
        reflect=lambda *_a, **_kw: {'span': {'start': 0, 'end': 1}, 'entry_assessments': []},
        mutate=lambda *_: {'operations': [{'type': 'ADD', 'section': 'strategies_and_insights', 'content': 'Inspect observed inputs before acting.'}], 'reasoning': 'same guidance'},
        judge=lambda *_: pytest.fail('alignment judge must never run'),
    )
    first, second = pipeline.step(), pipeline.step()
    assert pipeline.cursor == 2 and len(pipeline.states['bfcl']['beam']) == 2
    assert second['past_sample'] and any('replay_source' in call for call in calls)
    assert any('past' in call['label'] for call in calls)
    for artifact in [first, second]:
        assert artifact['selection']['alignment_scoring'] is False
        assert all('current_alignment' not in candidate and 'alignment' not in candidate['current_result']
                   for candidate in artifact['candidates'])
    saved = json.loads((pipeline.output / 'state.json').read_text())
    assert saved['config']['alignment_scoring'] is False
    assert PremisePipeline(**options, alignment_scoring=False).cursor == 2
    with pytest.raises(ValueError, match='configuration differs'):
        PremisePipeline(**options)

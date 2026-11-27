# SPDX-License-Identifier: Apache-2.0

from scripts.utils.attribute_checkpoint_usage import Attribution, new_rows


def test_shared_bridge_costs_require_exact_checkpoint_response_identity_in_either_arrival_order():
    ledger = Attribution()
    row = {"usage": {"prompt_tokens": 10, "completion_tokens": 7},
           "choices": [{"reasoning_tokens": 5, "visible_tokens": 2}]}
    ledger.shared.update(checkpoint=row, original_training=row)
    ledger.dedicated["dedicated_checkpoint"] = row
    usage, excluded = ledger.summary()
    assert usage["8006"]["requests"] == 0 and usage["8007"]["requests"] == 1 and excluded == []
    ledger.ids.add("checkpoint")
    usage, excluded = ledger.summary()
    assert usage["8006"] == {"requests": 1, "prompt_tokens": 10, "completion_tokens": 7,
                             "reasoning_tokens": 5, "visible_tokens": 2}
    assert excluded == ["checkpoint"]
    ledger.ids.add("checkpoint")
    assert ledger.summary() == (usage, excluded)


def test_incremental_attribution_keeps_incomplete_jsonl_until_the_line_is_committed(tmp_path):
    path = tmp_path / "trace.jsonl"
    positions = {}
    path.write_text('{"response":{"id":"a"}}\n{"response":')
    assert list(new_rows(path, positions)) == [{"response": {"id": "a"}}]
    assert list(new_rows(path, positions)) == []
    with path.open("a") as file:
        file.write('{"id":"b"}}\n')
    assert list(new_rows(path, positions)) == [{"response": {"id": "b"}}]

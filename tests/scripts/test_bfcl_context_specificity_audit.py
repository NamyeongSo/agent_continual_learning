# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Rule-boundary extraction and complete, valid close-reading annotation coverage."""

import pytest

from scripts.premise.prepare_bfcl_context_specificity_audit import clean, entries
from scripts.premise.summarize_bfcl_context_specificity import labels_for


def test_actual_entry_boundaries_preserve_nested_rendering_and_multiline_content():
    text = ("## SECTION\n[ph-00001] helpful=2 harmful=1 :: Keep state.\nAcross turns.\n"
            "[ph-00002] helpful=0 harmful=0 :: [ph-00002] helpful=1 harmful=0 :: Check IDs.\n"
            "## NEXT\n[err-00003] helpful=0 harmful=0 :: Respect schema.")
    parsed = entries(text)
    assert set(parsed) == {"ph-00001", "ph-00002", "err-00003"}
    assert clean(parsed["ph-00001"]) == "Keep state. Across turns."
    assert clean(parsed["ph-00002"]) == "Check IDs."
    assert "NEXT" not in parsed["ph-00002"]


@pytest.mark.parametrize("labels", [[], [{"audit_id": "N002", "scope": "A", "flags": []}],
                                   [{"audit_id": "N001", "scope": "A", "flags": []}] * 2])
def test_missing_extra_and_duplicate_annotations_rejected(labels):
    with pytest.raises(ValueError, match="exactly one"):
        labels_for([{"audit_id": "N001"}], labels)


@pytest.mark.parametrize("scope,flags", [("", []), ("GA", []), ("Z", []), ("G", ["error"])])
def test_invalid_category_or_unapproved_harmfulness_claim_rejected(scope, flags):
    with pytest.raises(ValueError, match="Unknown"):
        labels_for([{"audit_id": "N001"}], [{"audit_id": "N001", "scope": scope, "flags": flags}])


def test_benchmark_protocol_is_separate_from_api_family_specialization():
    rows = [{"audit_id": "N001"}, {"audit_id": "N002"}]
    labels = [{"audit_id": "N001", "scope": "P", "flags": ["T"]},
              {"audit_id": "N002", "scope": "A", "flags": []}]
    result = labels_for(rows, labels)
    assert result["N001"]["scope"] == "P"
    assert result["N002"]["scope"] == "A"

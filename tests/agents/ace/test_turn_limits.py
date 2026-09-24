from argparse import Namespace

from scripts.ace.run_experiment import resolve_max_turns


def test_benchmark_specific_turn_limits_override_global_fallback():
    args = Namespace(
        max_turns=30,
        appworld_max_turns=50,
        bfcl_max_turns=50,
        swebench_max_turns=100,
    )
    assert resolve_max_turns(args, "appworld") == 50
    assert resolve_max_turns(args, "bfcl") == 50
    assert resolve_max_turns(args, "swebench") == 100
    assert resolve_max_turns(args, "browsecompplus") == 30

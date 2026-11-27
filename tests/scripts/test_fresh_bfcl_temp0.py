# SPDX-License-Identifier: Apache-2.0
"""The fresh run starts all first-stage jobs and staggers only optimizer seeds."""
from scripts.utils.launch_bfcl_temp0_two_epochs import release_delay


def test_seed_pair_stagger_is_eight_then_ten_then_twelve():
    jobs = [(method, seed) for method in ('ace', 'premise', 'gepa', 'skillopt') for seed in (42, 43, 44)]
    delays = {(method, seed): release_delay(method, seed) for method, seed in jobs}
    assert sum(delay == 0 for delay in delays.values()) == 8
    assert {job for job, delay in delays.items() if delay == 600} == {('gepa', 43), ('skillopt', 43)}
    assert {job for job, delay in delays.items() if delay == 1200} == {('gepa', 44), ('skillopt', 44)}

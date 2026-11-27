# Terminal-Bench transport recovery and automatic container cleanup

The 2026-10-23 audit found completed queue results stalled for more than two
hours, with all 128 Docker SSH channels held by Compose exec connections.
Ordinary Docker HTTP keep-alive connections exhausted the transport before
clients could open the separate connection required for exec/attach hijacking.
Cancelled Harbor setup/verifier calls also left Compose and SSH descendants
alive, and unbounded teardown prevented errors from reaching the retry queue.

Increasing the pool to 64 masters reduced starvation but still produced SSH
session refusals. The final transport in `scripts/utils/docker_ssh_pool.py`
reuses an authenticated master to forward the remote Docker Unix socket into
a private local socket (directory mode 0700, socket mode 0600). Each Docker
connection uses an SSH direct-streamlocal channel instead of a remote shell
session, avoiding the SSH MaxSessions bottleneck. No TCP listener is exposed.
`scripts/utils/docker_stdio.py` closes ordinary HTTP connections after their
response and preserves exec/attach upgrades. Noninteractive exec may close
stdin before reading stdout; that half-close preserves the response direction.
Other SSH commands continue through the system client. Actor model,
temperature, seeds, splits, token/turn policy, optimizer budgets and the
existing native workers and 100 shared inference workers are unchanged.

A busy Unix listener must not be treated as a stale forward: connect timeouts
preserve its path. When a path is genuinely missing/refused, the wrapper first
cancels the registered SSH listener before rebinding; otherwise OpenSSH can
deduplicate the forward while leaving no usable socket. This fixes the repeated
`FileNotFoundError` during Docker setup, exec and verifier file transfers.

A daemon goroutine profile also identified hundreds of bridge-network creates
waiting behind Docker's global bridge mutex and firewall rule updates. Bridge
creation/deletion requests now acquire a shared local lock before being sent.
Cancelled local waiters therefore do not add more orphaned daemon requests.
Actor model calls and container exec streams do not acquire this lock.

Compose commands run in their own process group. Cancellation terminates the
CLI and its SSH descendants. Existing explicit command and official build
deadlines are retained; otherwise infrastructure commands have a 180-second
bound. Verifier `exec` calls without a command deadline are governed by the
official verifier deadline, not this infrastructure cap. Harbor omits an
explicit command timeout for `test.sh`; applying 180 seconds there interrupted
valid tests such as `query-optimize`, whose official verifier limit is 1800
seconds. The October 23 correction preserves that full limit and still cancels
the Compose process group when the official deadline expires.
Teardown has a separate 90-second bound and defers failures to cleanup,
without discarding an already saved official score. Successful and interrupted
sessions retain host-side trajectories, verifier artifacts and diagnostics.

Each session records its unique Compose project, PID and process start identity,
hostname and lifecycle phase under
`~/.cache/exgentic/terminalbench2/leases/`. PID reuse is checked by start identity.
`scripts/utils/cleanup_terminalbench_containers.py` runs independently of the
learning controllers and uses the daemon's Unix HTTP API through the same
private forwarded socket. It avoids fresh authentication and expensive CLI
size enumeration. It removes containers only when:

- The Compose project is owned by Exgentic and the working directory is within
  this user's configured Terminal-Bench task root.
- A lifecycle record proves the actor/verifier has finished, or its original
  owner has exited. Newly finished records have a 30-second grace period.
- Older unleased executions have no live session or exact Compose command and
  are at least ten minutes old. Unknown ownership is preserved.

The running process list is refreshed immediately before deletion, and Docker
labels are rechecked on the daemon. Other projects are excluded. Images and
volumes are retained; standalone network cleanup is disabled. Aborted
pre-recovery container instances can be explicitly retired using their exact
immutable IDs after their old actor processes are stopped. New executions use
different IDs and project nonces.

The current detached cleanup worker scans between batches every ten seconds,
removing at most ten containers. It requires explicit `--experiment-root`
arguments and a lifecycle artifact path inside one of those roots; a shared
benchmark directory or project prefix alone is insufficient. Network cleanup
is disabled by default and is not performed as a side effect of removing a
container. Earlier network cleanup operations remain in the audit log.
A slow batch can take longer than the
scan interval. Plans, per-batch events and status are in
`outputs/terminalbench_speed_audit_20261023/cleanup/`; the same parent directory
contains source overlays, interruption evidence and controller recovery records.
The private process handoff contains the original environment and must never be
uploaded or committed. Existing inference records and official failures are
preserved; interrupted requests without an official outcome resume through the
durable queue.

Cleanup candidates that time out back off for 30 seconds to ten minutes. Other
finished containers remain eligible, so a few locked containers cannot starve
the whole cleanup queue. Actor execution errors propagate to infrastructure
error handling before verifier scoring; they cannot become measured zero scores.
Actor token/turn failures still take precedence and retain their official zero.

PREMiSE tails skipped during the transport outage are retained in their original
directories. Recovery copies only the unaffected prefix into separate
`s*_transport_recovery_v4` outputs after replaying and verifying the complete
beam, memory and RNG state. Old checkpoint streams are invalidated and retained;
matching valid prefix evaluations can be reused by policy hash. Models, task
order, seeds and optimization settings are preserved, and old costs remain in
the original artifacts. The audit directory records the exact prefix boundaries.

Atomic final actor responses are recovered when an interrupted parent has not
yet committed its result cache. Recovery requires matching the full request,
task ID and policy hash; official budget failures retain their original zero
score and are not evaluated again as infrastructure retries.

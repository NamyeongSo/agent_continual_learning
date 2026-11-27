# Experiment result backups through Notion MCP

`scripts/utils/notion_experiment_backup.py` builds full and incremental snapshots of explicitly configured experiment
directories. The detached watcher runs every 1,800 seconds, with one cycle active at a time. A cycle that runs longer
than the interval skips missed slots; failed uploads resume from their saved snapshot before new snapshots are built.
The watcher requires this host to remain running. Restart its frozen launch command after a host reboot.

Each experiment has a dedicated Notion subpage under the user-selected destination. Each snapshot adds a child page
containing a manifest and downloadable ZIP volumes. The uploader uses `codex exec` with the user's existing Notion MCP
OAuth connection; it does not require copying a Notion integration token into the experiment. It uses Notion's MCP
upload tool and its returned short-lived multipart upload URL, then attaches each file to the snapshot page.

The first snapshot contains every eligible file, including original results, plots, trajectories, calls, checkpoints,
policies and frozen source. Subsequent snapshots contain bytes whose SHA256 changed and tombstones for removed files.
When a result reuses initial trajectories stored in an earlier cohort, include that cohort's `stage1` and provenance
files with `include_paths`, under the same experiment's backup parent. Do not mix its obsolete proposals into the
current cohort. Source directories must exist; an unavailable directory is an error rather than a mass deletion.
Reinstallable dependency directories, virtual environments, caches, symlinks, locks, temporary files and credential files
are excluded. A file that changes while being read is deferred to the next cycle and recorded in the manifest.
Snapshots are collections of stable individual files, not atomic snapshots of a running experiment.

ZIP volumes contain numbered pieces of a gzip-compressed tar archive, each below the MCP single-part upload limit.
Unzip all volumes for one snapshot, concatenate `payload.tar.gz.part*` in numeric order, then extract the resulting tar.gz.
Apply snapshots chronologically and use `deleted_files` from each manifest to reproduce deletions. The manifest retains
the complete eligible file inventory and SHA256 hashes; prior snapshots remain attached in Notion.

Configuration lives in an ignored, private state directory, outside the source experiment directories:

```json
{
  "interval_seconds": 1800,
  "workspace": "/absolute/project/path",
  "codex": "/absolute/path/to/codex",
  "sources": [{"path": "/absolute/experiment/path", "page_id": "notion-parent-page-id"}]
}
```

Run the watcher using its frozen helper and existing project environment:

```bash
python /private/state/backup_runtime.py --watch --state /private/state
```

`status.json` gives the last successful cycle, errors and next run time. `watcher.log` records cycle progress.
Each snapshot keeps upload progress and a receipt. Its source watermark advances only after all files are attached;
partial uploads never count as completed backups. Create a `STOP` file to stop before the next cycle.
The watcher lock prevents duplicate controllers. Keep the state directory private because upload response files and
MCP event logs can contain short-lived upload credentials. The helper never changes source experiment files.

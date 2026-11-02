# Model and prediction checkpoints

[The reproduction guide](../../docs/reproducibility.md) contains the current
commands. Checkpoints and source inputs are excluded from the public snapshot.

The final hybrid replay used the frozen prediction snapshot
`annual_turnover_400pct_20261028_v2`. Local checkpoint directories and archives
are preserved under `local/artifacts/final_v3/checkpoints/`. Extract an archive
into a separate directory before supplying it to `--from-predictions` or
`--from-weights`.

| Mode | Work performed |
| :--- | :--- |
| Normal run | Train forecasts, construct the portfolio and report |
| `--from-predictions PATH` | Verify saved forecasts, rebuild construction and report |
| `--from-weights PATH` | Verify saved allocations and regenerate reporting |
| `--save-checkpoint PATH` | Export a completed run to a new snapshot directory |

Checkpoints record payload hashes, input hashes, the evaluation period,
discovery mode, forecasts and portfolio allocations. Loading rejects changed
payloads, mismatched inputs, escaping paths and checkpoint/output overlap.
A replay does not extend forecasts beyond their archived holding months.

`--replay-with-local-reporting` is an explicit prediction-replay exception when
the original licensed training container is unavailable. It verifies forecast
payloads and competition inputs, and records the execution reporting source.
It does not claim to reopen or reverify the original training inputs. The
published replay retains this limitation.

Use a distinct `--output-dir` for experiments. `--reuse` is the ordinary
source/input-aware training cache; it is not a substitute for a verified
checkpoint. Checkpoint relocation preserves contents, but moved code or input
locations can invalidate a normal training cache and trigger rebuilding.

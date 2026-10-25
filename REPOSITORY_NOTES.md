# Repository after final submission

The finals framework design is in
[`POD_FRAMEWORK_IMPLEMENTATION_SPEC.md`](POD_FRAMEWORK_IMPLEMENTATION_SPEC.md).
The original submitted implementation and the research archive remain available
alongside that specification.

The repository root contains the contents of the submitted
`StochasticPirates_FIAM2026.zip`, including `MAIN.py`, the modular pipeline,
`submission/holdings.csv`, `submission/returns.csv`, `strategy_deck.pdf`, and `cvs/`.
The submitted README and source files are preserved byte for byte.
Only `.gitignore` has extra local archive/environment exclusions.
`submission_manifest.json` records the source ZIP checksum and every submitted
file checksum. An unchanged copy of the ZIP is kept locally under `local/`.

Earlier research, experiments, outputs, tests, and the previous `final/` tree
are preserved in `stuff_tried/`, which is excluded from Git. Local environments
remain in `local/` and `.deps/`. Nothing in the archive is included in a push.
The archive includes tracked and untracked work; Git history also retains the
previously committed files.

Competition inputs in `data/` and `external_data/external.parquet` are local
symlinks into the archive and are excluded from Git. A fresh checkout requires
these inputs separately, as described in README.md. Follow that README to run
the final pipeline from this repository root.

Workspace maintenance removes regenerable Python, test, lint and font caches. The
derived `stuff_tried/output/ideas/_cache/{panel,ranks}.parquet` files can be
recreated from retained raw inputs: from `stuff_tried/`, run
`python -c 'from src.ideas.common import build_cache; build_cache()'` in an
environment with the research dependencies. Saved research results and the
reference model configuration are retained.

Identical archived files share disk storage through filesystem copy-on-write
clones. Their contents, paths, permissions and modification times are preserved;
editing one file does not change another. These are ordinary independent files,
so standard disk-usage tools can count shared storage more than once. The
duplicate `construction_rerun.log` was removed; its complete contents remain in
`output/pod_design_20261021/research/book_construction/construction_study_output.txt`.

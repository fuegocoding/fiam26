# Publishing this project

The cleaned working tree separates team-authored source/results from local
data and personal material. It has not changed remote visibility or rewritten
the existing repository's history.

## Existing history

Earlier commits contain five team CV PDFs and historical Carbon Arc extracts.
Removing them from the latest tree or adding ignore rules does not remove
those earlier blobs. **Do not make the current history public as-is.**

A focused local scan found no matches for the checked provider-token,
AWS-key, private-key, JWT, embedded-credential and known-local-secret patterns
in the scanned tracked text or historical text blobs. The scan covered 1,891
historical text blobs; 13 text blobs over 5 MB were skipped. This is a bounded
pattern scan, not proof that every file or binary is free of sensitive content.

## Prepared source export

The cleanup produces `local/public-release/fiam26-public.zip`, containing
the cleaned source, final presentation, aggregate evidence and selected
historical code/decks. The accompanying `manifest.json` records each included
file's size and SHA-256. The ZIP excludes `.git`, `local/`, credentials,
personal reference files, raw vendor/competition inputs, trained models and
prediction caches.

To publish without the old history, extract the ZIP to a new directory and
initialize a new repository there:

```bash
git init
git add .
git diff --cached --stat
git commit -m "Publish Stochastic Pirates McGill-FIAM 2026 project"
```

Review the exact files before connecting or pushing to a public remote.
Another option is to sanitize the existing history, including branches,
tags and hosted copies, before changing visibility. No history rewrite or
remote operation was performed during this cleanup.

No new license has been selected on behalf of the team. Choose the intended
reuse terms with the project owners; third-party source data are excluded.

## Cleanup verification

- The preserved earlier implementation passes all 48 existing regression tests.
- Both strategy entrypoints load and return their CLI help successfully.
- All 162 public Python sources parse, and all 108 local Markdown links resolve.
- The 68 monthly returns reproduce the saved cumulative return, CAGR, annual
  returns, month-end drawdown, Sharpe and information ratio.
- The 17 copied result files retain their original hashes. The relocated beta
  input is byte-identical to the original local export.
- All 833 originally tracked paths are accounted for: 817 unchanged contents
  and 16 intentional documentation or path/help edits, with no missing files.

These checks cover publication cleanup and aggregate consistency. No fresh
full final-v3 training run, daily accounting reconciliation, live Arcana audit
or independent holdout evaluation was performed.

## Local preservation

Private/reference files remain under `local/private/`, full generated runs
under `local/artifacts/`, and earlier upload/research packages under
`local/archive/`. Source data remain in their ignored input locations. The
clean source export does not include any of those folders.

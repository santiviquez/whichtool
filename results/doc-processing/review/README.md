# Santiago's review of the classification (2026-11-08)

Every answer the scorer classified, grouped as in the central results spreadsheet:
one row per distinct stated answer (S001-S479, same text = same row) and one per
code answer (C001-C192).

- `reviews_20261108.csv`: one row per answer. `decision` is
  - `accepted`: reviewed in the review tool and kept as classified (82),
  - `changed`: reviewed and reclassified (2; these are in `tasks/doc-processing/manual_apis.yaml`),
  - `accepted_in_bulk`: not reviewed one by one. Santiago accepted Claude's classification
    of these after looking through it, and reviewed only the 80 answers Claude was unsure
    about (`claude_unsure_because`) plus 4 others (587 rows).
- `review_tool_export_20261108.jsonl`: the review tool's database as exported, unchanged.

`claude_apis` is the alias classification as of commit 3074927 (aliases.yaml of 2026-11-06).

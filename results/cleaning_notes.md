# Stage 1 cleaning notes

- The raw files are loaded as tab-separated files using the column schemas in
  the GroupLens archive README.
- `user_artists.dat` play counts are retained as implicit confidence values;
  they are not interpreted as star ratings.
- The exploratory interaction matrix is sparse and stores raw play counts.
- No interaction rows are removed in Stage 1. Duplicate keys, missing values,
  and ID coverage are reported below so later stages can make explicit,
  reproducible decisions.
- Train/validation/test splitting and leakage prevention are intentionally
  deferred to Stage 2.

## Basic data checks

- `user_artists`: 92834 rows, 0 missing cells, 0 duplicate rows.
- `artists`: 17632 rows, 444 missing cells, 0 duplicate rows.
- `tags`: 11946 rows, 0 missing cells, 0 duplicate rows.
- `user_taggedartists`: 186479 rows, 0 missing cells, 0 duplicate rows.
- Interaction matrix shape: (1892, 17632); stored nonzero entries: 92834.

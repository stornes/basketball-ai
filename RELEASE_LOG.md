# Release Log

Append-only record of shipped features, verified state, and known issues.
Read this at session start to know what's been built.

---

## v1.7.0-dx — 2026-03-17

### Shipped
- [x] VLM model switched from `gemini-3.1-pro-preview` (25 RPM) to `gemini-3-flash` (1000 RPM)
- [x] Stage 3 progress bar with %, frame count, fps, and ETA (updates every 50 frames)

### Verified
- CLI `--help` confirms all flags valid
- Both VLM model references updated in `jersey_number.py` (lines 40, 129)
- Progress bar uses `\r` carriage return for single-line in-place updates

### Known Issues
- Pipeline re-run in progress — VLM rate limit fix not yet validated end-to-end

---

## v1.6.2 — 2026-03-16

### Shipped
- [x] Quarter-by-quarter coaching narratives in Score by Quarter section
- [x] `quarter_narratives: list[str]` field on FilmReport dataclass
- [x] Auto-generated tactical analysis per quarter (opening control, half-time assessment, adjustment commentary, close-out analysis)
- [x] DOCX renderer embeds narratives as bold-labelled paragraphs below score table

### Verified
- 271 tests pass
- Report generated with 4 quarter narratives in JSON + DOCX
- Output: `~/Downloads/2026-03-14 — Notodden Thunders (D) 69 — EB-85 82.docx`

---

## v1.6.1 — 2026-03-16

### Shipped
- [x] Four Factors with real data (no pace estimation)
- [x] OREB detection from game flow analysis (possession transitions)
- [x] TOV detection from game flow analysis (same-team after made, long gaps)
- [x] Pipeline rebound re-labelling after team classification (run_analysis.py)
- [x] Gzip fix in fetch_game.py for kamper.basket.no API
- [x] Removed pace-based estimation from compile_film_report.py
- [x] Three-tier data provenance: API ground truth, pipeline heuristic, manual scorekeeper

### Verified
- 271 tests pass
- Report generated: all Four Factors graded from real data
- Output: `~/Downloads/2026-03-14T140700+0100 — Notodden Thunders (D) 69 — EB-85 82.docx`

### Known Issues
- Shot detection inflated ~3.5x (473 detected vs ~135 real FGA)
- TOV underdetected (5 home / 6 away vs likely 15-20 real)
- DREB inflated (88/84 — every missed shot counts as DREB)
- Possession-level grouping (256 possessions) still above real (~135)

---

## v1.6.0 — 2026-03-15

### Shipped
- [x] VLM-based jersey number detection
- [x] Shot charts per team per quarter (green X = made with jersey #, red X = missed)
- [x] Score flow chart (cumulative per-minute, Q1-Q4 vertical lines)
- [x] Team classification via jersey colour clustering (K-Means in Lab space)
- [x] Quarter assignment from timestamp
- [x] Roster JSON support with player name mapping
- [x] DOCX report generation via CreateDocx.ts
- [x] Game Summary, Four Factors, Individual Metrics, Scouting Reports, Coaching Assessment
- [x] API merge from kamper.basket.no (FGM, 3PM, FT, FTA ground truth)

### Verified
- Pipeline processes full game video end-to-end
- 11 shot chart PNGs generated per run
- Score flow chart matches actual quarter scores

---

## v1.5.0 — 2026-03-14

### Shipped
- [x] Quarter-based shot segmentation
- [x] Per-team per-quarter shot chart generation
- [x] Configurable quarter duration (default 10 min FIBA U16)

---

## v1.2.0 — 2026-03-13

### Shipped
- [x] Home/away team shot charts via jersey colour clustering
- [x] Made vs missed shot outcome (ball-through-hoop heuristic)
- [x] Green X = made, Red X = missed markers on shot chart

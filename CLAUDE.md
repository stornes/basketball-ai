# HoopsVision 2026 — Project Instructions

## Pre-Flight Checklist (MANDATORY)

Before ANY work on this project, complete these checks:

### 0. Read Release Log + Backlog
```bash
cat RELEASE_LOG.md  # What's shipped — don't break it
cat BACKLOG.md      # What's next — pick from here unless directed otherwise
```
This is the incremental improvement process. Every session starts by knowing what exists and what's pending.

### 1. Read All Manifests
```bash
ls docs/manifests/*.yaml
```
Read every manifest before designing or coding. The manifests are the source of truth for what the system should produce.

### 2. Manifest-First Development
**Never write code without a manifest.** If the work changes behaviour, output, or data:
1. Create a manifest YAML at `docs/manifests/vX.Y.Z-description.yaml` FIRST
2. Get approval
3. Then implement

### 3. Data Provenance Rules
- **API ground truth** (kamper.basket.no): FGM, 3PM, FTM, FTA — tagged `StatSource.MANUAL` but these are SCORING stats only
- **Pipeline heuristic**: OREB, DREB, AST, STL, TO, +/-, MIN — tagged `StatSource.HEURISTIC` or `StatSource.DETECTED`
- **Manual scorekeeper**: all stats including BLK, PF — tagged `StatSource.MANUAL` on counting-stat keys
- **CRITICAL**: `StatSource.MANUAL` on scoring stats does NOT imply counting-stat availability. Check the specific stat keys (`ast`, `orb`, `drb`, `stl`, `to`, `blk`, `pf`, `min_played`).

### 4. Missing Data = N/A, Not a Grade
When a stat is zero because data doesn't exist (not because the team performed poorly):
- Grade must be "N/A"
- Analysis must say "Data not available"
- Never present missing data as performance

### 5. Report Content Quality
- All narrative text: UK English, Barbara Minto style (lead with conclusion, cut filler)
- Test template output with REAL data before shipping
- Verify every claim in the report against actual data values
- eFG% > 100% must be explained (FGA = FGM from API)

### 6. Report Chapter Expectations
The v3.0.0 manifest defines 11 sections + appendix. Each has specific requirements — read them before touching any report generation code. Key gaps to track:
- Game Summary: 6 paragraphs, Minto pyramid, 75% page ✓ DONE
- Score by Quarter: per-quarter coaching narrative ✓ DONE + score flow chart ✓ DONE
  (remaining: half-by-half split table, shot detection volume table)
- Scouting Reports: LLM-generated 3-5 sentences per player
- Coaching Assessment: 4 categories with grades table
- Q4 Film Analysis: Gemini video analysis when 10+ point swing
- Cross-team advanced metrics ranked by Game Score

### 7. Date/Time Format — ISO 8601
All dates and times use **ISO 8601 with UTC offset**:
- Format: `2026-03-14T14:07:00+01:00` (CET example)
- In filenames, colons stripped: `2026-03-14T140700+0100`
- In prose/headers: `14 March 2026, 14:07 CET`
- Never bare datetime without timezone offset

### 8. Python Environment
- Use `.venv/bin/python` (not `python`)
- Run tests: `.venv/bin/python -m pytest tests/ -v`

## Architecture Quick Reference

| Layer | Module | Purpose |
|-------|--------|---------|
| Box Score | `app/analytics/box_score.py` | PlayerBoxScore, TeamBoxScore, GameBoxScore |
| Advanced Stats | `app/analytics/advanced_stats.py` | Four Factors, Game Score, TS%, eFG%, USG% |
| Report Generator | `app/reporting/film_report.py` | FilmReportGenerator — produces all report content |
| DOCX Renderer | `app/reporting/docx_renderer.py` | Renders report data to styled DOCX |
| Pipeline | `scripts/compile_film_report.py` | Orchestrates report generation |

## Key Design Decisions

- `_scan_stat_sources()` returns 3-tuple: `(has_manual_counting, has_heuristic_counting, has_manual_scoring)`
- FGA estimation: `Est. Poss = PTS / 1.8`, then `FGA = Est. Poss - 0.44 * FTA`
- Pseudo-player (jersey_number=0, "Unattributed") excluded from awards, scouting, coaching
- Four Factors grading benchmarks calibrated to Norwegian U16 competition

## Game Registry

| Date | Home | Away | Score | Video | API Match ID | Output Dir |
|------|------|------|-------|-------|--------------|------------|
| 2026-03-14T14:07:00+01:00 | Notodden Thunders (D) | EB-85 | 69–82 | `data/videos/2026-03-14_notodden-thunders-d_vs_eb-85.MP4` | 8254973 | `data/outputs/v1.7.0` |

### Pipeline Command Template
```bash
.venv/bin/python main.py analyse <VIDEO> \
  --model runs/detect/runs/basketball/finetune/weights/best.pt \
  --class-map '{"0":"ball","1":"basket","2":"player"}' \
  --output-dir <OUTPUT_DIR> \
  --roster <ROSTER_PATH> \
  --quarter-duration 600 \
  --game-start "<HH:MM:SS>" \
  --profile youth \
  --vlm-backend gemini \
  --llm-backend template \
  --no-agent
```

### Pipeline Performance (v1.7.0)
- **Video**: 4K (3840×2160) → decoded at 1920×1080 (`decode_width=1920`)
- **Sample rate**: default 6 (every 6th frame → ~4.7 effective fps from 28.2fps)
- **YOLO**: input always 960×540, runs on MPS (Apple Silicon GPU)
- **VLM jersey OCR**: Gemini 3 Flash (1000 RPM vs 25 RPM on 3.1 Pro preview), 5 concurrent workers
- **Runtime**: ~30-45 min for a 90-min 4K game (was ~3.5h before v1.7.0)
- **Cost**: ~$0.05-0.10 per game (Gemini pricing)
- **IMPORTANT**: Run from terminal, NOT from Claude Code (10min timeout kills long runs)

### Post-Pipeline: Compile Film Report
After pipeline completes, generate the game report with API ground truth:
```bash
.venv/bin/python scripts/compile_film_report.py \
  --data-dir <OUTPUT_DIR> \
  --match-id <API_MATCH_ID> \
  --output-dir <OUTPUT_DIR>/report \
  --competition "ØST GU16B" \
  --game-date "<YYYY-MM-DD>" \
  --docx
```

## v1.7.0 Architecture (Current)

### Detection Pipeline Stages
| Stage | Name | Description |
|-------|------|-------------|
| 1 | Video load | Decode at `decode_width` (1920), metadata |
| 2 | Init detectors | YOLO, tracker, shot/possession/ball-in-hands/pass detectors |
| 3 | Frame processing | YOLO detect → track → shot detect → possession → ball-in-hands → pass detect → rebound |
| 3.5 | Team classification | K-means on jersey colour crops → home/away labels |
| 3.5b | Possession re-run | Re-run possession tracker with real team labels |
| 3.5c | Assist/steal detection | Two-tier: pass-observed (Tier 1) + proximity fallback (Tier 2) |
| 3.6 | Jersey VLM | Gemini 3.1 Pro reads jersey numbers from player crops (5 concurrent) |
| 3.6b | Sherlock deduction | VLM deductive pass for unresolved tracks |
| 3.7 | Quarter boundaries | Scoreboard OCR for quarter start/end times |
| 4 | Analytics | Box score, advanced stats, shot charts |

### Key New Modules (v1.7.0)
- `app/events/ball_possession.py` — BallInHandsDetector (bbox overlap heuristic)
- `app/events/pass_detector.py` — PassDetector (ball trajectory between possessions)
- `app/events/assist_detector.py` — Two-tier: pass-based (Tier 1) + proximity (Tier 2)
- `app/events/event_types.py` — PassEvent dataclass

### Manifesto Status
- Manifesto: `docs/manifests/v1.7.0-fga-min-ast-manifesto.md` (rev6)
- 13/14 tasks complete, 1 deferred to v1.8.0 (VLM fallback for ambiguous possession)
- 306 tests passing

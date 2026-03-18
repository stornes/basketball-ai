# Backlog

Prioritised list of requested but unstarted/incomplete work.
Read this at session start to know what's next.

---

## Priority 1 — v1.7.0 Observational Stats (manifesto: docs/manifests/v1.7.0-fga-min-ast-manifesto.md)

### FGA Fix — Shot Over-Detection (478 detected, ~120 real)
- [ ] Resolution-aware thresholds: `MIN_VERTICAL_DISPLACEMENT = frame_height * 0.04` (1h)
- [ ] Basket proximity gate: only detect shots near baskets (2h)
- [ ] Temporal dedup: max 1 shot per 4s per basket (1h)

### MIN Fix — Playing Time Tracking (all players show 0 min)
- [ ] Fix key lookup in box_score.py: aggregate track frames per jersey number (1h)
- [ ] Multiply by sample_rate in frame-to-minutes conversion (included in above)

### AST Fix — True Observational Assists (not proximity inference)
- [ ] Store ball-player proximity during Stage 3 detection (30m)
- [ ] Wire Stage 3.5 team classification to PossessionTracker (30m)
- [ ] Scale POSSESSION_DISTANCE_PX to frame_width * 0.05 (included in above)
- [ ] Reduce MIN_POSSESSION_FRAMES to 2 (included in above)
- [ ] Post-processing possession pass after team classification (1h)
- [ ] Ball-in-hands detector: bbox overlap heuristic (ball center inside player bbox) (3h)
- [ ] Pass trajectory tracking: ball centroid path between possessions (3h)
- [ ] PassEvent emission + assist attribution from observed passes (2h)
- [ ] VLM fallback for ambiguous ball possession (1h)
- [ ] Verify end-to-end assist detection with real pass events (30m)

### Shot Chart Fix — Broken Coordinates (shots plot at midcourt, not near baskets)
- [ ] Basket-relative shot coordinates: use basket bbox as anchor instead of broken homography (2h)
- [ ] Half-court mirroring: far-basket shots mapped to near-basket frame of reference (30m)

**Total v1.7.0: ~19 hours**

## Priority 2 — Detection Accuracy (carried forward)

- [x] FGA attribution: jersey resolution now runs on ALL shots (not just made) — v1.6.3
- [x] AST detection: flows through pipeline when possessions detected — v1.6.3
- [x] API merge: preserves pipeline FGA when API has no miss data — v1.6.3
- [ ] Better TOV detection (currently 5-6 per team, real likely 15-20)
- [ ] DREB inflation fix (every missed shot counts as DREB)
- [ ] Scoreboard OCR for shot outcome (made vs missed from score changes)
- [ ] Track fragmentation reduction (5775 unique track IDs for ~17 players)

## Priority 3 — Report Quality (carried forward)

- [x] Game Summary: expand to 5-7 paragraphs (currently 4 sentences)
- [x] Score by Quarter: per-quarter narrative + half-by-half analysis
- [ ] Q4 Film Analysis: Gemini video analysis when 10+ point swing
- [ ] Cross-team advanced metrics ranked by Game Score
- [ ] In-depth coaching assessment of #4 Victor Stornes (EB-85) using coaching.yaml

## Priority 4 — v1.8.0 Pipeline Improvements (future)

- [ ] Full jersey resolution for ALL tracks (not just shooters) — enables complete MIN (4h)
- [ ] Confidence scoring for shot events — training data collection (2h)
- [ ] Scoreboard OCR cross-validation for made/missed shots (8h)
- [ ] Ball-through-hoop spatial analysis for shot outcome
- [ ] Referee signal detection (supplementary made-shot confirmation)
- [ ] Player tracking continuity (reduce track fragmentation)
- [x] VLM rate limit fix: gemini-3.1-pro-preview → gemini-3-flash (25 → 1000 RPM) — v1.7.0-dx
- [x] Stage 3 progress bar with %, fps, ETA — v1.7.0-dx

## Priority 1 — Report Charts (requested 2026-03-16) — DONE

- [x] Embed shot charts in DOCX report (per team, per quarter)
- [x] Embed score flow chart in DOCX report
- [x] Green numbered markers with shooter's jersey number (already in shot_chart.py)

## Completed (move here when done)

- [x] Embed shot charts in DOCX report (per team, per quarter) — v1.6.1
- [x] Embed score flow chart in DOCX report — v1.6.1
- [x] Green numbered markers with shooter's jersey number — v1.6.0
- [x] Game Summary: expand to 5-7 paragraphs — v1.6.1
- [x] Score by Quarter: per-quarter narrative + half-by-half analysis — v1.6.2
- [x] FGA attribution: jersey resolution for all shots — v1.6.3
- [x] AST detection plumbing: end-to-end flow when possessions exist — v1.6.3
- [x] API merge: smart FGA preservation logic — v1.6.3
- [x] Data Sources section moved to Appendix — v1.6.3
- [x] Causal quarter narratives (WHY teams win/lose) — v1.6.3

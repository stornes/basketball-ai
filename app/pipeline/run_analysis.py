"""Pipeline orchestration - runs all analysis stages in sequence."""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.analytics.box_score import BoxScoreCompiler, BoxScoreProfile, GameBoxScore
from app.events.assist_detector import AssistDetector
from app.events.rebound_detector import ReboundDetector
from app.events.steal_detector import StealDetector
from app.analytics.metrics import GameMetrics
from app.analytics.score_flow import ScoreFlowGenerator
from app.analytics.shot_chart import ShotChartGenerator
from app.config.roster import Roster, load_roster
from app.events.event_types import PossessionEvent, ShotEvent
from app.events.three_point import ThreePointClassifier
from app.reporting.box_score_renderer import BoxScoreRenderer
from app.events.possession import PossessionTracker
from app.events.possession_state import BallState, PossessionStateMachine
from app.events.shot_detector import ShotDetector
from app.ingest.video_loader import VideoLoader
from app.pipeline.pipeline_config import PipelineConfig
from app.reporting.clips import ClipGenerator
from app.reporting.coach_agent import run_coaching_agent
from app.tracking.jersey_number import JerseyNumberReader, sherlock_resolve
from app.tracking.team_classifier import TeamClassifier
from app.tracking.deepsort_tracker import DeepSortTracker
from app.tracking.tracker import PlayerTracker, TrackedPlayer
from app.vision.court_mapper import CourtMapper
from app.vision.detector import PlayerBallDetector
from app.vision.scoreboard_ocr import (
    QuarterBoundaryDetector,
    quarter_boundaries_to_ranges,
    timestamp_to_quarter,
)


@dataclass
class PipelineResult:
    video_path: str
    shot_events: list[ShotEvent] = field(default_factory=list)
    possession_events: list[PossessionEvent] = field(default_factory=list)
    chart_path: str = ""
    chart_paths: list[str] = field(default_factory=list)
    stats_path: str = ""
    possessions_path: str = ""
    report_path: str = ""
    box_score_json_path: str = ""
    box_score_txt_path: str = ""
    clip_paths: list[str] = field(default_factory=list)
    timing: dict = field(default_factory=dict)


class PipelineOrchestrator:
    """Runs the full video analysis pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, video_path: str) -> PipelineResult:
        """Execute full analysis pipeline on a video."""
        result = PipelineResult(video_path=video_path)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.perf_counter()

        # Load roster if provided
        roster: Roster | None = None
        if self.config.roster_path:
            try:
                roster = load_roster(self.config.roster_path)
                print(f"  Roster loaded: {len(roster.home_players)} home, "
                      f"{len(roster.away_players)} away players")
            except Exception as e:
                print(f"  Warning: Could not load roster: {e}")

        # Stage 1: Load video and get metadata
        print("Stage 1: Loading video...")
        with VideoLoader(video_path, decode_width=self.config.decode_width) as loader:
            meta = loader.metadata()
            print(f"  Video: {meta.width}x{meta.height}, {meta.fps:.1f}fps, "
                  f"{meta.duration_sec:.1f}s, {meta.frame_count} frames")

            # Stage 2: Initialize components
            print("Stage 2: Initializing detectors...")
            detector = PlayerBallDetector(self.config)
            if self.config.tracker_type == "deepsort":
                tracker = DeepSortTracker()
                iou_tracker = None
            else:
                tracker = None
                iou_tracker = PlayerTracker()
            court_mapper = CourtMapper()
            shot_detector = ShotDetector(
                meta.height, meta.fps,
                frame_width=meta.width,
                sample_rate=self.config.frame_sample_rate,
            )
            possession_tracker = PossessionTracker(meta.fps, frame_width=meta.width)
            team_classifier = TeamClassifier() if self.config.enable_team_classification else None
            jersey_reader = JerseyNumberReader(
                vlm_backend=self.config.vlm_backend,
            )
            rebound_detector = ReboundDetector(meta.fps)
            # v5.0.0: Three-state possession state machine (gating turnover/steal)
            psm: PossessionStateMachine | None = None
            if self.config.use_possession_state_machine:
                psm = PossessionStateMachine(
                    fps=meta.fps,
                    proximity_threshold_px=int(meta.width * 0.05),
                )
            # v1.7.0: Observational ball-in-hands and pass detection
            from app.events.ball_possession import BallInHandsDetector
            from app.events.pass_detector import PassDetector
            ball_hands_detector = BallInHandsDetector(frame_width=meta.width)
            pass_detector = PassDetector(
                fps=meta.fps, frame_width=meta.width, frame_height=meta.height,
            )

            all_tracks: list[TrackedPlayer] = []
            court_calibrated = False
            frames_processed = 0
            # Store ball-player data per frame for post-processing possession pass
            ball_player_frames: list[tuple[Detection | None, list[TrackedPlayer], int]] = []

            # Stage 3: Process frames
            total_sampled = meta.frame_count // self.config.frame_sample_rate
            print(f"Stage 3: Processing frames (sample_rate={self.config.frame_sample_rate}, "
                  f"~{total_sampled} frames to process)...")
            t_detect = time.perf_counter()

            batch_frames = []
            batch_indices = []

            def _process_batch(frames, indices):
                nonlocal frames_processed
                # Only pass court_bbox if calibration was successful
                court_bbox = None
                if court_calibrated:
                    court_bbox = court_mapper.get_court_bbox(frames[0].shape)

                # Batch detection using the new detector method
                batch_dets = detector.detect_batch(frames, indices, court_bbox=court_bbox)

                for i in range(len(frames)):
                    frame = frames[i]
                    frame_idx = indices[i]
                    detections = batch_dets[i]

                    # Track players
                    if self.config.tracker_type == "deepsort":
                        players = tracker.update(detections, frame)
                    else:
                        players = iou_tracker.update(detections, frame_idx)

                    # Apply court mapping to player positions
                    if court_calibrated:
                        for p in players:
                            cx, cy = p.bbox.center
                            p.court_position = court_mapper.to_court_coords(cx, cy)

                    all_tracks.extend(players)

                    # Collect jersey colour samples for team classification
                    if team_classifier is not None:
                        for p in players:
                            team_classifier.collect_sample(p.track_id, frame, p.bbox)

                    # Collect jersey number readings via OCR
                    for p in players:
                        jersey_reader.collect_sample(p.track_id, frame, p.bbox)

                    # Detect shots (with basket detection for outcome classification)
                    ball_dets = [d for d in detections if d.class_id == 32]
                    ball = ball_dets[0] if ball_dets else None
                    basket_dets = [d for d in detections if d.class_id == 1]
                    basket = basket_dets[0] if basket_dets else None

                    # v5.0.0: Update three-state PSM for ball-state context
                    ball_pos_tuple = ball.bbox.center if ball is not None else None
                    psm_players = [
                        {"track_id": p.track_id, "team": p.team,
                         "bbox_center": p.bbox.center}
                        for p in players
                    ]
                    current_ball_state: BallState | None = None
                    if psm is not None:
                        current_ball_state = psm.update(
                            frame_idx, ball_pos_tuple, psm_players
                        )

                    shot = shot_detector.update(ball, players, frame_idx, basket_detection=basket)
                    if shot:
                        # Basket-relative court coordinates (v1.7.0)
                        # Prefer basket-relative over homography — more reliable
                        if shot.ball_x is not None and basket is not None:
                            from app.analytics.shot_chart import ShotChartGenerator
                            bcx, bcy = basket.bbox.center
                            shot.court_position = ShotChartGenerator.basket_relative_coords(
                                shot.ball_x, shot.ball_y,
                                bcx, bcy, basket.bbox.width,
                                meta.width, meta.height,
                            )
                        elif court_calibrated and ball:
                            # Fallback to homography if no basket detection
                            bx, by = ball.bbox.center
                            shot.court_position = court_mapper.to_court_coords(bx, by)
                        result.shot_events.append(shot)
                        poss = possession_tracker.end_possession_on_shot(frame_idx)
                        if poss:
                            result.possession_events.append(poss)
                        # Register missed shots for rebound detection
                        # v5.0.0: gate on ball_state so LOOSE_BALL/UNKNOWN don't
                        # spuriously trigger rebound windows
                        rebound_detector.on_missed_shot(shot, ball_state=current_ball_state)
                    else:
                        poss = possession_tracker.update(ball, players, frame_idx)
                        if poss:
                            result.possession_events.append(poss)

                    # v1.7.0: Ball-in-hands detection (observational possession)
                    bih_transition = ball_hands_detector.update(ball, players, frame_idx)
                    if bih_transition:
                        pass_evt = pass_detector.on_transition(bih_transition)
                        if pass_evt and team_classifier is not None:
                            # Label pass with team info if available
                            pass_evt.from_team = team_classifier.get_team(
                                pass_evt.from_player_track_id
                            )
                            pass_evt.to_team = team_classifier.get_team(
                                pass_evt.to_player_track_id
                            )
                    # Track ball position during pass transit
                    pass_detector.track_ball(ball, frame_idx)

                    # Store ball-player data for post-processing possession pass
                    ball_player_frames.append((ball, players, frame_idx))

                    # Check for rebounds (runs every frame during rebound window)
                    rebound_detector.update(ball, players, frame_idx)

                    frames_processed += 1
                    if frames_processed % 50 == 0 or frames_processed == total_sampled:
                        pct = frames_processed / total_sampled * 100 if total_sampled else 0
                        bar_len = 30
                        filled = int(bar_len * frames_processed / total_sampled) if total_sampled else 0
                        bar = "█" * filled + "░" * (bar_len - filled)
                        elapsed = time.perf_counter() - t_detect
                        fps = frames_processed / elapsed if elapsed > 0 else 0
                        eta = (total_sampled - frames_processed) / fps if fps > 0 else 0
                        eta_min, eta_sec = divmod(int(eta), 60)
                        print(f"\r  {bar} {pct:5.1f}% │ {frames_processed}/{total_sampled} │ "
                              f"{fps:.0f} fps │ ETA {eta_min}m{eta_sec:02d}s", end="", flush=True)

            for frame_idx, frame in loader.frames(sample_rate=self.config.frame_sample_rate):
                # Court calibration (first 30 sampled frames)
                if not court_calibrated and frames_processed < 30:
                    court_calibrated = court_mapper.calibrate(frame)
                    if court_calibrated:
                        print("  Court calibration successful")

                batch_frames.append(frame)
                batch_indices.append(frame_idx)

                if len(batch_frames) == self.config.batch_size:
                    _process_batch(batch_frames, batch_indices)
                    batch_frames = []
                    batch_indices = []

            # Process remaining frames
            if len(batch_frames) > 0:
                _process_batch(batch_frames, batch_indices)

        result.timing["detection"] = time.perf_counter() - t_detect
        print()  # newline after progress bar
        print(f"  Detection complete: {frames_processed} frames in "
              f"{result.timing['detection']:.1f}s")
        print(f"  Found {len(result.shot_events)} shots, "
              f"{len(result.possession_events)} possessions")

        team_map: dict[int, str] = {}  # populated by Stage 3.5 if team classification succeeds

        # Stage 3.5: Classify teams and label shots + possessions
        if team_classifier is not None and team_classifier.track_count >= 2:
            print("Stage 3.5: Classifying teams via jersey colour...")
            team_map = team_classifier.classify()
            home_count = sum(1 for t in team_map.values() if t == "home")
            away_count = sum(1 for t in team_map.values() if t == "away")
            print(f"  Classified {len(team_map)} tracks: {home_count} home, {away_count} away")

            # Label shots with team
            for shot in result.shot_events:
                if shot.shooter_track_id is not None:
                    shot.team = team_classifier.get_team(shot.shooter_track_id)

            # Label possessions with team (override parity heuristic)
            for poss in result.possession_events:
                team = team_classifier.get_team(poss.player_track_id)
                if team:
                    poss.team = team

            # Label tracks with team
            for t in all_tracks:
                t.team = team_classifier.get_team(t.track_id)

            # Re-label rebound events with team info and re-classify OREB/DREB
            from app.events.rebound_detector import ReboundType
            relabelled = 0
            reclassified = 0
            for reb in rebound_detector.events:
                reb_team = team_classifier.get_team(reb.rebounder_track_id)
                shooter_team = None
                if reb.shooter_track_id is not None:
                    shooter_team = team_classifier.get_team(reb.shooter_track_id)
                if reb_team:
                    reb.rebounder_team = reb_team
                    relabelled += 1
                if reb_team and shooter_team:
                    old_type = reb.rebound_type
                    reb.rebound_type = (
                        ReboundType.OFFENSIVE
                        if reb_team == shooter_team
                        else ReboundType.DEFENSIVE
                    )
                    if reb.rebound_type != old_type:
                        reclassified += 1
                if shooter_team:
                    reb.shooter_team = shooter_team
            oreb_count = sum(
                1 for r in rebound_detector.events
                if r.rebound_type == ReboundType.OFFENSIVE
            )
            dreb_count = sum(
                1 for r in rebound_detector.events
                if r.rebound_type == ReboundType.DEFENSIVE
            )
            print(f"  Rebounds relabelled: {relabelled}, reclassified: {reclassified}")
            print(f"  OREB: {oreb_count}, DREB: {dreb_count}")

        # Stage 3.5b: Post-processing possession pass with real team map
        # Re-run possession tracking now that team classification is done.
        # The initial pass used parity heuristic; this pass uses real teams.
        if team_classifier is not None and team_map:
            print("Stage 3.5b: Post-processing possession tracking with real teams...")
            post_poss_tracker = PossessionTracker(
                meta.fps, frame_width=meta.width, team_map=team_map,
            )

            if self.config.use_possession_state_machine:
                # v5.0.0: Run PSM in parallel to gate which possession events
                # are real turnovers vs same-team passes or LOOSE_BALL recoveries.
                post_psm = PossessionStateMachine(
                    fps=meta.fps,
                    proximity_threshold_px=int(meta.width * 0.05),
                )
                for ball, players, frame_idx in ball_player_frames:
                    ball_pos_t = ball.bbox.center if ball is not None else None
                    psm_pl = [
                        {"track_id": p.track_id,
                         "team": team_map.get(p.track_id, p.team),
                         "bbox_center": p.bbox.center}
                        for p in players
                    ]
                    post_psm.update(frame_idx, ball_pos_t, psm_pl)
                    post_poss_tracker.update(ball, players, frame_idx)

                # Filter possession events: a turnover is ONLY a cross-team
                # PLAYER_CONTROL → PLAYER_CONTROL transition.  LOOSE_BALL
                # recoveries and same-team changes are NOT turnovers.
                psm_events = post_psm.possession_events  # state-change log
                # Build a set of frames where we saw a cross-team flip
                # (PLAYER_CONTROL(A) → PLAYER_CONTROL(B)) from the PSM log.
                cross_team_frames: set[int] = set()
                prev_controlling_team: str | None = None
                prev_state: BallState = BallState.UNKNOWN
                for ev in psm_events:
                    if (
                        ev["to_state"] == BallState.PLAYER_CONTROL
                        and ev["from_state"] == BallState.PLAYER_CONTROL
                        and prev_controlling_team is not None
                    ):
                        # Look up the team for the new controlling player
                        new_team: str | None = None
                        for p_entry in psm_pl:  # last frame's players — best available
                            if p_entry["track_id"] == ev["controlling_player"]:
                                new_team = p_entry["team"]
                                break
                        if new_team and new_team != prev_controlling_team:
                            cross_team_frames.add(ev["frame"])
                    if ev["to_state"] == BallState.PLAYER_CONTROL:
                        # Resolve team for this controlling player
                        _team: str | None = None
                        for p_entry in psm_pl:
                            if p_entry["track_id"] == ev["controlling_player"]:
                                _team = p_entry["team"]
                                break
                        prev_controlling_team = _team
                    prev_state = ev["to_state"]

                # Relabel possession events: non-cross-team "turnovers" → "pass"
                # so the steal detector doesn't count them as steals either.
                for poss_ev in post_poss_tracker.events:
                    if poss_ev.result == "turnover" and poss_ev.end_frame not in cross_team_frames:
                        poss_ev.result = "pass"

                n_real_turnovers = sum(
                    1 for pe in post_poss_tracker.events if pe.result == "turnover"
                )
                print(f"  PSM gating: {n_real_turnovers} real turnovers "
                      f"(cross-team PLAYER_CONTROL flips) identified")
            else:
                for ball, players, frame_idx in ball_player_frames:
                    post_poss_tracker.update(ball, players, frame_idx)

            # Replace the original (broken) possession events
            result.possession_events = post_poss_tracker.events
            print(f"  Post-processing produced {len(result.possession_events)} possessions "
                  f"(was {len(possession_tracker.events)} from initial pass)")

        # Label pass events with team info (now that team classification is done)
        if team_map:
            for pe in pass_detector.events:
                if pe.from_team is None:
                    pe.from_team = team_map.get(pe.from_player_track_id)
                if pe.to_team is None:
                    pe.to_team = team_map.get(pe.to_player_track_id)

        pass_count = len(pass_detector.events)
        bih_count = len(ball_hands_detector.transitions)
        print(f"  Ball-in-hands transitions: {bih_count}, passes detected: {pass_count}")

        # Stage 3.5d: Merge fragmented tracks
        from app.tracking.track_merger import merge_tracks
        all_tracks_as_dicts = [
            {
                "track_id": t.track_id,
                "frame_idx": t.frame_idx,
                "bbox": [float(t.bbox.x1), float(t.bbox.y1), float(t.bbox.x2), float(t.bbox.y2)],
                "team": t.team,
            }
            for t in all_tracks
        ]
        merge_map = merge_tracks(all_tracks_as_dicts, meta.fps)
        if merge_map:
            merged_count = len(merge_map)
            before_ids = len({t.track_id for t in all_tracks})
            for t in all_tracks:
                t.track_id = merge_map.get(t.track_id, t.track_id)
            after_ids = len({t.track_id for t in all_tracks})
            for s in result.shot_events:
                if s.shooter_track_id is not None:
                    s.shooter_track_id = merge_map.get(int(s.shooter_track_id), int(s.shooter_track_id))
            for p in result.possession_events:
                p.player_track_id = merge_map.get(p.player_track_id, p.player_track_id)
            print(f"Stage 3.5d: Track merger: {before_ids} -> {after_ids} unique IDs "
                  f"({merged_count} fragments merged)")
        else:
            print("Stage 3.5d: Track merger: no fragments merged")

        # Stage 3.5c: Detect assists and steals from possession/shot/pass events
        assist_detector = AssistDetector(meta.fps)
        steal_detector = StealDetector(meta.fps)

        # Pass observed pass events to assist detector (preferred over proximity)
        observed_passes = pass_detector.events if pass_detector.events else None
        for shot in result.shot_events:
            assist_detector.check(shot, result.possession_events,
                                  pass_events=observed_passes)

        steal_events = steal_detector.check(result.possession_events)

        assist_count = len(assist_detector.events)
        pass_assists = sum(1 for a in assist_detector.events if a.source == "pass")
        prox_assists = assist_count - pass_assists
        steal_count = len(steal_events)
        rebound_count = len(rebound_detector.events)
        print(f"  Events: {rebound_count} rebounds, {assist_count} assists "
              f"({pass_assists} observed, {prox_assists} proximity), {steal_count} steals")

        # Stage 3.6: Resolve jersey numbers via VLM
        print("Stage 3.6: Resolving jersey numbers via VLM...")
        print(f"  Collected crops from {jersey_reader.tracks_with_crops} tracks")
        # Resolve ALL event track IDs: shooters + assisters + rebounders + stealers
        shooter_ids = {
            int(s.shooter_track_id)
            for s in result.shot_events
            if s.shooter_track_id is not None
        }
        assister_ids = {
            int(a.assister_track_id)
            for a in assist_detector.events
        }
        rebounder_ids = {
            int(r.rebounder_track_id)
            for r in rebound_detector.events
        }
        stealer_ids = {
            int(s.stealer_track_id)
            for s in steal_events
        }
        all_event_ids = shooter_ids | assister_ids | rebounder_ids | stealer_ids
        print(f"  Resolving {len(all_event_ids)} event tracks "
              f"({len(shooter_ids)} shooters, {len(assister_ids)} assisters, "
              f"{len(rebounder_ids)} rebounders, {len(stealer_ids)} stealers)...")
        jersey_map = jersey_reader.resolve(track_ids=all_event_ids)
        print(f"  VLM: {jersey_reader.total_readings} readings from "
              f"{jersey_reader.tracks_with_readings} tracks → "
              f"{len(jersey_map)} resolved jersey numbers")
        for shot in result.shot_events:
            if shot.shooter_track_id is not None:
                shot.jersey_number = jersey_map.get(shot.shooter_track_id)

        # Save player descriptions
        descriptions = jersey_reader.player_descriptions
        if descriptions:
            desc_path = str(output_dir / "player_descriptions.json")
            import json as _json
            desc_data = {
                str(tid): {
                    "track_id": d.track_id,
                    "jersey_number": d.jersey_number,
                    "team_color": d.team_color,
                    "description": d.description,
                }
                for tid, d in descriptions.items()
            }
            with open(desc_path, "w") as f:
                _json.dump(desc_data, f, indent=2)
            print(f"  Saved {len(descriptions)} player descriptions")

        # Stage 3.6b: Sherlock deductive pass for ALL unresolved event tracks
        # Build a combined list of unresolved tracks from shots + assists + rebounds + steals
        # Sherlock expects objects with .shooter_track_id and .team attributes
        unresolved_shots = [
            s for s in result.shot_events
            if s.shooter_track_id is not None
            and s.jersey_number is None
        ]
        # Add assist/rebound/steal tracks as pseudo-shot objects for Sherlock
        from types import SimpleNamespace
        unresolved_event_tracks = list(unresolved_shots)
        for a in assist_detector.events:
            if int(a.assister_track_id) not in jersey_map:
                unresolved_event_tracks.append(SimpleNamespace(
                    shooter_track_id=a.assister_track_id,
                    team=a.assister_team,
                ))
        for r in rebound_detector.events:
            if int(r.rebounder_track_id) not in jersey_map:
                unresolved_event_tracks.append(SimpleNamespace(
                    shooter_track_id=r.rebounder_track_id,
                    team=r.rebounder_team,
                ))
        for s in steal_events:
            if int(s.stealer_track_id) not in jersey_map:
                unresolved_event_tracks.append(SimpleNamespace(
                    shooter_track_id=s.stealer_track_id,
                    team=s.stealer_team,
                ))
        if roster and unresolved_event_tracks:
            # Iterative Sherlock: each pass builds on previous resolutions.
            # New knowledge from pass N narrows the elimination space for pass N+1.
            MAX_SHERLOCK_PASSES = 3
            total_sherlock_resolved = 0
            remaining_tracks = list(unresolved_event_tracks)

            roster_home_dict = {"name": roster.home_team_name, "players": [
                {"number": p.number, "name": p.name} for p in roster.home_players
            ]}
            roster_away_dict = {"name": roster.away_team_name, "players": [
                {"number": p.number, "name": p.name} for p in roster.away_players
            ]}

            for sherlock_pass in range(1, MAX_SHERLOCK_PASSES + 1):
                if not remaining_tracks:
                    break
                print(f"Stage 3.6b: Sherlock pass {sherlock_pass}/{MAX_SHERLOCK_PASSES} "
                      f"on {len(remaining_tracks)} unresolved tracks...")
                sherlock_map, sherlock_desc = sherlock_resolve(
                    video_path=video_path,
                    all_tracks=all_tracks,
                    fps=meta.fps,
                    roster_home=roster_home_dict,
                    roster_away=roster_away_dict,
                    descriptions=descriptions,
                    jersey_map=jersey_map,
                    unresolved_shots=remaining_tracks,
                    vlm_backend=self.config.vlm_backend,
                )

                if not sherlock_map:
                    print(f"  Pass {sherlock_pass}: no new resolutions, stopping.")
                    break

                # Merge results into main maps
                jersey_map.update(sherlock_map)
                total_sherlock_resolved += len(sherlock_map)
                if sherlock_desc:
                    for tid, d in sherlock_desc.items():
                        descriptions[tid] = d

                # Update shot jersey numbers
                for shot in result.shot_events:
                    if shot.shooter_track_id is not None and shot.jersey_number is None:
                        shot.jersey_number = sherlock_map.get(int(shot.shooter_track_id))

                # Remove resolved tracks for next pass
                resolved_tids = set(sherlock_map.keys())
                remaining_tracks = [
                    t for t in remaining_tracks
                    if int(t.shooter_track_id) not in resolved_tids
                ]
                print(f"  Pass {sherlock_pass}: resolved {len(sherlock_map)}, "
                      f"{len(remaining_tracks)} still unresolved.")

            print(f"  Sherlock total: {total_sherlock_resolved} tracks resolved "
                  f"across {sherlock_pass} pass(es).")
            # Update descriptions file with all accumulated knowledge
            if total_sherlock_resolved > 0:
                desc_data = {
                    str(tid): {
                        "track_id": d.track_id,
                        "jersey_number": d.jersey_number,
                        "team_color": d.team_color,
                        "description": d.description,
                    }
                    for tid, d in descriptions.items()
                }
                with open(str(output_dir / "player_descriptions.json"), "w") as f:
                    json.dump(desc_data, f, indent=2)

        # Report FGA attribution rate
        if result.shot_events:
            attributed = sum(1 for s in result.shot_events if s.jersey_number is not None)
            total = len(result.shot_events)
            made_attr = sum(1 for s in result.shot_events if s.jersey_number is not None and s.outcome.value == "made")
            missed_attr = sum(1 for s in result.shot_events if s.jersey_number is not None and s.outcome.value == "missed")
            made_total = sum(1 for s in result.shot_events if s.outcome.value == "made")
            missed_total = sum(1 for s in result.shot_events if s.outcome.value == "missed")
            print(f"  FGA attribution: {attributed}/{total} shots have jersey numbers ({attributed/total*100:.0f}%)")
            print(f"    Made: {made_attr}/{made_total}, Missed: {missed_attr}/{missed_total}")

        # Stage 3.7: Detect quarter boundaries from scoreboard
        quarter_ranges = None
        game_start = self.config.game_start_sec
        if game_start > 0:
            print(f"Stage 3.7: Detecting quarter boundaries (game start={game_start:.0f}s)...")
            qbd = QuarterBoundaryDetector(sample_interval_sec=15.0)
            boundaries = qbd.detect(video_path, game_start_sec=game_start)
            if boundaries:
                quarter_ranges = quarter_boundaries_to_ranges(
                    boundaries, game_start, meta.duration_sec,
                )
                print(f"  Detected {len(boundaries)} boundaries → "
                      f"{len(quarter_ranges)} quarters")

            # Filter out warmup shots (before game start).
            # Runs unconditionally when game_start is set — even if quarter
            # boundary detection found nothing, pre-game shots are still noise.
            pre_game = [s for s in result.shot_events
                        if s.timestamp_sec < game_start]
            if pre_game:
                result.shot_events = [s for s in result.shot_events
                                      if s.timestamp_sec >= game_start]
                print(f"  Excluded {len(pre_game)} warmup shots")

        # Problem 1 fix: deduplicate bounce re-triggers AFTER warmup filter.
        # A ball bouncing off the rim creates a new arc; deduplicate_shots keeps
        # only the first shot within a 3-second window at the same location.
        before_dedup = len(result.shot_events)
        result.shot_events = ShotDetector.deduplicate_shots(result.shot_events)
        removed = before_dedup - len(result.shot_events)
        if removed:
            print(f"  Shot deduplication: removed {removed} bounce duplicates "
                  f"({before_dedup} → {len(result.shot_events)} shots)")

        # Stage 4: Compute analytics
        print("Stage 4: Computing analytics...")
        num_quarters = None
        if roster and roster.has_scores():
            num_quarters = max(len(roster.home_scores), len(roster.away_scores))

        metrics = GameMetrics(
            result.shot_events,
            result.possession_events,
            all_tracks,
            meta.fps,
            quarter_duration_sec=self.config.quarter_duration_sec,
            num_quarters=num_quarters,
            quarter_ranges=quarter_ranges,
        )

        # Save player stats (legacy format)
        stats_path = str(output_dir / "player_stats.json")
        with open(stats_path, "w") as f:
            json.dump(metrics.to_summary_dict(), f, indent=2)
        result.stats_path = stats_path

        # Stage 4b: Compile box score
        print("Stage 4b: Compiling box score...")
        profile = BoxScoreProfile(self.config.box_score_profile)
        three_pt = ThreePointClassifier()
        compiler = BoxScoreCompiler(roster=roster, three_point_classifier=three_pt)
        game_box_score = compiler.compile(
            result.shot_events,
            result.possession_events,
            all_tracks,
            meta.fps,
            profile=profile,
            rebound_events=rebound_detector.events,
            assist_events=assist_detector.events,
            steal_events=steal_detector.events,
            sample_rate=self.config.frame_sample_rate,
            jersey_map=jersey_map,
        )
        game_box_score.video_path = video_path
        game_box_score.detection_summary = {
            "frames_processed": frames_processed,
            "court_calibrated": court_calibrated,
            "total_shots": len(result.shot_events),
            "total_possessions": len(result.possession_events),
            "total_tracks": len(set(t.track_id for t in all_tracks)),
        }
        if roster:
            game_box_score.game_date = getattr(roster, 'game_date', None)

        # Stage 4b.5: Merge scorekeeper data (manual stats)
        if self.config.scorekeeper_path:
            from app.scoring.scorekeeper import ScorekeeperData, merge_scorekeeper
            print("Stage 4b.5: Merging scorekeeper data...")
            sk_data = ScorekeeperData.from_json(self.config.scorekeeper_path)
            merge_scorekeeper(game_box_score, sk_data)
            merged_stats = sum(
                1 for p in game_box_score.home.players + game_box_score.away.players
                for s in p.stat_sources.values() if s.value == "manual"
            )
            print(f"  Merged {merged_stats} manual stat entries")

        # Save box score JSON
        renderer = BoxScoreRenderer()
        box_json_path = str(output_dir / "box_score.json")
        with open(box_json_path, "w") as f:
            f.write(renderer.render_json(game_box_score))
        result.box_score_json_path = box_json_path

        # Save box score text
        box_txt_path = str(output_dir / "box_score.txt")
        with open(box_txt_path, "w") as f:
            f.write(renderer.render_text(game_box_score))
        result.box_score_txt_path = box_txt_path

        home_pts = game_box_score.home.total_pts
        away_pts = game_box_score.away.total_pts
        print(f"  Box score ({profile.value}): "
              f"{game_box_score.home.team_name} {home_pts} - "
              f"{game_box_score.away.team_name} {away_pts}")
        print(f"  Players: {len(game_box_score.home.players)} home, "
              f"{len(game_box_score.away.players)} away")

        # Save possessions
        poss_path = str(output_dir / "possessions.json")
        poss_df = metrics.possessions_dataframe()
        poss_df.to_json(poss_path, orient="records", indent=2)
        result.possessions_path = poss_path

        # Save shots
        shots_path = str(output_dir / "shots.json")
        shots_df = metrics.shots_dataframe()
        shots_df.to_json(shots_path, orient="records", indent=2)

        # Save player tracks
        tracks_path = str(output_dir / "player_tracks.json")
        track_data = []
        for t in all_tracks:
            cx, cy = t.court_position if hasattr(t, 'court_position') and t.court_position else (None, None)
            track_data.append({
                "track_id": t.track_id,
                "frame_idx": t.frame_idx,
                "bbox": [float(t.bbox.x1), float(t.bbox.y1), float(t.bbox.x2), float(t.bbox.y2)],
                "court_x": float(cx) if cx is not None else None,
                "court_y": float(cy) if cy is not None else None,
                "team": t.team,
            })
        with open(tracks_path, "w") as f:
            json.dump(track_data, f, indent=2)

        # Stage 5: Generate shot charts (all + per-team + per-quarter)
        print("Stage 5: Generating shot charts...")
        chart_gen = ShotChartGenerator()
        # Reuse shots_df from above (already computed at line 388)

        # 5a: All shots chart
        chart_path = str(output_dir / "shot_chart.png")
        chart_gen.generate(shots_df, chart_path, title="Shot Chart — All Shots")
        result.chart_path = chart_path
        result.chart_paths.append(chart_path)

        # 5b: Per-team charts
        if "team" in shots_df.columns:
            for team_label in ("home", "away"):
                team_df = shots_df[shots_df["team"] == team_label]
                if not team_df.empty:
                    team_name = roster.team_name(team_label) if roster else team_label.title()
                    path = str(output_dir / f"shot_chart_{team_label}.png")
                    chart_gen.generate(team_df, path, title=f"Shot Chart — {team_name}")
                    result.chart_paths.append(path)

            # 5c: Per-team per-quarter charts
            if "quarter" in shots_df.columns:
                max_quarter = int(shots_df["quarter"].max()) if not shots_df.empty else 0
                for team_label in ("home", "away"):
                    team_name = roster.team_name(team_label) if roster else team_label.title()
                    for q in range(1, max_quarter + 1):
                        q_df = shots_df[
                            (shots_df["team"] == team_label) &
                            (shots_df["quarter"] == q)
                        ]
                        q_label = f"OT{q - 4}" if q > 4 else f"Q{q}"
                        path = str(output_dir / f"shot_chart_{team_label}_{q_label.lower()}.png")
                        chart_gen.generate(
                            q_df, path,
                            title=f"Shot Chart — {team_name} {q_label}",
                        )
                        result.chart_paths.append(path)

        chart_count = len(result.chart_paths)
        print(f"  Generated {chart_count} shot chart(s)")

        # 5d: Score flow chart (if roster has quarter scores)
        if roster and roster.has_scores():
            print("  Generating score flow chart...")
            score_flow_gen = ScoreFlowGenerator(quarter_duration_min=self.config.quarter_duration_sec // 60)
            score_path = str(output_dir / "score_flow.png")
            score_flow_gen.generate(
                roster, score_path,
                shots_df=shots_df,
                video_duration_sec=meta.duration_sec,
            )
            result.chart_paths.append(score_path)
            print(f"  Score flow: {roster.home_team_name} {roster.home_scores[-1]} vs "
                  f"{roster.away_team_name} {roster.away_scores[-1]}")

        # Save roster to output dir if loaded
        if roster and self.config.roster_path:
            shutil.copy2(self.config.roster_path, str(output_dir / "roster.json"))

        # Stage 6: Generate highlight clips
        if self.config.enable_clips and result.shot_events:
            print("Stage 6: Generating highlight clips...")
            t_clips = time.perf_counter()
            clip_gen = ClipGenerator(video_path)
            highlights_dir = str(output_dir / "highlights")
            result.clip_paths = clip_gen.extract_shot_clips(
                result.shot_events, meta.fps, highlights_dir
            )
            result.timing["clips"] = time.perf_counter() - t_clips
            print(f"  Generated {len(result.clip_paths)} clips")
        else:
            print("Stage 6: Skipping clip generation")

        # Stage 7: Generate coaching report
        if self.config.enable_coaching_agent:
            print("Stage 7: Generating coaching report...")
            t_report = time.perf_counter()
            report_path = str(output_dir / "game_report.md")
            run_coaching_agent(
                metrics.to_summary_dict(),
                report_path,
                self.config.llm_backend,
            )
            result.report_path = report_path
            result.timing["report"] = time.perf_counter() - t_report
            print(f"  Report saved to {report_path}")
        else:
            print("Stage 7: Skipping coaching report")

        result.timing["total"] = time.perf_counter() - t_start
        print(f"\nPipeline complete in {result.timing['total']:.1f}s")
        print(f"Outputs in: {output_dir}")

        return result

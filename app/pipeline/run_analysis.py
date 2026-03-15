"""Pipeline orchestration - runs all analysis stages in sequence."""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.analytics.metrics import GameMetrics
from app.analytics.score_flow import ScoreFlowGenerator
from app.analytics.shot_chart import ShotChartGenerator
from app.config.roster import Roster, load_roster
from app.events.event_types import PossessionEvent, ShotEvent
from app.events.possession import PossessionTracker
from app.events.shot_detector import ShotDetector
from app.ingest.video_loader import VideoLoader
from app.pipeline.pipeline_config import PipelineConfig
from app.reporting.clips import ClipGenerator
from app.reporting.coach_agent import run_coaching_agent
from app.tracking.team_classifier import TeamClassifier
from app.tracking.tracker import PlayerTracker, TrackedPlayer
from app.vision.court_mapper import CourtMapper
from app.vision.detector import PlayerBallDetector


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
        with VideoLoader(video_path) as loader:
            meta = loader.metadata()
            print(f"  Video: {meta.width}x{meta.height}, {meta.fps:.1f}fps, "
                  f"{meta.duration_sec:.1f}s, {meta.frame_count} frames")

            # Stage 2: Initialize components
            print("Stage 2: Initializing detectors...")
            detector = PlayerBallDetector(self.config)
            tracker = PlayerTracker()
            court_mapper = CourtMapper()
            shot_detector = ShotDetector(meta.height, meta.fps)
            possession_tracker = PossessionTracker(meta.fps)
            team_classifier = TeamClassifier() if self.config.enable_team_classification else None

            all_tracks: list[TrackedPlayer] = []
            court_calibrated = False
            frames_processed = 0

            # Stage 3: Process frames
            print(f"Stage 3: Processing frames (sample_rate={self.config.frame_sample_rate})...")
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
                    players = tracker.update(detections, frame_idx)

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

                    # Detect shots (with basket detection for outcome classification)
                    ball_dets = [d for d in detections if d.class_id == 32]
                    ball = ball_dets[0] if ball_dets else None
                    basket_dets = [d for d in detections if d.class_id == 1]
                    basket = basket_dets[0] if basket_dets else None
                    shot = shot_detector.update(ball, players, frame_idx, basket_detection=basket)
                    if shot:
                        if court_calibrated and ball:
                            bx, by = ball.bbox.center
                            shot.court_position = court_mapper.to_court_coords(bx, by)
                        result.shot_events.append(shot)
                        poss = possession_tracker.end_possession_on_shot(frame_idx)
                        if poss:
                            result.possession_events.append(poss)
                    else:
                        poss = possession_tracker.update(ball, players, frame_idx)
                        if poss:
                            result.possession_events.append(poss)

                    frames_processed += 1
                    if frames_processed % 100 == 0:
                        print(f"  Processed {frames_processed} frames...")

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
        print(f"  Detection complete: {frames_processed} frames in "
              f"{result.timing['detection']:.1f}s")
        print(f"  Found {len(result.shot_events)} shots, "
              f"{len(result.possession_events)} possessions")

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

        # Stage 4: Compute analytics
        print("Stage 4: Computing analytics...")
        metrics = GameMetrics(
            result.shot_events,
            result.possession_events,
            all_tracks,
            meta.fps,
            quarter_duration_sec=self.config.quarter_duration_sec,
        )

        # Save player stats
        stats_path = str(output_dir / "player_stats.json")
        with open(stats_path, "w") as f:
            json.dump(metrics.to_summary_dict(), f, indent=2)
        result.stats_path = stats_path

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
        shots_df = metrics.shots_dataframe()

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

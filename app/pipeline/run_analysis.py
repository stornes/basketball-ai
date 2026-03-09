"""Pipeline orchestration - runs all analysis stages in sequence."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.analytics.metrics import GameMetrics
from app.analytics.shot_chart import ShotChartGenerator
from app.events.event_types import PossessionEvent, ShotEvent
from app.events.possession import PossessionTracker
from app.events.shot_detector import ShotDetector
from app.ingest.video_loader import VideoLoader
from app.pipeline.pipeline_config import PipelineConfig
from app.reporting.clips import ClipGenerator
from app.reporting.coach_agent import run_coaching_agent
from app.tracking.tracker import PlayerTracker, TrackedPlayer
from app.vision.court_mapper import CourtMapper
from app.vision.detector import PlayerBallDetector


@dataclass
class PipelineResult:
    video_path: str
    shot_events: list[ShotEvent] = field(default_factory=list)
    possession_events: list[PossessionEvent] = field(default_factory=list)
    chart_path: str = ""
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

            all_tracks: list[TrackedPlayer] = []
            court_calibrated = False
            frames_processed = 0

            # Stage 3: Process frames
            print(f"Stage 3: Processing frames (sample_rate={self.config.frame_sample_rate})...")
            t_detect = time.perf_counter()

            for frame_idx, frame in loader.frames(sample_rate=self.config.frame_sample_rate):
                # Court calibration (first 30 sampled frames)
                if not court_calibrated and frames_processed < 30:
                    court_calibrated = court_mapper.calibrate(frame)
                    if court_calibrated:
                        print("  Court calibration successful")

                # Detect players and ball
                detections = detector.detect_frame(frame, frame_idx)

                # Track players
                players = tracker.update(detections, frame, frame_idx)

                # Apply court mapping to player positions
                if court_calibrated:
                    for p in players:
                        cx, cy = p.bbox.center
                        p.court_position = court_mapper.to_court_coords(cx, cy)

                all_tracks.extend(players)

                # Detect shots
                ball_dets = [d for d in detections if d.class_id == 32]
                ball = ball_dets[0] if ball_dets else None
                shot = shot_detector.update(ball, players, frame_idx)
                if shot:
                    # Apply court mapping to shot position
                    if court_calibrated and ball:
                        bx, by = ball.bbox.center
                        shot.court_position = court_mapper.to_court_coords(bx, by)
                    result.shot_events.append(shot)
                    # End possession on shot
                    poss = possession_tracker.end_possession_on_shot(frame_idx)
                    if poss:
                        result.possession_events.append(poss)
                else:
                    # Track possession
                    poss = possession_tracker.update(ball, players, frame_idx)
                    if poss:
                        result.possession_events.append(poss)

                frames_processed += 1
                if frames_processed % 100 == 0:
                    print(f"  Processed {frames_processed} frames...")

        result.timing["detection"] = time.perf_counter() - t_detect
        print(f"  Detection complete: {frames_processed} frames in "
              f"{result.timing['detection']:.1f}s")
        print(f"  Found {len(result.shot_events)} shots, "
              f"{len(result.possession_events)} possessions")

        # Stage 4: Compute analytics
        print("Stage 4: Computing analytics...")
        metrics = GameMetrics(
            result.shot_events,
            result.possession_events,
            all_tracks,
            meta.fps,
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

        # Stage 5: Generate shot chart
        print("Stage 5: Generating shot chart...")
        chart_gen = ShotChartGenerator()
        shots_df = metrics.shots_dataframe()
        chart_path = str(output_dir / "shot_chart.png")
        chart_gen.generate(shots_df, chart_path)
        result.chart_path = chart_path

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

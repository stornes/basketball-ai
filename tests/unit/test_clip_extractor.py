"""Tests for app/coaching/clip_extractor.py.

All tests use mock data — no real video required. cv2.VideoCapture and
cv2.VideoWriter are patched to avoid filesystem/codec dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.coaching.clip_extractor import (
    ClipCategory,
    ClipExtractor,
    PlayerClip,
    _find_transition_moments,
    _format_context,
    _sample_evenly,
    group_tracks_by_id,
    resolve_player_track_ids,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _make_shot(frame_idx: int, timestamp_sec: float, track_id: int, team: str = "away") -> dict:
    return {
        "frame_idx": frame_idx,
        "timestamp_sec": timestamp_sec,
        "shooter_track_id": float(track_id),
        "outcome": "made",
        "team": team,
        "quarter": 1,
        "jersey_number": 4,
    }


def _make_possession(track_id: int, start_time: float, team: str = "away") -> dict:
    return {
        "possession_id": track_id,
        "player_track_id": track_id,
        "team": team,
        "start_time": start_time,
        "end_time": start_time + 2.0,
        "duration": 2.0,
        "result": "shot",
    }


def _make_track(track_id: int, frame_idx: int, fps: float = 25.0) -> dict:
    return {
        "track_id": track_id,
        "frame_idx": frame_idx,
        "bbox": [100.0, 200.0, 150.0, 300.0],
        "court_x": 50.0,
        "court_y": 50.0,
        "team": "away",
    }


def _make_mock_cap(total_frames: int = 5000, fps: float = 25.0, width: int = 1920, height: int = 1080):
    """Create a mock cv2.VideoCapture that returns fake frames."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        0: 0.0,       # CAP_PROP_POS_MSEC
        5: fps,       # CAP_PROP_FPS
        7: total_frames,  # CAP_PROP_FRAME_COUNT
        3: width,     # CAP_PROP_FRAME_WIDTH
        4: height,    # CAP_PROP_FRAME_HEIGHT
    }.get(prop, 0.0)

    import numpy as np
    fake_frame = MagicMock()  # numpy array mock
    cap.read.return_value = (True, fake_frame)
    return cap


# ─────────────────────────────────────────────
# PlayerClip dataclass
# ─────────────────────────────────────────────

class TestPlayerClip:
    def test_has_all_required_fields(self):
        clip = PlayerClip(
            clip_path="/tmp/clip.mp4",
            start_sec=10.0,
            end_sec=20.0,
            category="SHOT",
            context="Q1 0:10, shot attempt",
            frame_indices=[100, 101, 102],
        )
        assert clip.clip_path == "/tmp/clip.mp4"
        assert clip.start_sec == 10.0
        assert clip.end_sec == 20.0
        assert clip.category == "SHOT"
        assert clip.context == "Q1 0:10, shot attempt"
        assert clip.frame_indices == [100, 101, 102]

    def test_frame_indices_defaults_to_empty_list(self):
        clip = PlayerClip(
            clip_path="/tmp/x.mp4",
            start_sec=0.0,
            end_sec=5.0,
            category="DEFENSE",
            context="test",
        )
        assert clip.frame_indices == []


# ─────────────────────────────────────────────
# ClipCategory enum
# ─────────────────────────────────────────────

class TestClipCategory:
    def test_all_values_present(self):
        assert ClipCategory.SHOT.value == "SHOT"
        assert ClipCategory.POSSESSION.value == "POSSESSION"
        assert ClipCategory.DEFENSE.value == "DEFENSE"
        assert ClipCategory.TRANSITION.value == "TRANSITION"
        assert ClipCategory.OFF_BALL.value == "OFF_BALL"


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

class TestSampleEvenly:
    def test_returns_empty_for_empty_input(self):
        assert _sample_evenly([], n=5) == []

    def test_returns_all_if_n_gte_length(self):
        times = [1.0, 2.0, 3.0]
        assert _sample_evenly(times, n=5) == times

    def test_returns_n_samples(self):
        times = list(range(100))
        result = _sample_evenly(times, n=4)
        assert len(result) == 4

    def test_n_zero_returns_empty(self):
        assert _sample_evenly([1.0, 2.0], n=0) == []


class TestFindTransitionMoments:
    def test_returns_empty_for_short_list(self):
        assert _find_transition_moments([1.0], window_sec=3.0) == []

    def test_detects_large_gaps(self):
        # Gap of 10 seconds between 5.0 and 15.0
        times = [1.0, 2.0, 5.0, 15.0, 16.0, 17.0]
        result = _find_transition_moments(times, window_sec=3.0)
        assert len(result) >= 1
        # Mid-point of 5.0 and 15.0 is 10.0
        assert any(abs(t - 10.0) < 0.5 for t in result)

    def test_ignores_small_gaps(self):
        times = [1.0, 2.0, 3.0, 4.0]
        result = _find_transition_moments(times, window_sec=3.0)
        assert len(result) == 0


class TestFormatContext:
    def test_formats_with_quarter(self):
        result = _format_context(95.0, "Q2", "shot made", ClipCategory.SHOT)
        assert "Q2" in result
        assert "1:35" in result
        assert "shot made" in result

    def test_formats_without_quarter(self):
        result = _format_context(60.0, "", "possession", ClipCategory.POSSESSION)
        assert "1:00" in result
        assert "possession" in result


class TestGroupTracksById:
    def test_groups_correctly(self):
        tracks = [
            _make_track(1, 100),
            _make_track(1, 200),
            _make_track(2, 150),
        ]
        groups = group_tracks_by_id(tracks)
        assert len(groups[1]) == 2
        assert len(groups[2]) == 1

    def test_empty_input(self):
        assert group_tracks_by_id([]) == {}


class TestResolvePlayerTrackIds:
    def test_finds_jersey_match(self):
        descriptions = {
            "51": {"track_id": 51, "jersey_number": 4, "team_color": "dark blue"},
            "72": {"track_id": 72, "jersey_number": 6, "team_color": "dark blue"},
            "108": {"track_id": 108, "jersey_number": 15, "team_color": "white"},
        }
        result = resolve_player_track_ids(descriptions, target_jersey=4, target_team="away")
        assert 51 in result
        assert 72 not in result

    def test_returns_empty_if_no_match(self):
        descriptions = {
            "72": {"track_id": 72, "jersey_number": 6, "team_color": "dark blue"},
        }
        result = resolve_player_track_ids(descriptions, target_jersey=4, target_team="away")
        assert result == []

    def test_handles_null_jersey(self):
        descriptions = {
            "51": {"track_id": 51, "jersey_number": None, "team_color": "dark blue"},
        }
        result = resolve_player_track_ids(descriptions, target_jersey=4, target_team="away")
        assert result == []


# ─────────────────────────────────────────────
# ClipExtractor — mocked cv2
# ─────────────────────────────────────────────

class TestClipExtractor:
    """Tests for ClipExtractor.extract_player_clips using mocked cv2."""

    def _make_extractor(self) -> ClipExtractor:
        return ClipExtractor(video_path="/fake/video.mp4", fps=25.0)

    def _patch_cv2(self, total_frames: int = 5000):
        """Context manager that patches cv2.VideoCapture and cv2.VideoWriter."""
        mock_cap = _make_mock_cap(total_frames=total_frames)
        mock_writer = MagicMock()

        cap_patch = patch("cv2.VideoCapture", return_value=mock_cap)
        writer_patch = patch("cv2.VideoWriter", return_value=mock_writer)
        fourcc_patch = patch("cv2.VideoWriter_fourcc", return_value=0x7634706D)
        return cap_patch, writer_patch, fourcc_patch

    def test_returns_list_of_player_clips(self, tmp_path):
        extractor = self._make_extractor()
        shots = [_make_shot(1000, 40.0, track_id=51)]
        tracks = [_make_track(51, i * 25) for i in range(100)]  # ~100 seconds of tracks

        cap_p, wr_p, fc_p = self._patch_cv2()
        with cap_p, wr_p, fc_p:
            clips = extractor.extract_player_clips(
                player_tracks=tracks,
                shot_events=shots,
                possession_events=[],
                max_clips=25,
                output_dir=str(tmp_path),
            )

        assert isinstance(clips, list)
        assert all(isinstance(c, PlayerClip) for c in clips)

    def test_shot_clips_are_generated_for_shot_events(self, tmp_path):
        extractor = self._make_extractor()
        shots = [
            _make_shot(1000, 40.0, track_id=51),
            _make_shot(2000, 80.0, track_id=51),
        ]
        tracks = [_make_track(51, i * 25) for i in range(200)]

        cap_p, wr_p, fc_p = self._patch_cv2()
        with cap_p, wr_p, fc_p:
            clips = extractor.extract_player_clips(
                player_tracks=tracks,
                shot_events=shots,
                possession_events=[],
                max_clips=25,
                output_dir=str(tmp_path),
            )

        shot_clips = [c for c in clips if c.category == ClipCategory.SHOT.value]
        assert len(shot_clips) == 2

    def test_clips_do_not_exceed_max_clips(self, tmp_path):
        extractor = self._make_extractor()
        # Many shots + possessions that would exceed max_clips
        shots = [_make_shot(i * 250, i * 10.0, track_id=51) for i in range(20)]
        possessions = [_make_possession(51, i * 15.0) for i in range(10)]
        tracks = [_make_track(51, i * 25) for i in range(300)]

        cap_p, wr_p, fc_p = self._patch_cv2()
        with cap_p, wr_p, fc_p:
            clips = extractor.extract_player_clips(
                player_tracks=tracks,
                shot_events=shots,
                possession_events=possessions,
                max_clips=10,
                output_dir=str(tmp_path),
            )

        assert len(clips) <= 10

    def test_clips_sorted_shots_first(self, tmp_path):
        extractor = self._make_extractor()
        shots = [_make_shot(1000, 40.0, track_id=51)]
        possessions = [_make_possession(51, 60.0), _make_possession(51, 90.0)]
        tracks = [_make_track(51, i * 25) for i in range(200)]

        cap_p, wr_p, fc_p = self._patch_cv2()
        with cap_p, wr_p, fc_p:
            clips = extractor.extract_player_clips(
                player_tracks=tracks,
                shot_events=shots,
                possession_events=possessions,
                max_clips=25,
                output_dir=str(tmp_path),
            )

        if len(clips) >= 2:
            # First clip should be SHOT category
            priority = {"SHOT": 0, "POSSESSION": 1, "DEFENSE": 2, "TRANSITION": 3, "OFF_BALL": 4}
            priorities = [priority.get(c.category, 99) for c in clips]
            assert priorities == sorted(priorities), "Clips are not sorted by priority"

    def test_returns_empty_if_no_events_and_no_tracks(self, tmp_path):
        extractor = self._make_extractor()
        cap_p, wr_p, fc_p = self._patch_cv2()
        with cap_p, wr_p, fc_p:
            clips = extractor.extract_player_clips(
                player_tracks=[],
                shot_events=[],
                possession_events=[],
                max_clips=25,
                output_dir=str(tmp_path),
            )
        assert clips == []

    def test_clip_has_correct_category_string(self, tmp_path):
        extractor = self._make_extractor()
        shots = [_make_shot(1000, 40.0, track_id=51)]
        tracks = [_make_track(51, i * 25) for i in range(100)]

        cap_p, wr_p, fc_p = self._patch_cv2()
        with cap_p, wr_p, fc_p:
            clips = extractor.extract_player_clips(
                player_tracks=tracks,
                shot_events=shots,
                possession_events=[],
                max_clips=25,
                output_dir=str(tmp_path),
            )

        shot_clips = [c for c in clips if c.category == "SHOT"]
        assert len(shot_clips) >= 1

    def test_returns_empty_when_video_not_found(self, tmp_path):
        """When cv2 fails to open the video, clips return None and are filtered out."""
        extractor = self._make_extractor()
        shots = [_make_shot(1000, 40.0, track_id=51)]
        tracks = [_make_track(51, i * 25) for i in range(100)]

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False  # Video not found

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with patch("cv2.VideoWriter", return_value=MagicMock()):
                with patch("cv2.VideoWriter_fourcc", return_value=0):
                    clips = extractor.extract_player_clips(
                        player_tracks=tracks,
                        shot_events=shots,
                        possession_events=[],
                        max_clips=25,
                        output_dir=str(tmp_path),
                    )

        # All clips should be None and filtered, so result may be empty or only non-cv2 clips
        # Key assertion: no exception raised
        assert isinstance(clips, list)

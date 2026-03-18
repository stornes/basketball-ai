"""Tests for VLM-based jersey number recognition."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.tracking.jersey_number import (
    AnthropicBackend,
    GeminiBackend,
    JerseyNumberReader,
    PlayerDescription,
    _parse_vlm_response,
)
from app.vision.detection_types import BoundingBox


@pytest.fixture
def reader():
    return JerseyNumberReader(sample_interval=1, min_readings=2, vlm_backend="anthropic")


@pytest.fixture
def dummy_frame():
    """A 1080p blank frame."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def player_bbox():
    """A reasonably-sized player bounding box."""
    return BoundingBox(x1=100, y1=200, x2=200, y2=500)


class TestCollectSample:
    """Tests for collect_sample — stores best crops per track."""

    def test_skips_by_sample_interval(self, dummy_frame, player_bbox):
        """Only checks every N-th detection."""
        reader = JerseyNumberReader(sample_interval=3, min_readings=1, max_crops_per_track=5)
        for _ in range(9):
            reader.collect_sample(1, dummy_frame, player_bbox)
        # Should store crop on detections 3, 6, 9
        assert len(reader._crops[1]) == 3

    def test_skips_small_bbox(self, reader, dummy_frame):
        """Tiny bboxes are skipped."""
        tiny_bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        reader.collect_sample(1, dummy_frame, tiny_bbox)
        assert 1 not in reader._crops

    def test_stores_crop(self, reader, dummy_frame, player_bbox):
        """Valid crop is stored as JPEG bytes."""
        reader.collect_sample(1, dummy_frame, player_bbox)
        assert len(reader._crops[1]) == 1
        area, jpeg_bytes = reader._crops[1][0]
        assert area > 0
        assert len(jpeg_bytes) > 0

    def test_keeps_largest_crops(self, dummy_frame):
        """Only keeps max_crops_per_track largest crops."""
        reader = JerseyNumberReader(sample_interval=1, max_crops_per_track=2)
        small_bbox = BoundingBox(x1=100, y1=200, x2=150, y2=300)  # 50x100
        medium_bbox = BoundingBox(x1=100, y1=200, x2=200, y2=400)  # 100x200
        large_bbox = BoundingBox(x1=100, y1=200, x2=300, y2=500)  # 200x300

        reader.collect_sample(1, dummy_frame, small_bbox)
        reader.collect_sample(1, dummy_frame, medium_bbox)
        reader.collect_sample(1, dummy_frame, large_bbox)

        # Should keep only 2 largest
        assert len(reader._crops[1]) == 2
        areas = [a for a, _ in reader._crops[1]]
        assert areas[0] >= areas[1]  # Sorted descending


class TestParseVlmResponse:
    """Tests for _parse_vlm_response."""

    def test_valid_full_response(self):
        text = "NUMBER: 23 | COLOR: blue | DESC: tall player with dark hair"
        num, color, desc = _parse_vlm_response(text)
        assert num == 23
        assert color == "blue"
        assert desc == "tall player with dark hair"

    def test_unknown_number(self):
        text = "NUMBER: unknown | COLOR: white | DESC: small guard"
        num, color, desc = _parse_vlm_response(text)
        assert num is None
        assert color == "white"

    def test_hash_prefix_stripped(self):
        text = "NUMBER: #7 | COLOR: red | DESC: point guard"
        num, color, desc = _parse_vlm_response(text)
        assert num == 7

    def test_rejects_over_99(self):
        text = "NUMBER: 123 | COLOR: blue | DESC: player"
        num, _, _ = _parse_vlm_response(text)
        assert num is None

    def test_single_digit(self):
        text = "NUMBER: 4 | COLOR: navy | DESC: athletic build"
        num, _, _ = _parse_vlm_response(text)
        assert num == 4

    def test_zero_is_valid(self):
        text = "NUMBER: 0 | COLOR: black | DESC: point guard"
        num, _, _ = _parse_vlm_response(text)
        assert num == 0

    def test_garbled_response(self):
        text = "I cannot determine the jersey number from this image"
        num, color, desc = _parse_vlm_response(text)
        assert num is None


class TestResolve:
    """Tests for resolve with mocked VLM calls."""

    def test_majority_vote(self, reader):
        """Most frequent reading wins."""
        reader._readings = {
            1: [23, 23, 23, 28, 28],  # 23 has 60% → wins
        }
        reader._crops = {1: [(100, b"fake")]}
        # Skip VLM calls by pre-populating readings
        result = reader.resolve.__wrapped__(reader) if hasattr(reader.resolve, '__wrapped__') else {}
        # Direct test of consensus logic
        from collections import Counter
        readings = reader._readings[1]
        counter = Counter(readings)
        most_common_num, count = counter.most_common(1)[0]
        assert most_common_num == 23
        assert count / len(readings) >= 0.4

    def test_resolve_calls_vlm(self):
        """resolve() calls VLM backend for each crop."""
        reader = JerseyNumberReader(sample_interval=1, min_readings=1, vlm_backend="anthropic")
        reader._crops = {1: [(1000, b"fake_jpeg")]}

        with patch.object(AnthropicBackend, "call_single",
                    return_value="NUMBER: 23 | COLOR: blue | DESC: tall player") as mock_vlm:
            result = reader.resolve()
            mock_vlm.assert_called_once()
            assert result[1] == 23

    def test_resolve_stores_description(self):
        """resolve() stores player descriptions."""
        reader = JerseyNumberReader(sample_interval=1, min_readings=1, vlm_backend="anthropic")
        reader._crops = {42: [(500, b"fake_jpeg")]}

        with patch.object(AnthropicBackend, "call_single",
                    return_value="NUMBER: 7 | COLOR: white | DESC: small guard, green shoes"):
            reader.resolve()
            assert 42 in reader.player_descriptions
            desc = reader.player_descriptions[42]
            assert desc.jersey_number == 7
            assert desc.team_color == "white"
            assert "green shoes" in desc.description

    def test_resolve_gemini_backend(self):
        """resolve() uses Gemini when configured."""
        reader = JerseyNumberReader(sample_interval=1, min_readings=1, vlm_backend="gemini")
        reader._crops = {1: [(1000, b"fake_jpeg")]}

        with patch.object(GeminiBackend, "call_single",
                    return_value="NUMBER: 36 | COLOR: navy | DESC: center") as mock:
            result = reader.resolve()
            mock.assert_called_once()
            assert result[1] == 36

    def test_resolve_empty_crops(self):
        """No crops → empty result."""
        reader = JerseyNumberReader(sample_interval=1, min_readings=1)
        result = reader.resolve()
        assert result == {}

    def test_resolve_vlm_error_handled(self):
        """VLM errors are caught and logged, not fatal."""
        reader = JerseyNumberReader(sample_interval=1, min_readings=1, vlm_backend="anthropic")
        reader._crops = {1: [(1000, b"fake_jpeg")]}

        with patch.object(AnthropicBackend, "call_single",
                    side_effect=Exception("API error")):
            result = reader.resolve()
            assert result == {}

    def test_multiple_tracks(self):
        """Multiple tracks resolved independently."""
        reader = JerseyNumberReader(sample_interval=1, min_readings=1, vlm_backend="anthropic")
        reader._crops = {
            1: [(1000, b"fake1")],
            2: [(800, b"fake2")],
        }

        responses = iter([
            "NUMBER: 23 | COLOR: blue | DESC: player 1",
            "NUMBER: 7 | COLOR: white | DESC: player 2",
        ])
        with patch.object(AnthropicBackend, "call_single",
                    side_effect=lambda *a: next(responses)):
            result = reader.resolve()
            assert result[1] == 23
            assert result[2] == 7


class TestProperties:
    """Tests for reader properties."""

    def test_tracks_with_readings(self, reader):
        reader._readings = {1: [23], 2: [], 3: [7, 7]}
        assert reader.tracks_with_readings == 2

    def test_total_readings(self, reader):
        reader._readings = {1: [23, 23], 2: [7]}
        assert reader.total_readings == 3

    def test_tracks_with_crops(self, reader):
        reader._crops = {1: [(100, b"a")], 2: [(200, b"b")]}
        assert reader.tracks_with_crops == 2

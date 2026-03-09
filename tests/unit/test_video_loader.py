"""Tests for video loading and frame extraction."""

from app.ingest.video_loader import VideoLoader


def test_video_loads(synthetic_video_path):
    with VideoLoader(synthetic_video_path) as loader:
        meta = loader.metadata()
        assert meta.frame_count == 30
        assert meta.width == 640
        assert meta.height == 480
        assert meta.fps > 0


def test_frame_extraction(synthetic_video_path):
    frames = []
    with VideoLoader(synthetic_video_path) as loader:
        for idx, frame in loader.frames(sample_rate=1):
            frames.append((idx, frame.shape))

    assert len(frames) == 30
    assert frames[0][1] == (480, 640, 3)


def test_frame_sampling(synthetic_video_path):
    frames = []
    with VideoLoader(synthetic_video_path) as loader:
        for idx, frame in loader.frames(sample_rate=5):
            frames.append(idx)

    assert len(frames) == 6  # 30 / 5
    assert frames == [0, 5, 10, 15, 20, 25]


def test_extract_frames_to_dir(synthetic_video_path, tmp_path):
    from app.ingest.video_loader import extract_frames
    saved = extract_frames(synthetic_video_path, tmp_path / "frames", fps=10)
    assert len(saved) > 0
    assert all(p.exists() for p in saved)

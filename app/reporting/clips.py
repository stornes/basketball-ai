"""Highlight clip generation using FFmpeg."""

from pathlib import Path

from app.events.event_types import ShotEvent


class ClipGenerator:
    """Extracts video clips around shot events using ffmpeg."""

    def __init__(self, source_video: str):
        self.source = source_video
        self._ffmpeg_bin = self._find_ffmpeg()

    def generate_clip(self, start_sec: float, end_sec: float, output_path: str) -> str:
        """Extract a time-range clip from source video."""
        import ffmpeg

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        duration = end_sec - start_sec

        (
            ffmpeg
            .input(self.source, ss=start_sec, t=duration)
            .output(output_path, vcodec="libx264", acodec="copy", loglevel="error")
            .overwrite_output()
            .run(cmd=self._ffmpeg_bin, quiet=True)
        )
        return output_path

    def extract_shot_clips(
        self,
        shot_events: list[ShotEvent],
        fps: float,
        output_dir: str,
    ) -> list[str]:
        """Generate highlight clips for all shot events."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        clips = []

        for i, event in enumerate(shot_events):
            start_sec = max(0, event.clip_start_frame / fps)
            end_sec = event.clip_end_frame / fps
            filename = f"shot_{i:03d}_{event.outcome.value}.mp4"
            out_path = str(Path(output_dir) / filename)

            try:
                self.generate_clip(start_sec, end_sec, out_path)
                clips.append(out_path)
            except Exception as e:
                print(f"Warning: Failed to generate clip {filename}: {e}")

        return clips

    @staticmethod
    def _find_ffmpeg() -> str:
        """Find ffmpeg binary, preferring imageio-ffmpeg's bundled version."""
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        # Fallback to system ffmpeg
        import shutil
        path = shutil.which("ffmpeg")
        if path:
            return path
        raise RuntimeError(
            "ffmpeg not found. Install via: pip install imageio-ffmpeg"
        )

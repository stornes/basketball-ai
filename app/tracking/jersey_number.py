"""Jersey number recognition via Vision Language Model on player crops.

Collects the best (largest) crop per track_id during frame processing,
then batch-sends crops to a VLM (Gemini Flash, Claude, or GPT-4o) for
jersey number reading and player description.

VLM backends use the Strategy pattern — each backend is a VLMBackend
subclass registered in VLM_REGISTRY. Add new backends by subclassing
and calling VLMBackend.register().
"""

from __future__ import annotations

import abc
import base64
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from app.vision.detection_types import BoundingBox


@dataclass
class PlayerDescription:
    """VLM-generated description of a player."""

    track_id: int
    jersey_number: int | None
    team_color: str | None
    description: str | None


# ── VLM Strategy ────────────────────────────────────────────────────


class VLMBackend(abc.ABC):
    """Strategy interface for Vision Language Model backends."""

    @abc.abstractmethod
    def call_single(self, image_bytes: bytes, prompt: str) -> str:
        """Send one image + prompt, return text response."""

    @abc.abstractmethod
    def call_multi(self, images: list[bytes], prompt: str) -> str:
        """Send multiple images + prompt, return text response."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable backend name for logging."""

    _registry: dict[str, type[VLMBackend]] = {}

    @classmethod
    def register(cls, key: str, backend_cls: type[VLMBackend]) -> None:
        cls._registry[key] = backend_cls

    @classmethod
    def get(cls, key: str) -> VLMBackend:
        """Instantiate a registered backend by key. Falls back to gemini."""
        backend_cls = cls._registry.get(key, cls._registry.get("gemini"))
        if backend_cls is None:
            raise ValueError(f"No VLM backend registered for '{key}' and no fallback available")
        return backend_cls()


class GeminiBackend(VLMBackend):
    @property
    def name(self) -> str:
        return "gemini"

    def call_single(self, image_bytes: bytes, prompt: str) -> str:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )
        return response.text.strip()

    def call_multi(self, images: list[bytes], prompt: str) -> str:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        client = genai.Client(api_key=api_key)
        contents = [
            types.Part.from_bytes(data=img, mime_type="image/jpeg")
            for img in images
        ]
        contents.append(prompt)
        response = client.models.generate_content(
            model="gemini-3-flash",
            contents=contents,
        )
        return response.text.strip()


class AnthropicBackend(VLMBackend):
    @property
    def name(self) -> str:
        return "anthropic"

    def call_single(self, image_bytes: bytes, prompt: str) -> str:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        client = anthropic.Anthropic(api_key=api_key)
        b64 = base64.b64encode(image_bytes).decode()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.content[0].text.strip()

    def call_multi(self, images: list[bytes], prompt: str) -> str:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        client = anthropic.Anthropic(api_key=api_key)
        content = []
        for img in images:
            b64 = base64.b64encode(img).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })
        content.append({"type": "text", "text": prompt})
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text.strip()


class OpenAIBackend(VLMBackend):
    @property
    def name(self) -> str:
        return "openai"

    def call_single(self, image_bytes: bytes, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        b64 = base64.b64encode(image_bytes).decode()
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()

    def call_multi(self, images: list[bytes], prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        content = []
        for img in images:
            b64 = base64.b64encode(img).decode()
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low",
                },
            })
        content.append({"type": "text", "text": prompt})
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=500,
            messages=[{"role": "user", "content": content}],
        )
        return response.choices[0].message.content.strip()


class GrokBackend(VLMBackend):
    @property
    def name(self) -> str:
        return "grok"

    def _client(self):
        from openai import OpenAI

        api_key = os.environ.get("XAI_API_KEY", "")
        return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    def call_single(self, image_bytes: bytes, prompt: str) -> str:
        client = self._client()
        b64 = base64.b64encode(image_bytes).decode()
        response = client.chat.completions.create(
            model="grok-2-vision-latest",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()

    def call_multi(self, images: list[bytes], prompt: str) -> str:
        client = self._client()
        content = []
        for img in images:
            b64 = base64.b64encode(img).decode()
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low",
                },
            })
        content.append({"type": "text", "text": prompt})
        response = client.chat.completions.create(
            model="grok-2-vision-latest",
            max_tokens=500,
            messages=[{"role": "user", "content": content}],
        )
        return response.choices[0].message.content.strip()


VLMBackend.register("gemini", GeminiBackend)
VLMBackend.register("anthropic", AnthropicBackend)
VLMBackend.register("openai", OpenAIBackend)
VLMBackend.register("grok", GrokBackend)


# ── Backward-compat aliases (used by rerun_vlm.py imports) ──────────

def _call_gemini(image_bytes: bytes, prompt: str) -> str:
    return GeminiBackend().call_single(image_bytes, prompt)

def _call_anthropic(image_bytes: bytes, prompt: str) -> str:
    return AnthropicBackend().call_single(image_bytes, prompt)

def _call_openai(image_bytes: bytes, prompt: str) -> str:
    return OpenAIBackend().call_single(image_bytes, prompt)

def _call_grok(image_bytes: bytes, prompt: str) -> str:
    return GrokBackend().call_single(image_bytes, prompt)

def _call_gemini_multi_image(images: list[bytes], prompt: str) -> str:
    return GeminiBackend().call_multi(images, prompt)

def _call_anthropic_multi_image(images: list[bytes], prompt: str) -> str:
    return AnthropicBackend().call_multi(images, prompt)

def _call_openai_multi_image(images: list[bytes], prompt: str) -> str:
    return OpenAIBackend().call_multi(images, prompt)

def _call_grok_multi_image(images: list[bytes], prompt: str) -> str:
    return GrokBackend().call_multi(images, prompt)


_VLM_PROMPT = (
    "What jersey number is this basketball player wearing? "
    "Reply in EXACTLY this format on one line:\n"
    "NUMBER: [number or unknown] | COLOR: [jersey color] | "
    "DESC: [brief physical description including build, hair, shoes, "
    "any distinguishing features like undershirt, ankle bands, etc]\n"
    "If you cannot determine the number, use 'unknown'."
)


def _parse_vlm_response(text: str) -> tuple[int | None, str | None, str | None]:
    """Parse VLM response into (jersey_number, team_color, description).

    Expected format: NUMBER: 23 | COLOR: blue | DESC: tall player with...
    Returns (None, None, None) if unparseable.
    """
    parts = {}
    for segment in text.split("|"):
        segment = segment.strip()
        if ":" in segment:
            key, _, val = segment.partition(":")
            parts[key.strip().upper()] = val.strip()

    # Parse jersey number
    jersey_number = None
    num_str = parts.get("NUMBER", "unknown").strip().lstrip("#")
    if num_str.isdigit():
        num = int(num_str)
        if 0 <= num <= 99:
            jersey_number = num

    team_color = parts.get("COLOR")
    description = parts.get("DESC")

    return jersey_number, team_color, description


_SHERLOCK_PROMPT_TEMPLATE = """You are Sherlock Holmes identifying a basketball player. Study ALL {n_images} images carefully — they show the SAME player from different angles/moments.

TEAM: {team_name} ({team_label})
JERSEY COLOR: {jersey_color}
ROSTER (only these players exist on this team):
{roster_list}

KNOWN PLAYER FEATURES (already identified from other tracks):
{known_features}

YOUR TASK:
1. OBSERVE: List every visible detail across all images — skin tone, hair, build, shoes (color, style), accessories (headband, wristband, ankle bands, knee pads, undershirt, compression sleeves), any partial numbers or text visible.
2. ELIMINATE: For each roster player, state whether they can be ruled out and why.
3. DEDUCE: "When you have eliminated the impossible, whatever remains, however improbable, must be the truth."

Reply in EXACTLY this format:
OBSERVATIONS: [detailed physical inventory]
ELIMINATED: [#X because..., #Y because...]
DEDUCTION: NUMBER: [number or unknown] | CONFIDENCE: [high/medium/low] | REASONING: [one line]"""


def _build_sherlock_prompt(
    team_label: str,
    team_name: str,
    jersey_color: str,
    roster_players: list[dict],
    known_features: dict[int, str],
    n_images: int,
) -> str:
    """Build a Sherlock-style deductive prompt with roster context."""
    roster_lines = []
    for p in roster_players:
        num = p["number"]
        name = p["name"]
        features = known_features.get(num, "not yet identified")
        roster_lines.append(f"  #{num} {name} — {features}")
    roster_list = "\n".join(roster_lines)

    known_lines = []
    for num, feat in sorted(known_features.items()):
        known_lines.append(f"  #{num}: {feat}")
    known_str = "\n".join(known_lines) if known_lines else "  (none identified yet)"

    return _SHERLOCK_PROMPT_TEMPLATE.format(
        n_images=n_images,
        team_name=team_name,
        team_label=team_label,
        jersey_color=jersey_color,
        roster_list=roster_list,
        known_features=known_str,
    )


def _parse_sherlock_response(text: str) -> tuple[int | None, str | None, str]:
    """Parse Sherlock response into (jersey_number, confidence, full_text).

    Looks for DEDUCTION: NUMBER: X | CONFIDENCE: Y | REASONING: Z
    Returns (None, None, text) if unparseable or low confidence.
    """
    import re

    # Find the DEDUCTION line (may span multiple lines with markdown)
    deduction_text = ""
    for line in text.split("\n"):
        if "DEDUCTION" in line.upper() and "NUMBER" in line.upper():
            deduction_text = line
            break

    if not deduction_text:
        # Try regex across full text
        deduction_text = text

    # Extract NUMBER using regex — handles "NUMBER: 4", "NUMBER: #4", etc.
    jersey_number = None
    num_match = re.search(r'NUMBER:\s*#?(\d{1,2})\b', deduction_text, re.IGNORECASE)
    if not num_match:
        # Fallback: search entire text for the pattern
        num_match = re.search(r'NUMBER:\s*#?(\d{1,2})\b', text, re.IGNORECASE)
    if num_match:
        num = int(num_match.group(1))
        if 0 <= num <= 99:
            jersey_number = num

    # Extract CONFIDENCE
    confidence = None
    conf_match = re.search(r'CONFIDENCE:\s*(high|medium|low)', text, re.IGNORECASE)
    if conf_match:
        confidence = conf_match.group(1).lower()

    return jersey_number, confidence, text


def _extract_crop_from_video(cap, frame_idx: int, bbox: list[float]) -> bytes | None:
    """Extract a JPEG crop from video at given frame and bbox."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    fh, fw = frame.shape[:2]
    x1 = max(0, min(int(bbox[0]), fw - 1))
    y1 = max(0, min(int(bbox[1]), fh - 1))
    x2 = max(x1 + 1, min(int(bbox[2]), fw))
    y2 = max(y1 + 1, min(int(bbox[3]), fh))
    if (y2 - y1) < 40 or (x2 - x1) < 25:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def _find_temporal_crops(
    all_tracks: list[dict],
    track_id: int,
    fps: float,
    window_sec: float = 10.0,
    max_crops: int = 5,
) -> list[dict]:
    """Find diverse track appearances within a temporal window.

    Returns up to max_crops track entries, spread across the window,
    prioritizing the largest bboxes.
    """
    appearances = [t for t in all_tracks if t["track_id"] == track_id]
    if not appearances:
        return []
    for a in appearances:
        bbox = a["bbox"]
        a["_area"] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    appearances.sort(key=lambda x: x["_area"], reverse=True)

    window_frames = int(window_sec * fps)
    selected = []
    used_frame_ranges = []
    for a in appearances:
        frame_idx = a["frame_idx"]
        too_close = any(
            abs(frame_idx - used) < window_frames // max_crops
            for used in used_frame_ranges
        )
        if too_close and len(selected) > 0:
            continue
        selected.append(a)
        used_frame_ranges.append(frame_idx)
        if len(selected) >= max_crops:
            break
    return selected


def sherlock_resolve(
    video_path: str,
    all_tracks: list,
    fps: float,
    roster_home: dict,
    roster_away: dict,
    descriptions: dict[int, PlayerDescription],
    jersey_map: dict[int, int],
    unresolved_shots: list,
    vlm_backend: str = "gemini",
) -> tuple[dict[int, int], dict[int, PlayerDescription]]:
    """Sherlock deductive pass: resolve unresolved tracks using multi-frame reasoning.

    Args:
        video_path: Path to source video.
        all_tracks: List of TrackedPlayer or dicts with track_id, frame_idx, bbox.
        fps: Video FPS.
        roster_home: {"name": str, "players": [{"number": int, "name": str}, ...]}.
        roster_away: Same format for away team.
        descriptions: Existing PlayerDescription map from standard VLM pass.
        jersey_map: Already-resolved {track_id: jersey_number}.
        unresolved_shots: List of shot dicts/objects with shooter_track_id and team.
        vlm_backend: "gemini" or "anthropic".

    Returns:
        (new_jersey_map, new_descriptions) — only newly resolved entries.
    """
    import cv2
    import time as _time

    # Build known features from already-resolved descriptions
    home_known: dict[int, str] = {}
    away_known: dict[int, str] = {}
    for tid, d in descriptions.items():
        num = d.jersey_number
        desc_text = d.description or ""
        color = (d.team_color or "").lower()
        if num is None or not desc_text:
            continue
        if "white" in color:
            home_known[num] = desc_text
        elif "blue" in color or "dark" in color:
            away_known[num] = desc_text

    # Build track list as dicts for _find_temporal_crops compatibility
    track_dicts = []
    for t in all_tracks:
        if hasattr(t, "bbox"):
            bbox = [float(t.bbox.x1), float(t.bbox.y1), float(t.bbox.x2), float(t.bbox.y2)]
        else:
            bbox = t["bbox"]
        track_dicts.append({
            "track_id": t.track_id if hasattr(t, "track_id") else t["track_id"],
            "frame_idx": t.frame_idx if hasattr(t, "frame_idx") else t["frame_idx"],
            "bbox": bbox,
        })

    # Identify unresolved track IDs and their teams
    unresolved_ids: dict[int, str] = {}
    for s in unresolved_shots:
        tid = int(s.shooter_track_id if hasattr(s, "shooter_track_id") else s["shooter_track_id"])
        team = s.team if hasattr(s, "team") else s.get("team", "unknown")
        if tid not in jersey_map:
            unresolved_ids[tid] = team or "unknown"

    # Filter non-players from existing descriptions
    non_player_kws = [
        "no basketball player", "not visible", "court floor",
        "whistle", "referee", "grey t-shirt", "diadora",
        "grey beard", "black trousers", "black pants",
    ]
    non_player_ids = set()
    for tid, d in descriptions.items():
        desc_text = (d.description or "").lower()
        color = (d.team_color or "").lower()
        if any(kw in desc_text for kw in non_player_kws):
            non_player_ids.add(tid)
        if color in ("grey", "dark grey", "dark", "unknown") and d.jersey_number is None:
            if any(kw in desc_text for kw in ["t-shirt", "trousers", "watch", "wristband"]):
                non_player_ids.add(tid)

    filtered_ids = {tid: team for tid, team in unresolved_ids.items() if tid not in non_player_ids}
    print(f"  Sherlock: {len(unresolved_ids)} unresolved, "
          f"{len(non_player_ids)} non-player, {len(filtered_ids)} for deduction")

    if not filtered_ids:
        return {}, {}

    vlm = VLMBackend.get(vlm_backend)

    cap = cv2.VideoCapture(video_path)
    new_map: dict[int, int] = {}
    new_desc: dict[int, PlayerDescription] = {}

    for tid, team in sorted(filtered_ids.items()):
        # Find temporal crops
        temporal = _find_temporal_crops(track_dicts, tid, fps, window_sec=10.0, max_crops=5)
        if not temporal:
            continue

        crops = []
        for t in temporal:
            crop_bytes = _extract_crop_from_video(cap, t["frame_idx"], t["bbox"])
            if crop_bytes:
                crops.append(crop_bytes)
        if not crops:
            continue

        # Determine roster context
        if team == "home":
            roster_players = roster_home["players"]
            team_name = roster_home["name"]
            jersey_color = "white with blue accents"
            known = home_known
        elif team == "away":
            roster_players = roster_away["players"]
            team_name = roster_away["name"]
            jersey_color = "dark blue"
            known = away_known
        else:
            # Unknown team — try standard VLM
            try:
                response = vlm.call_single(crops[0], _VLM_PROMPT)
                num, color, desc = _parse_vlm_response(response)
                if num is not None:
                    new_map[tid] = num
                new_desc[tid] = PlayerDescription(tid, num, color, desc)
            except Exception:
                pass
            _time.sleep(0.2)
            continue

        prompt = _build_sherlock_prompt(
            team_label=team, team_name=team_name,
            jersey_color=jersey_color, roster_players=roster_players,
            known_features=known, n_images=len(crops),
        )

        try:
            response = vlm.call_multi(crops, prompt) if len(crops) > 1 else vlm.call_single(crops[0], prompt)
            num, confidence, full_text = _parse_sherlock_response(response)

            if num is not None and confidence in ("high", "medium"):
                new_map[tid] = num
                # Update known features for subsequent deductions
                obs_text = ""
                for line in full_text.split("\n"):
                    if "OBSERVATION" in line.upper():
                        obs_text = line.split(":", 1)[-1].strip() if ":" in line else ""
                        break
                if team == "home":
                    home_known[num] = obs_text or f"track {tid}"
                else:
                    away_known[num] = obs_text or f"track {tid}"

            new_desc[tid] = PlayerDescription(
                track_id=tid,
                jersey_number=num if confidence in ("high", "medium") else None,
                team_color=jersey_color,
                description=full_text[:500],
            )
        except Exception as e:
            print(f"    Sherlock error track {tid}: {e}")

        _time.sleep(0.3)

    cap.release()

    # Mark non-player tracks
    for tid in non_player_ids:
        if tid in unresolved_ids and tid in descriptions:
            d = descriptions[tid]
            new_desc[tid] = PlayerDescription(
                track_id=tid, jersey_number=None,
                team_color=d.team_color,
                description=f"[NON-PLAYER] {d.description or 'unknown'}",
            )

    print(f"  Sherlock resolved {len(new_map)} additional tracks: "
          f"{sorted(new_map.values())}")
    return new_map, new_desc


class JerseyNumberReader:
    """Read jersey numbers from player crops using a Vision Language Model.

    Usage:
        reader = JerseyNumberReader()
        # During frame processing — stores best crop per track:
        reader.collect_sample(track_id, frame, bbox)
        # After all frames — calls VLM API:
        jersey_map = reader.resolve()
        # Query:
        jersey_map.get(track_id)  # int | None
        # Descriptions:
        reader.player_descriptions  # dict[int, PlayerDescription]
    """

    def __init__(
        self,
        sample_interval: int = 30,
        min_readings: int = 2,
        vlm_backend: str = "anthropic",
        max_crops_per_track: int = 1,
    ):
        """
        Args:
            sample_interval: Check every N-th detection per track for a
                better crop. Only the best (largest) crops are kept.
            min_readings: Minimum VLM readings needed for consensus.
            vlm_backend: "gemini" or "anthropic".
            max_crops_per_track: Keep N best crops per track for consensus.
        """
        self._sample_interval = sample_interval
        self._min_readings = min_readings
        self._vlm_backend = vlm_backend
        self._max_crops = max_crops_per_track
        self._detection_counts: dict[int, int] = defaultdict(int)
        # Store (area, jpeg_bytes) tuples per track, sorted by area desc
        self._crops: dict[int, list[tuple[int, bytes]]] = defaultdict(list)
        self._readings: dict[int, list[int]] = defaultdict(list)
        self._descriptions: dict[int, PlayerDescription] = {}

    def collect_sample(
        self, track_id: int, frame: np.ndarray, bbox: BoundingBox
    ) -> None:
        """Store player crop if it's one of the best for this track.

        Keeps only the largest crops per track — bigger = better for VLM.
        No API calls here; crops are sent to VLM in resolve().
        """
        self._detection_counts[track_id] += 1
        if self._detection_counts[track_id] % self._sample_interval != 0:
            return
        self._collect_crop(track_id, frame, bbox)

    def force_collect_sample(
        self, track_id: int, frame: np.ndarray, bbox: BoundingBox
    ) -> None:
        """Store a crop unconditionally, bypassing the sample interval."""
        self._collect_crop(track_id, frame, bbox)

    def _collect_crop(
        self, track_id: int, frame: np.ndarray, bbox: BoundingBox
    ) -> None:
        """Internal: extract and store a crop for a track."""

        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        h = y2 - y1
        w = x2 - x1
        if h < 40 or w < 25:
            return

        # Clamp to frame bounds
        fh, fw = frame.shape[:2]
        y1 = max(0, min(y1, fh - 1))
        y2 = max(y1 + 1, min(y2, fh))
        x1 = max(0, min(x1, fw - 1))
        x2 = max(x1 + 1, min(x2, fw))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        area = (x2 - x1) * (y2 - y1)
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        jpeg_bytes = buf.tobytes()

        crops = self._crops[track_id]
        crops.append((area, jpeg_bytes))
        # Keep only the N largest crops
        crops.sort(key=lambda x: x[0], reverse=True)
        self._crops[track_id] = crops[: self._max_crops]

    def resolve(
        self, track_ids: set[int] | None = None
    ) -> dict[int, int]:
        """Send stored crops to VLM and resolve jersey numbers.

        Args:
            track_ids: If provided, only resolve these specific track IDs.
                      Otherwise resolves all tracks with crops.

        Returns mapping of track_id → jersey_number for tracks where
        the VLM confidently identified a number.
        """
        if not self._crops:
            return {}

        # Filter to requested tracks
        if track_ids is not None:
            crops_to_resolve = {
                tid: crops
                for tid, crops in self._crops.items()
                if tid in track_ids
            }
        else:
            crops_to_resolve = dict(self._crops)

        if not crops_to_resolve:
            return {}

        vlm = VLMBackend.get(self._vlm_backend)
        total_crops = sum(len(c) for c in crops_to_resolve.values())
        print(f"    VLM: Sending {total_crops} crops from "
              f"{len(crops_to_resolve)} tracks to {self._vlm_backend} "
              f"(5 concurrent workers)...")

        # Build flat list of (track_id, jpeg_bytes) tasks
        tasks: list[tuple[int, bytes]] = []
        for track_id, crops in crops_to_resolve.items():
            for _area, jpeg_bytes in crops:
                tasks.append((track_id, jpeg_bytes))

        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Rate limiter: min 0.05s between API calls across all workers
        rate_lock = threading.Lock()
        last_call_time = [0.0]  # mutable container for closure

        def _call_vlm(task: tuple[int, bytes]) -> tuple[int, str | None]:
            """Call VLM for one crop, return (track_id, response_text)."""
            tid, jpeg = task
            # Enforce minimum interval between API calls
            with rate_lock:
                now = time.monotonic()
                wait = 0.05 - (now - last_call_time[0])
                if wait > 0:
                    time.sleep(wait)
                last_call_time[0] = time.monotonic()
            try:
                response = vlm.call_single(jpeg, _VLM_PROMPT)
                return (tid, response)
            except Exception as e:
                print(f"    VLM error for track {tid}: {e}")
                return (tid, None)

        processed = 0
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(_call_vlm, t): t for t in tasks}
            for future in as_completed(futures):
                track_id, response = future.result()
                processed += 1

                if response is not None:
                    jersey_num, team_color, desc = _parse_vlm_response(response)

                    if jersey_num is not None:
                        self._readings[track_id].append(jersey_num)

                    # Store the best description we get
                    if track_id not in self._descriptions or (
                        jersey_num is not None
                        and self._descriptions[track_id].jersey_number is None
                    ):
                        self._descriptions[track_id] = PlayerDescription(
                            track_id=track_id,
                            jersey_number=jersey_num,
                            team_color=team_color,
                            description=desc,
                        )

                if processed % 50 == 0:
                    print(f"    VLM: {processed}/{total_crops} crops processed...")

        # Consensus vote
        result: dict[int, int] = {}
        for track_id, readings in self._readings.items():
            if len(readings) < self._min_readings:
                # Single reading is fine if we only have 1 crop
                if len(readings) == 1 and len(self._crops.get(track_id, [])) == 1:
                    result[track_id] = readings[0]
                continue
            counter = Counter(readings)
            most_common_num, most_common_count = counter.most_common(1)[0]
            if most_common_count / len(readings) >= 0.4:
                result[track_id] = most_common_num

        # Update descriptions with consensus jersey numbers
        for track_id, jersey_num in result.items():
            if track_id in self._descriptions:
                self._descriptions[track_id].jersey_number = jersey_num

        return result

    @property
    def player_descriptions(self) -> dict[int, PlayerDescription]:
        """Player descriptions from VLM, keyed by track_id."""
        return dict(self._descriptions)

    @property
    def tracks_with_readings(self) -> int:
        """Number of tracks that have at least one jersey reading."""
        return sum(1 for r in self._readings.values() if r)

    @property
    def total_readings(self) -> int:
        """Total number of jersey readings collected."""
        return sum(len(r) for r in self._readings.values())

    @property
    def tracks_with_crops(self) -> int:
        """Number of tracks that have stored crops."""
        return len(self._crops)

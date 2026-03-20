"""Visual coaching analyst — sends player clips to Gemini Vision for analysis.

Uses google.genai directly (NOT langchain) because langchain doesn't support
video file upload via the Files API.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.coaching.clip_extractor import PlayerClip


@dataclass
class ClipAnalysis:
    """Analysis of a single player clip by the VLM."""

    clip_path: str
    category: str
    context: str
    situation: str          # What was happening in the game
    what_player_did: str    # Observed behaviour
    what_should_have_done: str  # Coaching correction (empty if player did well)
    coaching_cue: str       # One-sentence actionable cue
    grade: str              # A / B / C / D
    error: Optional[str] = None  # Set if VLM call failed


@dataclass
class CoachingReport:
    """Complete 7-section coaching report for one player."""

    player_name: str
    jersey_number: int
    team: str
    game_date: str

    # 7 sections from v8.0.0 manifest
    player_profile: str = ""
    offensive_game: str = ""
    playmaking: str = ""
    off_ball_movement: str = ""
    defensive_game: str = ""
    intangibles: str = ""
    development_plan: str = ""

    # Raw clip analyses for reference in report
    clip_analyses: list[ClipAnalysis] = field(default_factory=list)


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

_CLIP_ANALYSIS_PROMPT_TEMPLATE = """You are a world class basketball coach who develops young players.

You are watching game film of {player_name} (#{jersey}, {team_color} jersey) in a U16 game.

This clip is categorised as: {category}
Context: {context}

Observe what he does, understand why, and coach him on what to do differently.
Be specific about what you see in the video. Speak to him directly as his coach.

Respond in this exact format:
SITUATION: [1-2 sentences — what was happening in the game when this clip starts]
WHAT YOU DID: [2-3 sentences — specific observed behaviour, movements, decisions]
WHAT TO DO INSTEAD: [2-3 sentences — coaching correction, or "Good job" if correct]
COACHING CUE: [One sentence — the single most important thing to remember next time]
GRADE: [A / B / C / D — A=excellent, B=good, C=needs work, D=poor decision]"""

_SYNTHESIS_PROMPT_TEMPLATE = """You are a world class basketball coach writing a development report for {player_name} (#{jersey}, {team_color} jersey) after watching his full game film from {game_date}.

Below are your observations from {n_clips} clips of his play during the game. Each clip is tagged with a category and grade.

{clip_analyses_text}

Write a comprehensive coaching report with exactly these 7 sections. Write directly to the player — use "you" not "he". Be specific and reference actual moments from the clips. Be encouraging but honest. UK English.

## 1. Player Profile
[Physical profile observed, position, role in team, minutes quality]

## 2. Offensive Game
[Primary offensive role, scoring areas, shot selection quality, ball handling under pressure]

## 3. Playmaking and Decision-Making
[Court vision, pass timing, pick-and-roll reads, turnover patterns]

## 4. Off-Ball Movement
[Cutting quality and timing, screening, spacing, transition running]

## 5. Defensive Game
[On-ball stance, lateral movement, closeout technique, help defense, rebounding]

## 6. Intangibles and Effort
[Body language after misses, communication, hustle plays, leadership moments]

## 7. Development Priorities
[Top 3 strengths to keep. Top 3 growth areas — be specific. One drill per growth area. What to focus on in the next game.]"""


class VisualCoachingAnalyst:
    """Analyses player clips using Gemini Vision and synthesises a coaching report."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        import google.genai as genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment.")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def analyse_clip(self, clip: PlayerClip, player_context: dict) -> ClipAnalysis:
        """Send one clip to Gemini Vision for coaching analysis.

        player_context keys:
          - player_name (str)
          - jersey (int)
          - team_color (str, e.g. "dark blue")
        """
        player_name = player_context.get("player_name", "the player")
        jersey = player_context.get("jersey", "?")
        team_color = player_context.get("team_color", "coloured")

        prompt = _CLIP_ANALYSIS_PROMPT_TEMPLATE.format(
            player_name=player_name,
            jersey=jersey,
            team_color=team_color,
            category=clip.category,
            context=clip.context,
        )

        try:
            # Upload clip to Gemini Files API
            video_file = self.client.files.upload(file=clip.clip_path)

            # Wait for processing if needed
            _wait_for_file_active(self.client, video_file)

            # Generate analysis
            response = self.client.models.generate_content(
                model=self.model,
                contents=[video_file, prompt],
            )

            text = response.text or ""
            return _parse_clip_analysis(clip, text)

        except Exception as exc:
            return ClipAnalysis(
                clip_path=clip.clip_path,
                category=clip.category,
                context=clip.context,
                situation="[Analysis failed]",
                what_player_did="[Analysis failed]",
                what_should_have_done="",
                coaching_cue="",
                grade="?",
                error=str(exc),
            )

    def analyse_clips(
        self,
        clips: list[PlayerClip],
        player_context: dict,
        delay_sec: float = 1.0,
    ) -> list[ClipAnalysis]:
        """Analyse all clips sequentially, with a small delay to avoid rate limits."""
        analyses: list[ClipAnalysis] = []
        for i, clip in enumerate(clips):
            print(f"  Analysing clip {i + 1}/{len(clips)}: {clip.category} — {clip.context}")
            analysis = self.analyse_clip(clip, player_context)
            analyses.append(analysis)
            if delay_sec > 0 and i < len(clips) - 1:
                time.sleep(delay_sec)
        return analyses

    def synthesise_report(
        self,
        analyses: list[ClipAnalysis],
        player_context: dict,
    ) -> CoachingReport:
        """Feed all clip text analyses to Gemini to generate the 7-section coaching report.

        Uses text-only (not video) for the synthesis step — all clip analyses
        are concatenated and sent as a single prompt to the 1M-context model.
        """
        player_name = player_context.get("player_name", "the player")
        jersey = player_context.get("jersey", "?")
        team_color = player_context.get("team_color", "coloured")
        game_date = player_context.get("game_date", "")

        clip_analyses_text = _format_analyses_for_synthesis(analyses)

        prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(
            player_name=player_name,
            jersey=jersey,
            team_color=team_color,
            game_date=game_date,
            n_clips=len(analyses),
            clip_analyses_text=clip_analyses_text,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
            )
            report_text = response.text or ""
        except Exception as exc:
            report_text = f"[Report synthesis failed: {exc}]"

        return _parse_coaching_report(
            report_text=report_text,
            analyses=analyses,
            player_name=player_name,
            jersey=int(jersey) if str(jersey).isdigit() else 0,
            team=player_context.get("team", ""),
            game_date=game_date,
        )


# ─────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────

def _wait_for_file_active(client, video_file, max_wait_sec: int = 60) -> None:
    """Poll until the uploaded file is in ACTIVE state."""
    import time as _time
    deadline = _time.time() + max_wait_sec
    while _time.time() < deadline:
        state = getattr(video_file, "state", None)
        if state is None:
            break
        state_name = state.name if hasattr(state, "name") else str(state)
        if state_name == "ACTIVE":
            break
        if state_name == "FAILED":
            raise RuntimeError(f"File upload failed: {video_file.name}")
        _time.sleep(2)
        # Re-fetch the file object to get updated state
        video_file = client.files.get(name=video_file.name)


def _parse_clip_analysis(clip: PlayerClip, text: str) -> ClipAnalysis:
    """Parse the structured VLM response into a ClipAnalysis."""
    lines: dict[str, str] = {}
    current_key = ""
    for line in text.splitlines():
        line = line.strip()
        for key in ("SITUATION", "WHAT YOU DID", "WHAT TO DO INSTEAD", "COACHING CUE", "GRADE"):
            if line.startswith(f"{key}:"):
                current_key = key
                lines[key] = line[len(key) + 1:].strip()
                break
        else:
            if current_key and line:
                lines[current_key] = lines.get(current_key, "") + " " + line

    grade = lines.get("GRADE", "?").strip().upper()
    if grade not in ("A", "B", "C", "D"):
        grade = "?"

    return ClipAnalysis(
        clip_path=clip.clip_path,
        category=clip.category,
        context=clip.context,
        situation=lines.get("SITUATION", ""),
        what_player_did=lines.get("WHAT YOU DID", ""),
        what_should_have_done=lines.get("WHAT TO DO INSTEAD", ""),
        coaching_cue=lines.get("COACHING CUE", ""),
        grade=grade,
    )


def _format_analyses_for_synthesis(analyses: list[ClipAnalysis]) -> str:
    """Format all clip analyses as readable text for the synthesis prompt."""
    parts: list[str] = []
    for i, a in enumerate(analyses, 1):
        status = f"[FAILED: {a.error}]" if a.error else f"Grade: {a.grade}"
        parts.append(
            f"--- Clip {i} ({a.category}, {a.context}) | {status} ---\n"
            f"Situation: {a.situation}\n"
            f"What he did: {a.what_player_did}\n"
            f"Coaching note: {a.what_should_have_done}\n"
            f"Cue: {a.coaching_cue}"
        )
    return "\n\n".join(parts)


def _parse_coaching_report(
    report_text: str,
    analyses: list[ClipAnalysis],
    player_name: str,
    jersey: int,
    team: str,
    game_date: str,
) -> CoachingReport:
    """Parse the 7-section synthesis response into a CoachingReport."""
    sections: dict[str, list[str]] = {}
    current_section = ""
    section_map = {
        "## 1.": "player_profile",
        "## 2.": "offensive_game",
        "## 3.": "playmaking",
        "## 4.": "off_ball_movement",
        "## 5.": "defensive_game",
        "## 6.": "intangibles",
        "## 7.": "development_plan",
    }

    for line in report_text.splitlines():
        stripped = line.strip()
        matched = False
        for header, field_name in section_map.items():
            if stripped.startswith(header) or stripped.lower().startswith(header.lower()):
                current_section = field_name
                sections[current_section] = []
                matched = True
                break
        if not matched and current_section:
            sections.setdefault(current_section, []).append(line)

    def _join(key: str) -> str:
        return "\n".join(sections.get(key, [])).strip()

    return CoachingReport(
        player_name=player_name,
        jersey_number=jersey,
        team=team,
        game_date=game_date,
        player_profile=_join("player_profile"),
        offensive_game=_join("offensive_game"),
        playmaking=_join("playmaking"),
        off_ball_movement=_join("off_ball_movement"),
        defensive_game=_join("defensive_game"),
        intangibles=_join("intangibles"),
        development_plan=_join("development_plan"),
        clip_analyses=analyses,
    )

"""Three-state possession model: PLAYER_CONTROL, LOOSE_BALL, FLIGHT.

Replaces nothing — runs in parallel with the binary PossessionTracker when
PipelineConfig.use_possession_state_machine is True. Pure logic, no model calls.

State machine rules
-------------------
PLAYER_CONTROL: Ball is within proximity_threshold_px of one player AND that
    player has a clear proximity advantage (closest player is < 1/LOOSE_BALL_RATIO
    times the distance to the second-closest player).

FLIGHT: Ball velocity between consecutive frames exceeds FLIGHT_SPEED_THRESHOLD_PX.
    Takes precedence over proximity — a fast-moving ball is in flight regardless
    of where players are standing.

LOOSE_BALL: Ball is near multiple players without a clear proximity winner.
    Also the fallback state when ball is near players but not clearly in control.

UNKNOWN: Ball position unavailable for more than STATE_DECAY_FRAMES consecutive
    frames, or no prior state established yet.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from app.events.spatial import court_distance


class BallState(Enum):
    PLAYER_CONTROL = "player_control"
    LOOSE_BALL = "loose_ball"
    FLIGHT = "flight"
    UNKNOWN = "unknown"


# Pixels-per-frame velocity above which the ball is considered in flight.
# Calibrated to hard passes (~450 px/s at 30fps ≈ 15 px/frame) vs
# a dribble bounce (~3-5 px/frame lateral movement). Deliberately conservative.
FLIGHT_SPEED_THRESHOLD_PX: float = 15.0

# If the second-closest player's distance is less than this multiple of the
# closest player's distance, neither player has clear control → LOOSE_BALL.
# E.g. ratio=2.0: closest must be at least 2x closer than 2nd to win control.
LOOSE_BALL_RATIO: float = 2.0

# Frames with no ball detection before state decays to UNKNOWN.
STATE_DECAY_FRAMES: int = 10


class PossessionStateMachine:
    """Three-state possession state machine.

    Usage::

        psm = PossessionStateMachine(fps=30.0)
        state = psm.update(frame_idx, ball_pos=(x, y), players=[...])

    Args:
        fps: Frames per second of the video (used for future time calculations).
        proximity_threshold_px: Distance (pixels) within which a player is
            considered near the ball.
    """

    def __init__(
        self,
        fps: float,
        proximity_threshold_px: float = 80.0,
        homography: np.ndarray | None = None,
        proximity_threshold_ft: float | None = None,
    ) -> None:
        """Initialise the state machine.

        Args:
            fps: Frames per second of the video.
            proximity_threshold_px: Pixel-space distance within which a player
                is considered near the ball. Used when no homography is provided
                or as a fallback if projection fails.
            homography: Optional 3x3 perspective transform from CourtMapper.H.
                When provided, proximity is evaluated in court feet instead of
                pixels.
            proximity_threshold_ft: Court-plane distance (feet) within which a
                player is considered near the ball. Only meaningful when a
                homography is supplied. Defaults to None (pixel threshold used).
        """
        self.fps = fps
        self.proximity_threshold_px = proximity_threshold_px
        self.homography = homography
        self.proximity_threshold_ft = proximity_threshold_ft

        self.state: BallState = BallState.UNKNOWN
        self.controlling_player: int | None = None
        self.controlling_team: str | None = None
        self.state_start_frame: int = 0

        self._prev_ball_pos: tuple[float, float] | None = None
        self._frames_without_ball: int = 0
        self._events: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        ball_pos: tuple[float, float] | None,
        players: list[dict],
    ) -> BallState:
        """Update state for the current frame.

        Args:
            frame_idx: Current frame index.
            ball_pos: (x, y) centre of the ball, or None if not detected.
            players: Each dict must contain:
                - ``track_id`` (int)
                - ``team`` (str or None)
                - ``bbox_center`` (tuple[float, float]) — (x, y) player centre

        Returns:
            The current BallState after processing this frame.
        """
        if ball_pos is None:
            return self._handle_missing_ball(frame_idx)

        self._frames_without_ball = 0

        velocity = self._compute_velocity(ball_pos)
        self._prev_ball_pos = ball_pos

        new_state, candidate_player = self._classify(ball_pos, velocity, players)
        self._transition(new_state, frame_idx, players, candidate_player)
        return self.state

    @property
    def possession_events(self) -> list[dict]:
        """State-change events captured so far.

        Each event is a dict with keys:
            frame, from_state (BallState), to_state (BallState),
            controlling_player (int | None).
        """
        return list(self._events)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_missing_ball(self, frame_idx: int) -> BallState:
        """Maintain last state for STATE_DECAY_FRAMES, then go UNKNOWN."""
        self._frames_without_ball += 1
        self._prev_ball_pos = None
        if self._frames_without_ball > STATE_DECAY_FRAMES:
            self._transition(BallState.UNKNOWN, frame_idx, [])
        return self.state

    def _compute_velocity(
        self, ball_pos: tuple[float, float]
    ) -> float | None:
        """Return pixel distance moved since the previous frame, or None."""
        if self._prev_ball_pos is None:
            return None
        dx = ball_pos[0] - self._prev_ball_pos[0]
        dy = ball_pos[1] - self._prev_ball_pos[1]
        return math.hypot(dx, dy)

    def _classify(
        self,
        ball_pos: tuple[float, float],
        velocity: float | None,
        players: list[dict],
    ) -> tuple[BallState, int | None]:
        """Classify ball state from position, velocity, and player proximity.

        Returns:
            (state, candidate_track_id) where candidate_track_id is the
            closest player's track_id when state is PLAYER_CONTROL, else None.
            This avoids a second distance scan in _transition.
        """
        # FLIGHT takes priority: fast ball is always in flight
        if velocity is not None and velocity >= FLIGHT_SPEED_THRESHOLD_PX:
            return BallState.FLIGHT, None

        # Determine active threshold and whether to use court projection
        use_court = (
            self.homography is not None
            and self.proximity_threshold_ft is not None
        )
        active_threshold = (
            self.proximity_threshold_ft
            if use_court
            else self.proximity_threshold_px
        )

        # Find players within proximity threshold: (dist, track_id)
        nearby: list[tuple[float, int]] = []
        for p in players:
            cx, cy = p["bbox_center"]
            dist = court_distance(
                ball_pos, (cx, cy),
                homography=self.homography if use_court else None,
            )
            if dist <= active_threshold:
                nearby.append((dist, p["track_id"]))

        if not nearby:
            # Ball visible but no player near it — preserve FLIGHT if already
            # in flight, otherwise LOOSE_BALL
            if self.state == BallState.FLIGHT:
                return BallState.FLIGHT, None
            return BallState.LOOSE_BALL, None

        nearby.sort(key=lambda t: t[0])
        dist_closest, closest_id = nearby[0]

        if len(nearby) == 1:
            return BallState.PLAYER_CONTROL, closest_id

        # Two or more players nearby: check proximity advantage
        dist_second = nearby[1][0]

        # Avoid division by zero (players stacked on each other)
        if dist_closest < 1.0:
            return BallState.PLAYER_CONTROL, closest_id

        if dist_second / dist_closest >= LOOSE_BALL_RATIO:
            return BallState.PLAYER_CONTROL, closest_id

        return BallState.LOOSE_BALL, None

    def _transition(
        self,
        new_state: BallState,
        frame_idx: int,
        players: list[dict],
        candidate_player: int | None = None,
    ) -> None:
        """Apply state transition, updating controlling player.

        Args:
            new_state: The classified state for this frame.
            frame_idx: Current frame index.
            players: Player dicts (used to look up team for candidate_player).
            candidate_player: track_id of the closest player when new_state is
                PLAYER_CONTROL. Already computed by _classify — avoids a second
                distance scan.
        """
        if new_state == self.state:
            # No state change — refresh controlling player identity in case
            # the same player moved or a different player is now closer.
            if new_state == BallState.PLAYER_CONTROL and candidate_player is not None:
                self._apply_controlling_player(candidate_player, players)
            return

        from_state = self.state
        self.state = new_state
        self.state_start_frame = frame_idx

        if new_state == BallState.PLAYER_CONTROL and candidate_player is not None:
            self._apply_controlling_player(candidate_player, players)
        else:
            self.controlling_player = None
            self.controlling_team = None

        self._events.append(
            {
                "frame": frame_idx,
                "from_state": from_state,
                "to_state": new_state,
                "controlling_player": self.controlling_player,
            }
        )

    def _apply_controlling_player(
        self,
        track_id: int,
        players: list[dict],
    ) -> None:
        """Set controlling_player and look up their team from the players list."""
        self.controlling_player = track_id
        # Look up team for this track_id — O(n) but n is always small
        for p in players:
            if p["track_id"] == track_id:
                self.controlling_team = p.get("team")
                return
        self.controlling_team = None

"""
exercise.py — Exercise state machine for structured rep-based rehabilitation.

Manages rep counting, hold timing, timeout prompts, and finger isolation
(freezing motors so only the selected finger moves).

State flow per rep:
    WAITING -> ASSISTING -> HOLDING -> RETURNING -> PAUSE -> WAITING (or COMPLETED)
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Callable


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Exercise:
    """Definition of a single exercise in a programme."""
    name: str
    target_intent: int          # 0=close, 1=open
    finger: str                 # "all"/"thumb"/"index"/"middle"/"ring"/"pinky"
    reps: int
    hold_duration: float        # seconds to hold at end of movement
    rest_between_reps: float    # seconds of rest between reps


@dataclass
class MotorCommand:
    """Command to send to the Teensy."""
    action: str      # "open" / "close" / "rest"
    finger: str      # "all"/"thumb"/"index"/"middle"/"ring"/"pinky"
    strength: float  # 0.0-1.0 from assist curve


class ExerciseState(Enum):
    WAITING = auto()
    ASSISTING = auto()
    HOLDING = auto()
    RETURNING = auto()
    PAUSE = auto()
    COMPLETED = auto()


class Event(Enum):
    EFFORT_DETECTED = auto()
    REP_COMPLETED = auto()
    EXERCISE_COMPLETED = auto()
    TIMEOUT_WARNING = auto()
    TIMEOUT_PROMPT = auto()


# ── Serial protocol for finger isolation ─────────────────────────────────────

FINGER_SERIAL_CODES = {
    "all": "A", "thumb": "T", "index": "I",
    "middle": "M", "ring": "R", "pinky": "P",
}

INTENT_TO_ACTION = {0: "close", 1: "open", 2: "rest"}
ACTION_TO_SERIAL = {"close": "c", "open": "o", "rest": "r"}

# ── Timing constants ─────────────────────────────────────────────────────────

MOTOR_TRAVEL_TIME = 0.5     # seconds for motor to complete movement
TIMEOUT_WARNING_S = 10.0    # seconds of no effort before warning
TIMEOUT_PROMPT_S = 30.0     # seconds of no effort before skip prompt


# ── ExerciseRunner — runs one exercise through the rep cycle ─────────────────

class ExerciseRunner:
    """Runs a single Exercise through its rep cycle."""

    def __init__(self, exercise: Exercise, assist_strength_fn: Callable[[], float]):
        self.exercise = exercise
        self._get_assist_strength = assist_strength_fn

        self.state = ExerciseState.WAITING
        self.reps_completed = 0
        self.skipped = False

        self._state_entered_at = time.perf_counter()
        self._waiting_since = time.perf_counter()
        self._warning_sent = False
        self._prompt_sent = False

    @property
    def state_elapsed(self) -> float:
        return time.perf_counter() - self._state_entered_at

    @property
    def waiting_elapsed(self) -> float:
        return time.perf_counter() - self._waiting_since

    @property
    def is_completed(self) -> bool:
        return self.state == ExerciseState.COMPLETED

    def _enter_state(self, new_state: ExerciseState):
        self.state = new_state
        self._state_entered_at = time.perf_counter()

    def skip(self):
        self.skipped = True
        self._enter_state(ExerciseState.COMPLETED)

    def update(self, intent: int, confidence: float) -> List[Event]:
        """Called each prediction cycle. Returns list of events produced."""
        events = []
        ex = self.exercise

        if self.state == ExerciseState.WAITING:
            # Only the target intent triggers a transition
            if intent == ex.target_intent and confidence > 0:
                events.append(Event.EFFORT_DETECTED)
                self._enter_state(ExerciseState.ASSISTING)
                self._warning_sent = False
                self._prompt_sent = False
            else:
                # Timeout logic
                elapsed = self.waiting_elapsed
                if elapsed >= TIMEOUT_PROMPT_S and not self._prompt_sent:
                    events.append(Event.TIMEOUT_PROMPT)
                    self._prompt_sent = True
                elif elapsed >= TIMEOUT_WARNING_S and not self._warning_sent:
                    events.append(Event.TIMEOUT_WARNING)
                    self._warning_sent = True

        elif self.state == ExerciseState.ASSISTING:
            # Motor travel — runs to completion regardless of EMG
            if self.state_elapsed >= MOTOR_TRAVEL_TIME:
                self._enter_state(ExerciseState.HOLDING)

        elif self.state == ExerciseState.HOLDING:
            if self.state_elapsed >= ex.hold_duration:
                self._enter_state(ExerciseState.RETURNING)

        elif self.state == ExerciseState.RETURNING:
            if self.state_elapsed >= MOTOR_TRAVEL_TIME:
                self.reps_completed += 1
                events.append(Event.REP_COMPLETED)
                if self.reps_completed >= ex.reps:
                    events.append(Event.EXERCISE_COMPLETED)
                    self._enter_state(ExerciseState.COMPLETED)
                else:
                    self._enter_state(ExerciseState.PAUSE)

        elif self.state == ExerciseState.PAUSE:
            if self.state_elapsed >= ex.rest_between_reps:
                self._enter_state(ExerciseState.WAITING)
                self._waiting_since = time.perf_counter()
                self._warning_sent = False
                self._prompt_sent = False

        return events

    def get_motor_command(self) -> MotorCommand:
        """Return the current motor command based on state."""
        ex = self.exercise
        action_name = INTENT_TO_ACTION[ex.target_intent]

        if self.state in (ExerciseState.ASSISTING, ExerciseState.HOLDING):
            return MotorCommand(
                action=action_name,
                finger=ex.finger,
                strength=self._get_assist_strength(),
            )
        elif self.state == ExerciseState.RETURNING:
            return MotorCommand(action="rest", finger=ex.finger, strength=1.0)
        else:
            # WAITING, PAUSE, COMPLETED — motors at rest
            return MotorCommand(action="rest", finger=ex.finger, strength=0.0)


# ── SessionRunner — manages a programme of exercises ─────────────────────────

@dataclass
class ExerciseResult:
    """Post-exercise result for summary."""
    name: str
    finger: str
    reps_target: int
    reps_completed: int
    skipped: bool


class SessionRunner:
    """Manages a programme of exercises, running them sequentially."""

    def __init__(self, exercises: List[Exercise],
                 assist_strength_fn: Callable[[], float]):
        self.exercises = exercises
        self._assist_strength_fn = assist_strength_fn

        self.current_index = 0
        self.results: List[ExerciseResult] = []
        self.started_at = time.perf_counter()

        self._current_runner: Optional[ExerciseRunner] = None
        self._current_finger: Optional[str] = None
        self._completed = False

        if exercises:
            self._start_exercise(0)

    @property
    def is_completed(self) -> bool:
        return self._completed

    @property
    def current_exercise(self) -> Optional[Exercise]:
        if self.current_index < len(self.exercises):
            return self.exercises[self.current_index]
        return None

    @property
    def current_runner(self) -> Optional[ExerciseRunner]:
        return self._current_runner

    @property
    def session_duration(self) -> float:
        return time.perf_counter() - self.started_at

    def _start_exercise(self, index: int):
        self.current_index = index
        ex = self.exercises[index]
        self._current_runner = ExerciseRunner(ex, self._assist_strength_fn)

    def _record_result(self, runner: ExerciseRunner):
        ex = runner.exercise
        self.results.append(ExerciseResult(
            name=ex.name,
            finger=ex.finger,
            reps_target=ex.reps,
            reps_completed=runner.reps_completed,
            skipped=runner.skipped,
        ))

    def _advance(self):
        """Move to the next exercise or complete the session."""
        next_idx = self.current_index + 1
        if next_idx < len(self.exercises):
            self._start_exercise(next_idx)
        else:
            self._current_runner = None
            self._completed = True

    def skip_exercise(self) -> List[Event]:
        """Skip the current exercise (carer override)."""
        if self._current_runner and not self._current_runner.is_completed:
            self._current_runner.skip()
            self._record_result(self._current_runner)
            self._advance()
            return [Event.EXERCISE_COMPLETED]
        return []

    def stop(self):
        """Emergency stop — record current state and mark completed."""
        if self._current_runner and not self._current_runner.is_completed:
            self._record_result(self._current_runner)
        self._current_runner = None
        self._completed = True

    def update(self, intent: int, confidence: float) -> List[Event]:
        """Called each prediction cycle. Returns events."""
        if self._completed or self._current_runner is None:
            return []

        events = self._current_runner.update(intent, confidence)

        if self._current_runner.is_completed:
            self._record_result(self._current_runner)
            self._advance()

        return events

    def get_motor_command(self) -> MotorCommand:
        """Get current motor command from the active exercise runner."""
        if self._current_runner:
            return self._current_runner.get_motor_command()
        return MotorCommand(action="rest", finger="all", strength=0.0)

    def finger_changed(self) -> Optional[str]:
        """Return finger code if it changed since last call, else None."""
        cmd = self.get_motor_command()
        if cmd.finger != self._current_finger:
            self._current_finger = cmd.finger
            return cmd.finger
        return None

    def get_summary(self) -> List[ExerciseResult]:
        return list(self.results)


# ── Default programme ────────────────────────────────────────────────────────

def default_programme() -> List[Exercise]:
    """Standard rehabilitation session programme."""
    return [
        Exercise(
            name="Whole hand open/close (warm-up)",
            target_intent=0,  # close
            finger="all",
            reps=10,
            hold_duration=2.0,
            rest_between_reps=2.0,
        ),
        Exercise(
            name="Index finger extension",
            target_intent=1,  # open
            finger="index",
            reps=8,
            hold_duration=2.0,
            rest_between_reps=2.0,
        ),
        Exercise(
            name="Middle finger extension",
            target_intent=1,  # open
            finger="middle",
            reps=8,
            hold_duration=2.0,
            rest_between_reps=2.0,
        ),
        Exercise(
            name="Thumb opposition",
            target_intent=0,  # close
            finger="thumb",
            reps=6,
            hold_duration=2.0,
            rest_between_reps=2.5,
        ),
        Exercise(
            name="Whole hand open/close (cool-down)",
            target_intent=0,  # close
            finger="all",
            reps=10,
            hold_duration=2.0,
            rest_between_reps=2.0,
        ),
    ]

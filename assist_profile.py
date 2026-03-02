"""
assist_profile.py — 5-level assist-as-needed rehabilitation profiles.

Stroke patients produce weak, noisy EMG signals. These profiles bias the
system toward detecting movement so the patient gets positive feedback for
any effort, then progressively reduce assistance as the patient recovers.

Level 1 (Max Assist)   — most forgiving; detects the faintest intent
Level 5 (Minimal)      — equivalent to the unmodified system
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssistProfile:
    """Tuning parameters for one assist level."""

    level: int                  # 1-5
    label: str                  # human-readable name

    # --- Movement bias (post-classifier) ---
    confidence_floor: float     # minimum confidence to act
    movement_bias: float        # probability added to close/open, removed from rest

    # --- Assist curve (confidence -> motor strength) ---
    assist_exponent: float      # strength = confidence ^ exponent  (<1 = boost)

    # --- Stability filter ---
    stability_required: int     # consecutive same-predictions before changing output

    # --- Anti-jitter (motor safety) ---
    proba_ema_alpha: float      # EMA smoothing on predict_proba (0=no smooth, 0.9=heavy)
    hysteresis_enter: float     # confidence needed to leave rest → movement
    hysteresis_exit: float      # confidence needed to leave movement → rest
    cooldown_ms: float          # minimum ms between state transitions

    # --- Adaptive gain normalization (pre-feature-extraction) ---
    adaptive_gain: bool         # whether gain normalization is active
    gain_floor: float           # minimum per-channel gain multiplier
    gain_ceiling: float         # maximum per-channel gain multiplier
    ema_decay: float            # EMA decay for amplitude tracking (higher = slower)


# ── Preset profiles ──────────────────────────────────────────────────────────

ASSIST_PROFILES = {
    1: AssistProfile(
        level=1,
        label="Max Assist",
        confidence_floor=0.15,
        movement_bias=0.25,
        assist_exponent=0.3,
        stability_required=1,
        proba_ema_alpha=0.7,        # heavy smoothing — slow, forgiving
        hysteresis_enter=0.30,      # low bar to trigger movement
        hysteresis_exit=0.20,       # very easy to stay in movement
        cooldown_ms=800,            # long cooldown — no rapid switching
        adaptive_gain=True,
        gain_floor=1.0,
        gain_ceiling=50.0,
        ema_decay=0.998,
    ),
    2: AssistProfile(
        level=2,
        label="High Assist",
        confidence_floor=0.25,
        movement_bias=0.15,
        assist_exponent=0.5,
        stability_required=1,
        proba_ema_alpha=0.6,
        hysteresis_enter=0.40,
        hysteresis_exit=0.25,
        cooldown_ms=600,
        adaptive_gain=True,
        gain_floor=1.0,
        gain_ceiling=30.0,
        ema_decay=0.999,
    ),
    3: AssistProfile(
        level=3,
        label="Moderate Assist",
        confidence_floor=0.35,
        movement_bias=0.08,
        assist_exponent=0.7,
        stability_required=2,
        proba_ema_alpha=0.5,
        hysteresis_enter=0.50,
        hysteresis_exit=0.30,
        cooldown_ms=500,
        adaptive_gain=True,
        gain_floor=1.0,
        gain_ceiling=20.0,
        ema_decay=0.9993,
    ),
    4: AssistProfile(
        level=4,
        label="Light Assist",
        confidence_floor=0.45,
        movement_bias=0.03,
        assist_exponent=0.9,
        stability_required=3,
        proba_ema_alpha=0.4,
        hysteresis_enter=0.60,
        hysteresis_exit=0.35,
        cooldown_ms=400,
        adaptive_gain=True,
        gain_floor=1.0,
        gain_ceiling=10.0,
        ema_decay=0.9996,
    ),
    5: AssistProfile(
        level=5,
        label="Minimal Assist",
        confidence_floor=0.55,
        movement_bias=0.0,
        assist_exponent=1.0,
        stability_required=3,
        proba_ema_alpha=0.3,        # light smoothing — responsive
        hysteresis_enter=0.70,      # high bar to trigger
        hysteresis_exit=0.40,       # must clearly stop to return to rest
        cooldown_ms=300,            # short cooldown — fast response
        adaptive_gain=False,
        gain_floor=1.0,
        gain_ceiling=1.0,
        ema_decay=0.9998,
    ),
}


def get_profile(level: int) -> AssistProfile:
    """Return the AssistProfile for a given level (1-5).

    Raises ValueError if level is out of range.
    """
    if level not in ASSIST_PROFILES:
        raise ValueError(
            f"Assist level must be 1-5, got {level}. "
            "1 = Max Assist (early rehab), 5 = Minimal (near-healthy)."
        )
    return ASSIST_PROFILES[level]


def adjust_profile_for_patient(profile: AssistProfile, signal_to_noise: float) -> float:
    """Compute a threshold scaling factor based on observed SNR.

    Very weak patients (SNR < 2) get thresholds halved so faint intent
    still triggers movement.  Strong signals (SNR > 10) use defaults.

    Args:
        profile: the current assist profile
        signal_to_noise: ratio of active EMG amplitude to rest amplitude

    Returns:
        Scaling factor (0.5–1.0) to multiply hysteresis/confidence thresholds.
    """
    if signal_to_noise >= 10.0:
        return 1.0
    if signal_to_noise <= 2.0:
        return 0.5
    # Linear interpolation between SNR=2 (scale=0.5) and SNR=10 (scale=1.0)
    return 0.5 + 0.5 * (signal_to_noise - 2.0) / 8.0


def print_profile(profile: AssistProfile) -> None:
    """Print a human-readable summary of the profile."""
    print(f"  Assist Level {profile.level}: {profile.label}")
    print(f"    Confidence floor : {profile.confidence_floor:.2f}")
    print(f"    Movement bias    : {profile.movement_bias:.2f}")
    print(f"    Assist exponent  : {profile.assist_exponent:.1f}")
    print(f"    Stability filter : {profile.stability_required} prediction(s)")
    print(f"    Proba smoothing  : {profile.proba_ema_alpha:.1f}")
    print(f"    Hysteresis       : enter={profile.hysteresis_enter:.2f}  exit={profile.hysteresis_exit:.2f}")
    print(f"    Cooldown         : {profile.cooldown_ms:.0f}ms")
    print(f"    Adaptive gain    : {'ON' if profile.adaptive_gain else 'OFF'}")
    if profile.adaptive_gain:
        print(f"    Gain range       : {profile.gain_floor:.1f}x – {profile.gain_ceiling:.1f}x")
        print(f"    EMA decay        : {profile.ema_decay}")

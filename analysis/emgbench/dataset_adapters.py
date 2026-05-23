"""
Per-dataset adapters: channel selection + gesture-to-3-class mapping.

Each EMGBench dataset has its own channel count, electrode geometry, and
gesture vocabulary. This module centralizes the dataset-specific decisions
so the main runner stays generic.

CHANNEL SELECTION RATIONALE:
  We map every dataset to a canonical 4-channel layout indexed as [0, 4, 9, 13]
  (matching exohand_adapted_model.pkl's channel_map). The actual dataset
  channel indices are chosen to approximate flexor digitorum + extensor
  digitorum positions on the forearm. Where the dataset lacks anatomical
  labels we use evenly-spaced positions around the forearm grid as a
  conservative default. SURFACED AS A DECISION — verify against per-dataset
  electrode layout figures before locking in.

GESTURE MAPPING RATIONALE:
  Same principle as ml/build_intent_dataset.py applied to GrabMyo:
    - Flexor-dominant grip gestures (fist, pinch, lateral) → close
    - Any finger-extension gesture (single or multi) → open
    - Rest → rest
    - Wrist flexion/extension, forearm rotation, idiosyncratic gestures → drop
  Per-dataset maps below; gestures not in the map are dropped.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class DatasetAdapter:
    name: str                            # EMGBench --dataset arg
    sample_rate_hz: float
    n_channels_total: int
    selected_channels: Sequence[int]     # 4 indices into the dataset's channel array
    channel_name_map: Sequence[int]      # canonical naming, always [0, 4, 9, 13]
    gesture_to_intent: dict              # raw label → 'close' | 'open' | 'rest'
    notes: str = ""


# Mapping of intent strings to GrabMyo-canonical indices (matches build_intent_dataset.py)
INTENT_TO_IDX = {"rest": 0, "close": 1, "open": 2}


# --- CapgMyo: 8 channels circular around forearm, 1 kHz ---
#   Source: utils_CapgMyo.py in jehanyang/emgbench (verify before locking).
#   Gestures: 8 per session (DB-B subset used by EMGBench). Mapping below
#   reflects EMGBench's published label set; CONFIRM against utils file.
CAPGMYO = DatasetAdapter(
    name="capgmyo",
    sample_rate_hz=1000.0,
    n_channels_total=8,
    selected_channels=[0, 2, 4, 6],      # evenly-spaced around forearm
    channel_name_map=[0, 4, 9, 13],
    gesture_to_intent={
        # PLACEHOLDER — verify against utils_CapgMyo.gesture_labels.
        # CapgMyo-B common labels (subject to confirmation):
        #   "thumb_up", "extension_index_middle", "flexion_ring_little",
        #   "thumb_opposing", "pinch_grasp", "abduction_all", "fist", "wave_in"
        "fist": "close",
        "pinch_grasp": "close",
        "thumb_opposing": "close",
        "wave_in": "close",
        "extension_index_middle": "open",
        "abduction_all": "open",
        "thumb_up": "open",
        "flexion_ring_little": "open",     # NOTE: 'flexion' name but in CapgMyo B this is an extension-like posture; verify
    },
    notes="Channel selection is evenly-spaced (every other electrode); confirm "
          "anatomical mapping before locking. Gesture map placeholder — must be "
          "verified against utils_CapgMyo.gesture_labels exact strings.",
)


# --- NinaproDB5: 16 channels (2 wrist arrays × 8), 200 Hz, gestures from exercises 1+2 ---
NINAPRO_DB5 = DatasetAdapter(
    name="ninapro-db5",
    sample_rate_hz=200.0,
    n_channels_total=16,
    selected_channels=[0, 4, 8, 12],     # spans both arrays
    channel_name_map=[0, 4, 9, 13],
    gesture_to_intent={
        # PLACEHOLDER — NinaproDB5 has many gestures across exercises 1-3.
        # Confirm against utils_NinaproDB5.gesture_labels.
        # Typical Ninapro labels include: index_flex, middle_flex, ring_flex,
        # little_flex, thumb_flex (close family); index_ext, middle_ext, ring_ext,
        # little_ext, thumb_ext (open family); wrist_flex, wrist_ext (drop).
        # Names below are placeholders; replace with exact utils strings.
        "rest": "rest",
    },
    notes="EMGBench by default uses --partial_dataset_ninapro=True (exercise 2 only). "
          "Channel selection spans both wrist arrays. Mapping must be verified.",
)


# --- MyoArmbandDataset: 8 channels (Myo armband), 200 Hz ---
MYOARMBAND = DatasetAdapter(
    name="myoarmbanddataset",
    sample_rate_hz=200.0,
    n_channels_total=8,
    selected_channels=[0, 2, 4, 6],
    channel_name_map=[0, 4, 9, 13],
    gesture_to_intent={
        # Myo standard set: hand_close, hand_open, wrist_pron, wrist_supin, rest
        "hand_close": "close",
        "hand_open": "open",
        "rest": "rest",
        # wrist_pronation / wrist_supination dropped per GrabMyo principle
    },
    notes="Myo at 200 Hz is the lowest-bandwidth dataset; expect lower per-window "
          "feature quality.",
)


# --- Hyser: 256 channels HD-sEMG, 2 kHz ---
HYSER = DatasetAdapter(
    name="hyser",
    sample_rate_hz=2048.0,
    n_channels_total=256,
    selected_channels=[31, 63, 159, 191],  # rough flexor/extensor strip indices in a 16x16 grid; VERIFY
    channel_name_map=[0, 4, 9, 13],
    gesture_to_intent={
        # Hyser is multi-finger force tracking + gesture; placeholder.
        "rest": "rest",
    },
    notes="256 channels — channel selection has the highest information loss. "
          "Empirical activation-based selection strongly recommended.",
)


# --- UCI EMG: variable, 200 Hz ---
UCIEMG = DatasetAdapter(
    name="uciemg",
    sample_rate_hz=200.0,
    n_channels_total=8,
    selected_channels=[0, 2, 4, 6],
    channel_name_map=[0, 4, 9, 13],
    gesture_to_intent={
        "rest": "rest",
    },
    notes="Verify channel count + gesture labels from utils_UCI.py before running.",
)


# --- FlexWear-HD: HD-sEMG flexible wearable, sample rate TBD ---
FLEXWEAR_HD = DatasetAdapter(
    name="flexwear-hd",
    sample_rate_hz=4000.0,               # TBD — verify
    n_channels_total=64,                 # TBD — verify
    selected_channels=[0, 16, 32, 48],   # placeholder
    channel_name_map=[0, 4, 9, 13],
    gesture_to_intent={
        "rest": "rest",
    },
    notes="EMGBench's own contributed dataset; specs and gesture list need "
          "verification against utils_FlexWearHD.py.",
)


ADAPTERS = {
    "capgmyo": CAPGMYO,
    "ninapro-db5": NINAPRO_DB5,
    "myoarmbanddataset": MYOARMBAND,
    "hyser": HYSER,
    "uciemg": UCIEMG,
    "flexwear-hd": FLEXWEAR_HD,
}


def get_adapter(dataset_name: str) -> DatasetAdapter:
    name = dataset_name.lower()
    if name not in ADAPTERS:
        raise KeyError(f"No adapter for dataset '{dataset_name}'. "
                       f"Available: {sorted(ADAPTERS.keys())}")
    return ADAPTERS[name]

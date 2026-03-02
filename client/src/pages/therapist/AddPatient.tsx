import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Slider } from "../../components/ui/Slider";
import { api } from "../../lib/api";

const ASSIST_LEVELS: Record<number, { label: string; description: string }> = {
  1: {
    label: "Minimal Assist",
    description: "Patient has near-full voluntary control. The exoskeleton provides subtle guidance only when drift is detected.",
  },
  2: {
    label: "Light Assist",
    description: "Patient initiates movement independently. The system supplements force during the last 20–30% of range of motion.",
  },
  3: {
    label: "Moderate Assist",
    description: "Patient provides partial intent via EMG. The motor shares roughly equal effort, filling gaps in strength or coordination.",
  },
  4: {
    label: "Significant Assist",
    description: "The exoskeleton drives most of the movement. Patient contributes detectable EMG intent to trigger and guide motion.",
  },
  5: {
    label: "Full Assist",
    description: "The system executes the full range of motion. Used for early-stage patients with minimal voluntary activation.",
  },
};

export function AddPatient() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [pin, setPin] = useState("");
  const [dob, setDob] = useState("");
  const [hospital, setHospital] = useState("");
  const [description, setDescription] = useState("");
  const [assistLevel, setAssistLevel] = useState(3);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const level = ASSIST_LEVELS[assistLevel];

  const handleSubmit = async () => {
    setError("");

    if (!name.trim()) { setError("Patient name is required."); return; }
    if (!pin.trim() || pin.length < 4) { setError("PIN must be at least 4 digits."); return; }

    setSaving(true);
    try {
      const patient = await api.createPatient({
        name: name.trim(),
        pin: pin.trim(),
        description: description.trim(),
        assist_level: assistLevel,
        dob: dob.trim(),
        hospital: hospital.trim(),
      });
      // Navigate to full calibration for this new patient
      navigate(`/therapist/patient/${patient.id}?calibrate=true`);
    } catch (err: any) {
      setError(err.message || "Failed to create patient.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="h-full overflow-auto bg-bg">
      <div className="max-w-lg mx-auto px-6 py-16">
        <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-4">
          New Patient
        </p>
        <h1 className="text-h2 font-medium text-text mb-10">
          Register a patient for rehabilitation.
        </h1>

        <div className="space-y-6">
          {/* Name */}
          <div>
            <label className="text-small text-muted block mb-2 font-mono uppercase tracking-wider">
              Patient Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Full name"
              className="w-full px-4 py-3 bg-white/[0.04] border border-border rounded-lg text-text placeholder-muted/40 focus:outline-none focus:border-white/20 transition-colors font-mono"
            />
          </div>

          {/* PIN */}
          <div>
            <label className="text-small text-muted block mb-2 font-mono uppercase tracking-wider">
              Patient PIN
            </label>
            <input
              type="text"
              value={pin}
              onChange={(e) => setPin(e.target.value.replace(/\D/g, "").slice(0, 6))}
              placeholder="4–6 digit login PIN"
              className="w-full px-4 py-3 bg-white/[0.04] border border-border rounded-lg text-text placeholder-muted/40 focus:outline-none focus:border-white/20 transition-colors font-mono tracking-[0.3em]"
            />
          </div>

          {/* Date of Birth */}
          <div>
            <label className="text-small text-muted block mb-2 font-mono uppercase tracking-wider">
              Date of Birth
            </label>
            <input
              type="date"
              value={dob}
              onChange={(e) => setDob(e.target.value)}
              className="w-full px-4 py-3 bg-white/[0.04] border border-border rounded-lg text-text focus:outline-none focus:border-white/20 transition-colors font-mono [color-scheme:dark]"
            />
          </div>

          {/* Hospital */}
          <div>
            <label className="text-small text-muted block mb-2 font-mono uppercase tracking-wider">
              Registered Hospital
            </label>
            <input
              type="text"
              value={hospital}
              onChange={(e) => setHospital(e.target.value)}
              placeholder="Hospital or clinic name"
              className="w-full px-4 py-3 bg-white/[0.04] border border-border rounded-lg text-text placeholder-muted/40 focus:outline-none focus:border-white/20 transition-colors font-mono"
            />
          </div>

          {/* Description */}
          <div>
            <label className="text-small text-muted block mb-2 font-mono uppercase tracking-wider">
              Current State
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe the patient's current condition, injury history, and rehabilitation goals..."
              rows={4}
              className="w-full px-4 py-3 bg-white/[0.04] border border-border rounded-lg text-text placeholder-muted/40 focus:outline-none focus:border-white/20 transition-colors resize-none text-small leading-relaxed"
            />
          </div>

          {/* Assist Level */}
          <div>
            <Slider
              label="Assist Level"
              value={assistLevel}
              min={1}
              max={5}
              onChange={setAssistLevel}
            />
            <div className="mt-3 p-4 bg-white/[0.03] border border-border rounded-lg">
              <p className="text-small font-mono text-text mb-1">{level.label}</p>
              <p className="text-small text-muted leading-relaxed">{level.description}</p>
            </div>
          </div>

          {/* Error */}
          {error && (
            <p className="text-small text-danger">{error}</p>
          )}

          {/* Actions */}
          <div className="flex gap-3 pt-4">
            <button
              onClick={() => navigate("/therapist")}
              className="px-6 py-3 rounded-lg border border-white/10 text-muted font-mono text-small tracking-wide hover:bg-white/[0.04] transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={saving}
              className="flex-1 px-6 py-3 rounded-lg border border-white/20 text-text font-mono text-body tracking-wide hover:bg-white/[0.06] transition-colors disabled:opacity-40"
            >
              {saving ? "Creating..." : "Create Patient & Calibrate"}
            </button>
          </div>

          <p className="text-[11px] text-muted/40 font-mono">
            After creation, you will be redirected to run full calibration for this patient.
          </p>
        </div>
      </div>
    </div>
  );
}

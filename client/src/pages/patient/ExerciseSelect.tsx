import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../../components/ui/Button";
import { TopBar } from "../../components/ui/TopBar";
import { LoadingScreen } from "../../components/ui/LoadingScreen";

/* ================================================================
   Exercise definitions
   Hardware angles: 110°=OPEN, 145°=REST, 180°=CLOSE
   The startAngle/targetAngle values here drive the 3D hand animation
   (not motor commands directly).
   ================================================================ */

export interface ExerciseDef {
  id: string;
  name: string;
  description: string;
  startAngle: number;
  targetAngle: number;
  holdSeconds: number;
  reps: number;
  category: "close" | "open" | "combined";
}

const EXERCISES: ExerciseDef[] = [
  {
    id: "close_full",
    name: "Close",
    description: "Close from rest to full flexion",
    startAngle: 110,
    targetAngle: 180,
    holdSeconds: 0,
    reps: 10,
    category: "close",
  },
  {
    id: "close_hold",
    name: "Close & Hold",
    description: "Close fully and hold for 3 seconds",
    startAngle: 110,
    targetAngle: 180,
    holdSeconds: 3,
    reps: 10,
    category: "close",
  },
  {
    id: "open_full",
    name: "Open",
    description: "Open from rest to full extension",
    startAngle: 110,
    targetAngle: 0,
    holdSeconds: 0,
    reps: 10,
    category: "open",
  },
  {
    id: "open_hold",
    name: "Open & Hold",
    description: "Open fully and hold for 3 seconds",
    startAngle: 110,
    targetAngle: 0,
    holdSeconds: 3,
    reps: 10,
    category: "open",
  },
  {
    id: "open_close",
    name: "Open & Close",
    description: "Full range open then close — trains both directions",
    startAngle: 110,
    targetAngle: 180,
    holdSeconds: 0,
    reps: 10,
    category: "combined",
  },
];

interface ExerciseSelectProps {
  patientId: number;
}

export function ExerciseSelect({ patientId }: ExerciseSelectProps) {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const selectedExercise = EXERCISES.find((e) => e.id === selected);

  const handleSelect = (ex: ExerciseDef) => {
    setSelected(ex.id);
    // Store in sessionStorage for the calibration + session flow
    sessionStorage.setItem("activeExercise", JSON.stringify(ex));
  };

  const handleContinue = () => {
    if (!selected) return;
    setLoading(true);
    setTimeout(() => navigate("/patient/session"), 1500);
  };

  if (loading) {
    return (
      <LoadingScreen
        message="Loading demo..."
        submessage="Preparing guided movement preview."
      />
    );
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <TopBar title="Select Exercise" subtitle="Choose your exercise for this session" />
      <div className="flex-1 flex flex-col items-center justify-center p-6">
        <div className="w-full max-w-3xl space-y-4">
          {(["close", "open", "combined"] as const).map((cat) => {
            const exercises = EXERCISES.filter((e) => e.category === cat);
            if (exercises.length === 0) return null;
            const label = cat === "close" ? "Close" : cat === "open" ? "Open" : "Combined";
            return (
              <div key={cat}>
                <h3 className="text-small text-muted uppercase tracking-wider mb-2">
                  {label} Exercises
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  {exercises.map((ex) => (
                    <button
                      key={ex.id}
                      onClick={() => handleSelect(ex)}
                      className={`text-left p-3 rounded-lg border transition-all ${
                        selected === ex.id
                          ? "border-purple bg-purple/10"
                          : "border-border bg-panel hover:bg-panel2"
                      }`}
                    >
                      <p className="text-body font-semibold text-text">{ex.name}</p>
                      <p className="text-small text-muted mt-0.5">{ex.description}</p>
                      <div className="mt-2 flex gap-3 text-[11px] font-mono text-muted">
                        {ex.holdSeconds > 0 && (
                          <span>Hold {ex.holdSeconds}s</span>
                        )}
                        <span>{ex.reps} reps</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            );
          })}

          {/* Continue button */}
          <div className="flex justify-end pt-2">
            <Button size="lg" disabled={!selected} onClick={handleContinue}>
              Start Exercise
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

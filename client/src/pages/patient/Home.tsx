import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../../lib/api";

const SECTION_COUNT = 3;

/* ── Exact same scroll hook as Landing.tsx ── */
function useFullPageScroll(sectionCount: number) {
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    let index = 0;
    let cooldown = false;
    let touchStartY = 0;

    const go = (dir: number) => {
      if (cooldown) return;
      const next = index + dir;
      if (next < 0 || next >= sectionCount) return;
      cooldown = true;
      index = next;
      setCurrent(next);
      setTimeout(() => { cooldown = false; }, 1000);
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (Math.abs(e.deltaY) < 15) return;
      go(e.deltaY > 0 ? 1 : -1);
    };

    const onTouchStart = (e: TouchEvent) => { touchStartY = e.touches[0].clientY; };
    const onTouchEnd = (e: TouchEvent) => {
      const d = touchStartY - e.changedTouches[0].clientY;
      if (Math.abs(d) > 50) go(d > 0 ? 1 : -1);
    };

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown" || e.key === " ") { e.preventDefault(); go(1); }
      if (e.key === "ArrowUp") { e.preventDefault(); go(-1); }
    };

    window.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchend", onTouchEnd, { passive: true });
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("wheel", onWheel);
      window.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchend", onTouchEnd);
      window.removeEventListener("keydown", onKey);
    };
  }, [sectionCount]);

  return current;
}

interface HomeProps {
  patientId: number;
  displayName?: string;
}

export function PatientHome({ patientId, displayName }: HomeProps) {
  const navigate = useNavigate();
  const [patient, setPatient] = useState<any>(null);
  const [progress, setProgress] = useState<any>(null);
  const current = useFullPageScroll(SECTION_COUNT);

  useEffect(() => {
    api.getPatient(patientId).then(setPatient);
    api.getPatientProgress(patientId).then(setProgress);
  }, [patientId]);

  if (!patient) return <div className="h-screen flex items-center justify-center text-muted">Loading...</div>;

  return (
    <div className="h-screen overflow-hidden bg-bg">
      <div
        className="transition-transform duration-[600ms] ease-[cubic-bezier(0.16,1,0.3,1)]"
        style={{ transform: `translateY(-${current * 100}vh)` }}
      >

        {/* ── 1. Welcome ── */}
        <section className="h-screen flex flex-col items-center justify-center px-6 text-center relative">
          <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-6">
            Welcome back
          </p>
          <h1 className="text-h1 font-bold text-text font-mono tracking-wide">
            {displayName || patient.name}
          </h1>

          <div className="absolute bottom-10 text-muted/30 animate-bounce">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </section>

        {/* ── 2. Start Session ── */}
        <section className="h-screen flex flex-col items-center justify-center px-6 text-center">
          <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-4">
            Session
          </p>
          <h2 className="text-h2 font-medium text-text mb-4 max-w-md">
            Begin your next exercise session.
          </h2>
          <p className="text-small text-muted mb-10 max-w-sm">
            A quick calibration will prepare the sensors, then you'll choose your exercise.
          </p>
          <button
            onClick={() => navigate("/patient/calibration")}
            className="px-8 py-3 rounded-lg border border-white/20 text-text font-mono text-body tracking-wide hover:bg-white/[0.06] transition-colors"
          >
            Start Session
          </button>
        </section>

        {/* ── 3. Progress ── */}
        <section className="h-screen flex flex-col items-center justify-center px-6 text-center">
          <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-6">
            Progress
          </p>

          <div className="grid grid-cols-3 gap-12 mb-12">
            <div>
              <p className="text-[clamp(2rem,4vw,3rem)] font-bold font-mono text-text leading-none mb-2">
                {progress?.totalSessions ?? 0}
              </p>
              <p className="text-small text-muted">Sessions</p>
            </div>
            <div>
              <p className="text-[clamp(2rem,4vw,3rem)] font-bold font-mono text-text leading-none mb-2">
                {progress?.trends?.score?.current?.toFixed(0) ?? "—"}
              </p>
              <p className="text-small text-muted">Current Score</p>
            </div>
            <div>
              <p className="text-[clamp(2rem,4vw,3rem)] font-bold font-mono text-text leading-none mb-2">
                {patient.assist_level}
              </p>
              <p className="text-small text-muted">Assist Level</p>
            </div>
          </div>

          <button
            onClick={() => navigate("/patient/progress")}
            className="px-8 py-3 rounded-lg border border-white/20 text-text font-mono text-body tracking-wide hover:bg-white/[0.06] transition-colors"
          >
            View Progress
          </button>
        </section>

      </div>
    </div>
  );
}

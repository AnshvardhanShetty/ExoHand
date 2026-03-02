import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Badge } from "../../components/ui/Badge";
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

interface DashboardProps {
  therapistName: string;
}

export function TherapistDashboard({ therapistName }: DashboardProps) {
  const navigate = useNavigate();
  const [patients, setPatients] = useState<any[]>([]);
  const current = useFullPageScroll(SECTION_COUNT);

  useEffect(() => {
    api.getTherapistPatients().then(setPatients);
  }, []);

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
            {therapistName}
          </h1>

          <div className="absolute bottom-10 text-muted/30 animate-bounce">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </section>

        {/* ── 2. Registered Patients ── */}
        <section className="h-screen flex flex-col justify-center px-6 relative">
          <div className="max-w-4xl mx-auto w-full">
            <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-4">
              Patients
            </p>
            <h2 className="text-h2 font-medium text-text mb-10">
              {patients.length} registered {patients.length === 1 ? "patient" : "patients"}.
            </h2>

            {patients.length === 0 ? (
              <p className="text-body text-muted">No patients yet. Scroll down to add one.</p>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {patients.map((p) => (
                  <button
                    key={p.id}
                    onClick={() => navigate(`/therapist/patient/${p.id}`)}
                    className="bg-panel border border-border rounded-xl p-5 text-left hover:border-white/10 hover:bg-white/[0.02] transition-colors group"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-body font-semibold text-text font-mono tracking-wide">
                        {p.name}
                      </h3>
                      <span className="text-small font-mono text-muted">
                        Lvl {p.assist_level}
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-3 mb-3">
                      <div>
                        <p className="text-[11px] uppercase tracking-wider text-muted">Sessions</p>
                        <p className="text-h3 font-bold font-mono text-text">{p.session_count ?? 0}</p>
                      </div>
                      <div>
                        <p className="text-[11px] uppercase tracking-wider text-muted">Score</p>
                        <p className="text-h3 font-bold font-mono text-text">
                          {p.last_score != null ? p.last_score.toFixed(0) : "—"}
                        </p>
                      </div>
                      <div>
                        <p className="text-[11px] uppercase tracking-wider text-muted">Stability</p>
                        <p className="text-h3 font-bold font-mono text-text">
                          {p.last_stability != null ? p.last_stability.toFixed(0) : "—"}
                        </p>
                      </div>
                    </div>

                    {(p.alerts || []).length > 0 && (
                      <div className="flex gap-1 flex-wrap">
                        {p.alerts.map((alert: string, i: number) => (
                          <Badge
                            key={i}
                            variant={
                              alert.includes("progress")
                                ? "success"
                                : alert.includes("Declining")
                                  ? "danger"
                                  : "warning"
                            }
                          >
                            {alert}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="absolute bottom-10 left-1/2 -translate-x-1/2 text-muted/30 animate-bounce">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </section>

        {/* ── 3. Add New Patient ── */}
        <section className="h-screen flex flex-col items-center justify-center px-6 text-center">
          <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-4">
            Onboard
          </p>
          <h2 className="text-h2 font-medium text-text mb-4 max-w-md">
            Register a new patient.
          </h2>
          <p className="text-small text-muted mb-10 max-w-sm">
            Set up their profile, configure assist level, and run full calibration.
          </p>
          <button
            onClick={() => navigate("/therapist/add-patient")}
            className="px-8 py-3 rounded-lg border border-white/20 text-text font-mono text-body tracking-wide hover:bg-white/[0.06] transition-colors"
          >
            Add Patient
          </button>
        </section>

      </div>
    </div>
  );
}

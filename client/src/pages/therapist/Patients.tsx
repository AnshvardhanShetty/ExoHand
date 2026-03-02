import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Badge } from "../../components/ui/Badge";
import { TopBar } from "../../components/ui/TopBar";
import { api } from "../../lib/api";

export function Patients() {
  const navigate = useNavigate();
  const [patients, setPatients] = useState<any[]>([]);
  const [confirmId, setConfirmId] = useState<number | null>(null);

  useEffect(() => {
    api.getTherapistPatients().then(setPatients);
  }, []);

  const handleDelete = async (id: number) => {
    await api.deletePatient(id);
    setConfirmId(null);
    setPatients((prev) => prev.filter((p) => p.id !== id));
  };

  return (
    <div>
      <TopBar title="Patients" subtitle={`${patients.length} registered`} />
      <div className="p-6 max-w-4xl mx-auto">
        {patients.length === 0 ? (
          <p className="text-body text-muted">No patients yet.</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {patients.map((p) => (
              <div
                key={p.id}
                className="bg-panel border border-border rounded-xl p-5 text-left hover:border-white/10 hover:bg-white/[0.02] transition-colors group relative"
              >
                <button
                  onClick={() => navigate(`/therapist/patient/${p.id}`)}
                  className="w-full text-left"
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
                        {p.last_score != null ? p.last_score.toFixed(0) : "\u2014"}
                      </p>
                    </div>
                    <div>
                      <p className="text-[11px] uppercase tracking-wider text-muted">Stability</p>
                      <p className="text-h3 font-bold font-mono text-text">
                        {p.last_stability != null ? p.last_stability.toFixed(0) : "\u2014"}
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

                {confirmId === p.id ? (
                  <div className="mt-3 pt-3 border-t border-border flex items-center gap-2">
                    <span className="text-small text-muted flex-1">Remove patient?</span>
                    <button
                      onClick={() => handleDelete(p.id)}
                      className="px-3 py-1 rounded text-small font-mono bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
                    >
                      Confirm
                    </button>
                    <button
                      onClick={() => setConfirmId(null)}
                      className="px-3 py-1 rounded text-small font-mono text-muted hover:bg-white/[0.06] transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setConfirmId(p.id)}
                    className="mt-3 pt-3 border-t border-border w-full text-left text-small text-muted/50 hover:text-red-400 transition-colors font-mono"
                  >
                    Remove
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

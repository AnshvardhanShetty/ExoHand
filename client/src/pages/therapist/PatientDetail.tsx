import React, { useEffect, useState, useCallback } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Card } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Badge";
import { Slider } from "../../components/ui/Slider";
import { Tabs } from "../../components/ui/Tabs";
import { TopBar } from "../../components/ui/TopBar";
import { api } from "../../lib/api";
import { Button } from "../../components/ui/Button";

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

const tooltipStyle = {
  backgroundColor: "#111520",
  border: "1px solid rgba(148,163,184,0.08)",
  borderRadius: "8px",
  color: "#E2E8F0",
  fontSize: "12px",
};

export function PatientDetail() {
  const { patientId } = useParams();
  const location = useLocation();
  const shouldCalibrate = new URLSearchParams(location.search).get("calibrate") === "true";
  const [data, setData] = useState<any>(null);
  const [assistLevel, setAssistLevel] = useState(3);
  const [saving, setSaving] = useState(false);
  const navigate = useNavigate();

  const loadData = useCallback(() => {
    if (patientId) {
      api.getTherapistPatientDetail(Number(patientId)).then((d) => {
        setData(d);
        if (d.patient) {
          setAssistLevel(d.patient.assist_level);
        }
      });
    }
  }, [patientId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const saveAssistLevel = async () => {
    if (!patientId) return;
    setSaving(true);
    await api.updatePatientSettings(Number(patientId), { assist_level: assistLevel });
    setSaving(false);
    loadData();
  };

  if (!data) return <div className="p-8 text-center text-muted">Loading...</div>;

  const { patient, sessions, safetyEvents } = data;

  const EXERCISE_NAMES: Record<string, string> = {
    close_full: "Close", close_hold: "Close & Hold",
    open_full: "Open", open_hold: "Open & Hold",
    open_close: "Open & Close",
  };
  const EXERCISE_COLORS: Record<string, string> = {
    close_full: "#3B82F6", close_hold: "#8B5CF6",
    open_full: "#F59E0B", open_hold: "#EF4444",
    open_close: "#22C55E",
  };

  const reversedSessions = [...sessions].reverse();

  const chartData = reversedSessions.map((s: any, i: number) => ({
    session: i + 1,
    score: s.overall_score,
    stability: s.avg_stability,
    completion: s.completion_rate,
    accuracy: s.avg_accuracy,
    date: new Date(s.started_at).toLocaleDateString(),
  }));

  // Per-exercise chart data
  const exerciseTypes = [...new Set(reversedSessions.map((s: any) => s.exercise_type).filter(Boolean))] as string[];
  const exerciseChartData = reversedSessions.map((s: any, i: number) => {
    const point: any = { session: i + 1 };
    for (const ex of exerciseTypes) {
      point[ex] = s.exercise_type === ex ? s.overall_score : null;
    }
    return point;
  });
  const hasExerciseData = reversedSessions.filter((s: any) => s.exercise_type).length >= 2;

  const level = ASSIST_LEVELS[assistLevel];

  /* ── Details Tab ── */
  const detailsTab = (
    <div className="space-y-5">
      <Card>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-6">
            <div>
              <p className="text-[11px] font-mono uppercase tracking-wider text-muted mb-1">Patient Name</p>
              <p className="text-body text-text font-mono">{patient.name}</p>
            </div>
            <div>
              <p className="text-[11px] font-mono uppercase tracking-wider text-muted mb-1">Patient ID</p>
              <p className="text-body text-text font-mono">{patient.pin}</p>
            </div>
            <div>
              <p className="text-[11px] font-mono uppercase tracking-wider text-muted mb-1">Date of Birth</p>
              <p className="text-body text-text font-mono">{patient.dob || "—"}</p>
            </div>
            <div>
              <p className="text-[11px] font-mono uppercase tracking-wider text-muted mb-1">Registered Hospital</p>
              <p className="text-body text-text font-mono">{patient.hospital || "—"}</p>
            </div>
            <div>
              <p className="text-[11px] font-mono uppercase tracking-wider text-muted mb-1">Registered</p>
              <p className="text-body text-text font-mono">
                {new Date(patient.created_at).toLocaleDateString()}
              </p>
            </div>
            <div>
              <p className="text-[11px] font-mono uppercase tracking-wider text-muted mb-1">Total Sessions</p>
              <p className="text-body text-text font-mono">{sessions.length}</p>
            </div>
          </div>

          {patient.description && (
            <div className="pt-4 border-t border-border">
              <p className="text-[11px] font-mono uppercase tracking-wider text-muted mb-2">Current State</p>
              <p className="text-small text-muted leading-relaxed">{patient.description}</p>
            </div>
          )}
        </div>
      </Card>
    </div>
  );

  /* ── Trends Tab ── */
  const trendsTab = (
    <div className="space-y-5">
      {chartData.length > 1 ? (
        <>
          <Card>
            <h3 className="text-small font-medium text-muted mb-4">Session Scores</h3>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="session" stroke="rgba(255,255,255,0.2)" fontSize={11} />
                <YAxis domain={[0, 100]} stroke="rgba(255,255,255,0.2)" fontSize={11} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="score" stroke="#1D4ED8" strokeWidth={2} name="Score" />
                <Line type="monotone" dataKey="accuracy" stroke="#F59E0B" strokeWidth={2} name="Accuracy" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
          <Card>
            <h3 className="text-small font-medium text-muted mb-4">Stability & Completion</h3>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="session" stroke="rgba(255,255,255,0.2)" fontSize={11} />
                <YAxis domain={[0, 100]} stroke="rgba(255,255,255,0.2)" fontSize={11} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="stability" stroke="#1D4ED8" strokeWidth={2} name="Stability" />
                <Line type="monotone" dataKey="completion" stroke="#22C55E" strokeWidth={2} name="Completion" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
          {hasExerciseData && (
            <Card>
              <h3 className="text-small font-medium text-muted mb-4">Score by Exercise</h3>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={exerciseChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="session" stroke="rgba(255,255,255,0.2)" fontSize={11} />
                  <YAxis domain={[0, 100]} stroke="rgba(255,255,255,0.2)" fontSize={11} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend />
                  {exerciseTypes.map((ex) => (
                    <Line
                      key={ex}
                      type="monotone"
                      dataKey={ex}
                      stroke={EXERCISE_COLORS[ex] || "#94A3B8"}
                      strokeWidth={2}
                      name={EXERCISE_NAMES[ex] || ex}
                      connectNulls
                      dot={{ fill: EXERCISE_COLORS[ex] || "#94A3B8", r: 3 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}
        </>
      ) : (
        <Card>
          <p className="text-center text-muted py-8">Not enough sessions for charts.</p>
        </Card>
      )}
    </div>
  );

  /* ── Assist Level Tab ── */
  const assistTab = (
    <div className="space-y-5">
      <Card>
        <h3 className="text-[11px] font-mono uppercase tracking-wider text-muted mb-6">Assist Level</h3>
        <div className="space-y-4">
          <Slider
            label="Level"
            value={assistLevel}
            min={1}
            max={5}
            onChange={setAssistLevel}
          />
          <div className="p-4 bg-white/[0.03] border border-border rounded-lg">
            <p className="text-small font-mono text-text mb-1">{level.label}</p>
            <p className="text-small text-muted leading-relaxed">{level.description}</p>
          </div>
          <button
            onClick={saveAssistLevel}
            disabled={saving || assistLevel === patient.assist_level}
            className="px-6 py-2.5 rounded-lg border border-white/20 text-text font-mono text-small tracking-wide hover:bg-white/[0.06] transition-colors disabled:opacity-40"
          >
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </Card>
    </div>
  );

  /* ── Calibration Tab ── */
  const calibrationTab = (
    <Card>
      <h3 className="text-[11px] font-mono uppercase tracking-wider text-muted mb-4">EMG Calibration</h3>
      <p className="text-small text-muted mb-6">
        Run a calibration session to train the EMG model for this patient. Calibration opens in a full-screen view.
      </p>
      <div className="flex gap-3">
        <Button size="md" onClick={() => navigate(`/therapist/patient/${patientId}/calibration`)}>
          Start Calibration
        </Button>
      </div>
    </Card>
  );

  /* ── Safety Tab ── */
  const safetyTab = (
    <Card>
      <h3 className="text-[11px] font-mono uppercase tracking-wider text-muted mb-4">Safety Events</h3>
      {safetyEvents.length === 0 ? (
        <p className="text-muted text-small">No safety events recorded.</p>
      ) : (
        <div className="space-y-2 max-h-96 overflow-auto">
          {safetyEvents.map((evt: any) => (
            <div key={evt.id} className="flex items-center gap-3 text-small py-2 border-b border-border/50">
              <Badge variant={evt.event_type === "fail_safe_open" ? "danger" : "warning"}>
                {evt.event_type}
              </Badge>
              <span className="text-muted font-mono text-[11px]">
                {new Date(evt.timestamp).toLocaleString()}
              </span>
              <span className="text-text">{evt.details}</span>
            </div>
          ))}
        </div>
      )}
    </Card>
  );

  return (
    <div>
      <TopBar
        title={patient.name}
        subtitle={`Level ${patient.assist_level} · ${sessions.length} sessions`}
      />
      <div className="p-6 max-w-4xl mx-auto">
        <Tabs
          defaultTab={shouldCalibrate ? "calibration" : undefined}
          tabs={[
            { id: "details", label: "Details", content: detailsTab },
            { id: "trends", label: "Trends", content: trendsTab },
            { id: "assist", label: "Assist Level", content: assistTab },
            { id: "calibration", label: "Calibration", content: calibrationTab },
            { id: "safety", label: "Safety", content: safetyTab },
          ]}
        />
      </div>
    </div>
  );
}

import React, { useEffect, useState } from "react";
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
import { TopBar } from "../../components/ui/TopBar";
import { api } from "../../lib/api";

interface ProgressProps {
  patientId: number;
}

const tooltipStyle = {
  backgroundColor: "#111520",
  border: "1px solid rgba(148,163,184,0.08)",
  borderRadius: "8px",
  color: "#E2E8F0",
  fontSize: "12px",
};

export function PatientProgress({ patientId }: ProgressProps) {
  const [progress, setProgress] = useState<any>(null);
  const [patient, setPatient] = useState<any>(null);

  useEffect(() => {
    api.getPatientProgress(patientId).then(setProgress);
    api.getPatient(patientId).then(setPatient);
  }, [patientId]);

  if (!progress || !patient)
    return <div className="p-8 text-center text-muted">Loading...</div>;

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

  const chartData = progress.sessions.map((s: any, i: number) => ({
    session: i + 1,
    score: s.overall_score,
    stability: s.avg_stability,
    completion: s.completion_rate,
    date: new Date(s.started_at).toLocaleDateString(),
  }));

  const exerciseTypes = [...new Set(progress.sessions.map((s: any) => s.exercise_type).filter(Boolean))] as string[];
  const exerciseChartData = progress.sessions.map((s: any, i: number) => {
    const point: any = { session: i + 1 };
    for (const ex of exerciseTypes) {
      point[ex] = s.exercise_type === ex ? s.overall_score : null;
    }
    return point;
  });
  const hasExerciseData = progress.sessions.filter((s: any) => s.exercise_type).length >= 2;

  return (
    <div>
      <TopBar title="Progress" subtitle={`${progress.totalSessions} sessions total`} />
      <div className="p-6 max-w-4xl mx-auto space-y-5">
        <div className="grid grid-cols-3 gap-4">
          <Card>
            <p className="text-small text-muted">Current Score</p>
            <p className="text-h3 font-bold font-mono text-text">
              {progress.trends.score.current?.toFixed(0) ?? "—"}
            </p>
          </Card>
          <Card>
            <p className="text-small text-muted">Stability</p>
            <p className="text-h3 font-bold font-mono text-text">
              {progress.trends.stability.current?.toFixed(0) ?? "—"}
            </p>
          </Card>
          <Card>
            <p className="text-small text-muted">Assist Level</p>
            <p className="text-h3 font-bold font-mono text-text">
              {patient.assist_level}
            </p>
          </Card>
        </div>

        {chartData.length > 1 && (
          <>
            <Card>
              <h3 className="text-small font-medium text-muted mb-4">
                Session Score
              </h3>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="session" stroke="rgba(255,255,255,0.2)" fontSize={11} />
                  <YAxis domain={[0, 100]} stroke="rgba(255,255,255,0.2)" fontSize={11} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Line
                    type="monotone"
                    dataKey="score"
                    stroke="#1D4ED8"
                    strokeWidth={2}
                    dot={{ fill: "#1D4ED8", r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <h3 className="text-small font-medium text-muted mb-4">
                Stability & Completion
              </h3>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="session" stroke="rgba(255,255,255,0.2)" fontSize={11} />
                  <YAxis domain={[0, 100]} stroke="rgba(255,255,255,0.2)" fontSize={11} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Line
                    type="monotone"
                    dataKey="stability"
                    stroke="#1D4ED8"
                    strokeWidth={2}
                    name="Stability"
                  />
                  <Line
                    type="monotone"
                    dataKey="completion"
                    stroke="#22C55E"
                    strokeWidth={2}
                    name="Completion"
                  />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            {hasExerciseData && (
              <Card>
                <h3 className="text-small font-medium text-muted mb-4">
                  Score by Exercise
                </h3>
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
        )}

        {chartData.length <= 1 && (
          <Card>
            <p className="text-center text-muted py-8">
              Complete more sessions to see trends.
            </p>
          </Card>
        )}

        {/* Feedback */}
        <Card>
          <h3 className="text-small font-medium text-text mb-2">Feedback</h3>
          <p className="text-small text-muted leading-relaxed">
            {(() => {
              const score = progress.trends?.score?.current;
              const sessions = progress.totalSessions || 0;
              const stability = progress.trends?.stability?.current;
              if (sessions === 0) return "Start your first session to begin tracking your progress.";
              if (sessions === 1) return "Great start. Keep up regular sessions and you'll begin to see measurable improvement in your control and consistency.";
              if (score != null && score >= 80) return "Your performance is strong. Consistent scores above 80 show solid motor control. Focus on maintaining this level across different exercises.";
              if (score != null && score >= 60) return "You're making steady progress. Your scores are improving — keep focusing on smooth, deliberate movements during each rep.";
              if (stability != null && stability < 50) return "Try to keep your movements steady and controlled. Stability improves with practice — focus on holding each position firmly before releasing.";
              return "Every session builds on the last. Stay consistent and focus on completing each rep with control rather than speed.";
            })()}
          </p>
        </Card>

        {/* How We Measure */}
        <Card>
          <h3 className="text-small font-medium text-text mb-3">How We Measure Your Progress</h3>
          <div className="space-y-3">
            <div>
              <p className="text-small font-medium text-text">Score</p>
              <p className="text-small text-muted leading-relaxed">
                Your overall session score combines accuracy and completion. It reflects how well you performed the intended movements across all reps.
              </p>
            </div>
            <div>
              <p className="text-small font-medium text-text">Stability</p>
              <p className="text-small text-muted leading-relaxed">
                Measures how consistent your EMG signal is during each rep. Higher stability means your muscles are producing a clear, repeatable pattern — a sign of improving neuromuscular control.
              </p>
            </div>
            <div>
              <p className="text-small font-medium text-text">Completion</p>
              <p className="text-small text-muted leading-relaxed">
                The percentage of target reps you successfully completed. A rep counts as successful when the correct movement is detected and held for the required duration.
              </p>
            </div>
            <div>
              <p className="text-small font-medium text-text">Accuracy</p>
              <p className="text-small text-muted leading-relaxed">
                The ratio of successful reps to total attempts. Missed or incorrect gestures lower your accuracy. This improves as you develop more precise muscle activation patterns.
              </p>
            </div>
            <div>
              <p className="text-small font-medium text-text">Assist Level</p>
              <p className="text-small text-muted leading-relaxed">
                How much the device helps you complete movements. As your control improves, your therapist may reduce this — meaning you're doing more of the work yourself.
              </p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

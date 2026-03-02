import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { Badge } from "../../components/ui/Badge";
import { TopBar } from "../../components/ui/TopBar";
import { api } from "../../lib/api";

export function SessionSummary() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    if (sessionId) {
      api.getSessionSummary(Number(sessionId)).then(setData);
    }
  }, [sessionId]);

  if (!data)
    return <div className="p-8 text-center text-muted">Loading summary...</div>;

  const { session, reps, previous, recommendation } = data;

  const trend = (current: number | null, prev: number | null) => {
    if (current == null || prev == null) return null;
    const diff = current - prev;
    if (Math.abs(diff) < 1) return { label: "—", color: "text-muted" };
    return diff > 0
      ? { label: `+${diff.toFixed(1)}`, color: "text-success" }
      : { label: diff.toFixed(1), color: "text-danger" };
  };

  const scoreTrend = trend(session.overall_score, previous?.overall_score);
  const stabilityTrend = trend(session.avg_stability, previous?.avg_stability);
  const completionTrend = trend(session.completion_rate, previous?.completion_rate);

  return (
    <div>
      <TopBar title="Session Complete" />
      <div className="p-6 max-w-2xl mx-auto space-y-5">
        <div className="text-center py-4">
          <p className="text-h1 font-bold font-mono text-text">
            {session.overall_score?.toFixed(0) ?? "—"}
          </p>
          <p className="text-small text-muted mt-1">Overall Score</p>
          {scoreTrend && (
            <span className={`text-small font-mono font-medium ${scoreTrend.color}`}>
              {scoreTrend.label} vs last
            </span>
          )}
        </div>

        <div className="grid grid-cols-3 gap-4">
          <Card>
            <p className="text-small text-muted">Reps</p>
            <p className="text-h3 font-bold font-mono text-text">
              {reps.filter((r: any) => r.success).length}/{reps.length}
            </p>
          </Card>
          <Card>
            <p className="text-small text-muted">Stability</p>
            <p className="text-h3 font-bold font-mono text-text">
              {session.avg_stability?.toFixed(0) ?? "—"}
            </p>
            {stabilityTrend && (
              <span className={`text-small font-mono ${stabilityTrend.color}`}>
                {stabilityTrend.label}
              </span>
            )}
          </Card>
          <Card>
            <p className="text-small text-muted">Completion</p>
            <p className="text-h3 font-bold font-mono text-text">
              {session.completion_rate?.toFixed(0) ?? "—"}%
            </p>
            {completionTrend && (
              <span className={`text-small font-mono ${completionTrend.color}`}>
                {completionTrend.label}
              </span>
            )}
          </Card>
        </div>

        {recommendation && (
          <Card className="border-success/20">
            <Badge variant="success">Ready to Progress</Badge>
            <p className="mt-2 text-small text-muted">
              {recommendation.message}
            </p>
            <p className="text-small text-muted mt-1">
              Your therapist will review this.
            </p>
          </Card>
        )}

        <div className="flex gap-3">
          <Button
            variant="secondary"
            className="flex-1"
            onClick={() => navigate("/patient/progress")}
          >
            View Progress
          </Button>
          <Button className="flex-1" onClick={() => navigate("/patient/session/new")}>
            New Exercise
          </Button>
        </div>
      </div>
    </div>
  );
}

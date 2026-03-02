import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

interface LoginProps {
  onLogin: (pin: string) => Promise<any>;
  error: string | null;
}

export function Login({ onLogin, error }: LoginProps) {
  const navigate = useNavigate();
  const [role, setRole] = useState<"patient" | "therapist" | null>(null);
  const [pin, setPin] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    await onLogin(pin);
    setLoading(false);
  };

  // Role selection screen
  if (!role) {
    return (
      <div className="min-h-screen bg-bg flex flex-col items-center justify-center px-6">
        <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-6">
          Sign In
        </p>
        <h1 className="text-h2 font-medium text-text mb-10">
          Select your role.
        </h1>

        <div className="flex gap-4 w-full max-w-md">
          <button
            onClick={() => setRole("patient")}
            className="flex-1 py-6 rounded-lg border border-white/10 text-text font-mono text-body tracking-wide hover:bg-white/[0.04] hover:border-white/20 transition-colors"
          >
            Patient
          </button>
          <button
            onClick={() => setRole("therapist")}
            className="flex-1 py-6 rounded-lg border border-white/10 text-text font-mono text-body tracking-wide hover:bg-white/[0.04] hover:border-white/20 transition-colors"
          >
            Therapist
          </button>
        </div>

        <button
          onClick={() => navigate("/")}
          className="mt-8 text-small text-muted hover:text-text transition-colors font-mono"
        >
          Back
        </button>
      </div>
    );
  }

  // PIN entry screen
  return (
    <div className="min-h-screen bg-bg flex flex-col items-center justify-center px-6">
      <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-6">
        {role === "patient" ? "Patient Login" : "Therapist Login"}
      </p>
      <h1 className="text-h2 font-medium text-text mb-2">
        Enter your {role === "patient" ? "Patient ID" : "PIN"}.
      </h1>
      <p className="text-small text-muted mb-10">
        {role === "patient"
          ? "Your Patient ID was provided when you were registered."
          : "Use your therapist credentials to access the platform."}
      </p>

      <form onSubmit={handleSubmit} className="w-full max-w-xs space-y-4">
        <input
          type="password"
          inputMode="numeric"
          pattern="[0-9]*"
          maxLength={8}
          value={pin}
          onChange={(e) => setPin(e.target.value)}
          placeholder={role === "patient" ? "Patient ID" : "PIN"}
          className="w-full text-center text-2xl font-mono tracking-[0.3em] px-4 py-3 bg-white/[0.04] border border-border rounded-lg text-text placeholder-muted/40 focus:outline-none focus:border-white/20 transition-colors"
          autoFocus
        />
        {error && (
          <p className="text-danger text-small text-center">{error}</p>
        )}
        <button
          type="submit"
          disabled={loading || pin.length < 4}
          className="w-full px-6 py-3 rounded-lg border border-white/20 text-text font-mono text-body tracking-wide hover:bg-white/[0.06] transition-colors disabled:opacity-40"
        >
          {loading ? "Signing in..." : "Sign In"}
        </button>
      </form>

      <button
        onClick={() => { setRole(null); setPin(""); }}
        className="mt-8 text-small text-muted hover:text-text transition-colors font-mono"
      >
        Back
      </button>
    </div>
  );
}

import { useState, useCallback } from "react";
import { api } from "../lib/api";

interface AuthState {
  role: "patient" | "therapist" | null;
  id: number | null;
  name: string | null;
}

export function useAuth() {
  const [auth, setAuth] = useState<AuthState>(() => {
    const saved = localStorage.getItem("exohand_auth");
    return saved ? JSON.parse(saved) : { role: null, id: null, name: null };
  });
  const [error, setError] = useState<string | null>(null);

  const login = useCallback(async (pin: string) => {
    setError(null);
    try {
      const result = await api.login(pin);
      const state: AuthState = {
        role: result.role as "patient" | "therapist",
        id: result.id,
        name: result.name,
      };
      setAuth(state);
      localStorage.setItem("exohand_auth", JSON.stringify(state));
      return state;
    } catch (err: any) {
      setError(err.message);
      return null;
    }
  }, []);

  const logout = useCallback(() => {
    setAuth({ role: null, id: null, name: null });
    localStorage.removeItem("exohand_auth");
  }, []);

  const startSimulation = useCallback(async () => {
    try {
      const result = await api.startSimulation();
      const state: AuthState = {
        role: "patient",
        id: result.id,
        name: result.name,
      };
      setAuth(state);
      localStorage.setItem("exohand_auth", JSON.stringify(state));
    } catch (err: any) {
      setError(err.message || "Failed to start simulation.");
    }
  }, []);

  return { ...auth, error, login, logout, startSimulation };
}

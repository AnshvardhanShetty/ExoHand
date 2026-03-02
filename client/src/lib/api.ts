const BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

export const api = {
  // Auth
  login: (pin: string) =>
    request<{ role: string; id: number; name: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ pin }),
    }),

  // Simulation
  startSimulation: () =>
    request<{ role: string; id: number; name: string }>("/auth/simulation", {
      method: "POST",
    }),

  // Patients
  getPatient: (id: number) => request<any>(`/patients/${id}`),
  updatePatient: (id: number, data: any) =>
    request<any>(`/patients/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    }),
  getPatientSessions: (id: number) => request<any[]>(`/patients/${id}/sessions`),
  getPatientProgress: (id: number) => request<any>(`/patients/${id}/progress`),

  // Sessions
  startSession: (patientId: number, exercise?: any) =>
    request<any>("/sessions/start", {
      method: "POST",
      body: JSON.stringify({ patient_id: patientId, exercise }),
    }),
  endSession: (sessionId: number, exerciseDuration?: number) =>
    request<any>(`/sessions/${sessionId}/end`, {
      method: "POST",
      body: JSON.stringify({ exercise_duration: exerciseDuration }),
    }),
  getSessionSummary: (sessionId: number) =>
    request<any>(`/sessions/${sessionId}/summary`),
  recordRep: (sessionId: number, rep: any) =>
    request<any>(`/sessions/${sessionId}/reps`, {
      method: "POST",
      body: JSON.stringify(rep),
    }),

  // Therapist
  getTherapistPatients: () => request<any[]>("/therapist/patients"),
  getTherapistPatientDetail: (id: number) =>
    request<any>(`/therapist/patients/${id}`),
  updatePatientSettings: (id: number, settings: any) =>
    request<any>(`/therapist/patients/${id}/settings`, {
      method: "PUT",
      body: JSON.stringify(settings),
    }),
  approveRecommendation: (
    patientId: number,
    recommendationId: number,
    approved: boolean
  ) =>
    request<any>(`/therapist/patients/${patientId}/approve-recommendation`, {
      method: "POST",
      body: JSON.stringify({ recommendation_id: recommendationId, approved }),
    }),

  // Exercises
  getExercises: (patientId: number) =>
    request<any[]>(`/therapist/patients/${patientId}/exercises`),
  saveExercises: (patientId: number, exercises: any[]) =>
    request<any>(`/therapist/patients/${patientId}/exercises`, {
      method: "PUT",
      body: JSON.stringify({ exercises }),
    }),

  // Create patient
  createPatient: (data: { name: string; pin: string; description: string; assist_level: number; dob: string; hospital: string }) =>
    request<any>("/therapist/patients", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  // Delete patient
  deletePatient: (id: number) =>
    request<any>(`/therapist/patients/${id}`, { method: "DELETE" }),

  // Calibration
  startCalibration: (mode: "full" | "quick", patientId?: number) =>
    request<any>("/calibration/start", {
      method: "POST",
      body: JSON.stringify({ mode, patient_id: patientId }),
    }),
  stopCalibration: () =>
    request<any>("/calibration/stop", { method: "POST" }),
  calibrationPhaseReady: () =>
    request<any>("/calibration/phase-ready", { method: "POST" }),
  getCalibrationStatus: () =>
    request<any>("/calibration/status"),

  // Bridge status
  getBridgeStatus: () =>
    request<{ running: boolean; ready: boolean; error: string | null }>("/sessions/bridge-status"),
};

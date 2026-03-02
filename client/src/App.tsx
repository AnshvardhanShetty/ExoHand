import React from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import { useAuth } from "./hooks/useAuth";
import { Landing } from "./pages/Landing";
import { Login } from "./pages/Login";
import { Welcome } from "./pages/Welcome";
import { PatientLayout } from "./components/PatientLayout";
import { TherapistLayout } from "./components/TherapistLayout";
import { PatientHome } from "./pages/patient/Home";
import { ExerciseSelect } from "./pages/patient/ExerciseSelect";
import { Calibration } from "./pages/patient/Calibration";
import { PatientSession } from "./pages/patient/Session";
import { SessionSummary } from "./pages/patient/SessionSummary";
import { PatientProgress } from "./pages/patient/Progress";
import { TherapistDashboard } from "./pages/therapist/Dashboard";
import { AddPatient } from "./pages/therapist/AddPatient";
import { PatientDetail } from "./pages/therapist/PatientDetail";
import { Patients } from "./pages/therapist/Patients";
import { TherapistCalibration } from "./pages/therapist/TherapistCalibration";

export default function App() {
  const { role, id, name, error, login, logout, startSimulation } = useAuth();

  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route
          path="/"
          element={role ? <Navigate to={role === "patient" ? "/patient" : "/therapist"} replace /> : <Landing onSimulation={startSimulation} />}
        />
        <Route
          path="/login"
          element={role ? <Navigate to="/welcome" replace /> : <Login onLogin={login} error={error} />}
        />
        <Route
          path="/welcome"
          element={role ? <Welcome name={name || "User"} role={role} /> : <Navigate to="/" replace />}
        />

        {/* Patient routes */}
        {role === "patient" && id && (
          <Route
            element={
              <PatientLayout name={name || "Patient"} onLogout={logout} />
            }
          >
            <Route path="/patient" element={<PatientHome patientId={id} displayName={name || undefined} />} />
            <Route
              path="/patient/session/new"
              element={<ExerciseSelect patientId={id} />}
            />
            <Route
              path="/patient/calibration"
              element={<Calibration patientId={id} />}
            />
            <Route
              path="/patient/session"
              element={<PatientSession patientId={id} />}
            />
            <Route
              path="/patient/summary/:sessionId"
              element={<SessionSummary />}
            />
            <Route
              path="/patient/progress"
              element={<PatientProgress patientId={id} />}
            />
            <Route path="*" element={<Navigate to="/patient" replace />} />
          </Route>
        )}

        {/* Therapist routes */}
        {role === "therapist" && (
          <Route
            element={
              <TherapistLayout
                name={name || "Therapist"}
                onLogout={logout}
              />
            }
          >
            <Route path="/therapist" element={<TherapistDashboard therapistName={name || "Doctor"} />} />
            <Route path="/therapist/patients" element={<Patients />} />
            <Route path="/therapist/add-patient" element={<AddPatient />} />
            <Route
              path="/therapist/patient/:patientId"
              element={<PatientDetail />}
            />
            <Route path="*" element={<Navigate to="/therapist" replace />} />
          </Route>
        )}

        {/* Full-screen therapist calibration — outside layout so no sidebar */}
        {role === "therapist" && (
          <Route
            path="/therapist/patient/:patientId/calibration"
            element={<TherapistCalibration />}
          />
        )}

        {/* Catch-all for unauthenticated users */}
        {!role && <Route path="*" element={<Navigate to="/" replace />} />}
      </Routes>
    </BrowserRouter>
  );
}

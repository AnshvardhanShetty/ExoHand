import React, { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Calibration } from "../patient/Calibration";

export function TherapistCalibration() {
  const { patientId } = useParams();
  const navigate = useNavigate();

  return (
    <div className="h-screen bg-bg relative">
      {/* Logo in corner */}
      <div className="absolute top-5 left-5 z-10">
        <img src="/logo.png" alt="ExoHand" className="h-6 object-contain opacity-50" />
      </div>

      <Calibration
        patientId={Number(patientId)}
        therapistMode
        onComplete={() => {
          navigate(`/therapist/patient/${patientId}`);
        }}
        onCancel={() => {
          navigate(`/therapist/patient/${patientId}`);
        }}
      />
    </div>
  );
}

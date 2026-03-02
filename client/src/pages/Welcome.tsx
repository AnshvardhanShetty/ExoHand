import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";

interface WelcomeProps {
  name: string;
  role: "patient" | "therapist";
}

export function Welcome({ name, role }: WelcomeProps) {
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => {
      navigate(role === "patient" ? "/patient" : "/therapist", { replace: true });
    }, 2500);
    return () => clearTimeout(timer);
  }, [navigate, role]);

  return (
    <div className="min-h-screen bg-bg flex flex-col items-center justify-center px-6">
      <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-6">
        Welcome
      </p>
      <h1 className="text-h1 font-bold text-text font-mono tracking-wide">
        {name}
      </h1>
      <div className="mt-8">
        <div className="relative w-12 h-12 mx-auto">
          <div className="absolute inset-0 rounded-full border-2 border-border" />
          <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-white animate-spin" />
        </div>
      </div>
    </div>
  );
}

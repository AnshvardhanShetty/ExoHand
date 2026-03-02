import React from "react";
import { Outlet } from "react-router-dom";
import { Sidebar } from "./ui/Sidebar";
import { Button } from "./ui/Button";

interface TherapistLayoutProps {
  name: string;
  onLogout: () => void;
}

const navItems = [
  { to: "/therapist", label: "Dashboard", icon: "\u2630" },
  { to: "/therapist/patients", label: "Patients", icon: "\u2302" },
  { to: "/therapist/add-patient", label: "Add Patient", icon: "+" },
];

export function TherapistLayout({ name, onLogout }: TherapistLayoutProps) {
  return (
    <div className="flex h-screen bg-bg">
      <Sidebar
        title="ExoHand Therapist"
        items={navItems}
        footer={
          <div className="space-y-2">
            <p className="text-small text-muted truncate">{name}</p>
            <Button variant="ghost" size="sm" onClick={onLogout} className="w-full">
              Sign Out
            </Button>
          </div>
        }
      />
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}

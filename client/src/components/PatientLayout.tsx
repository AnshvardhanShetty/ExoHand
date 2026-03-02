import React from "react";
import { Outlet, useLocation } from "react-router-dom";
import { Sidebar } from "./ui/Sidebar";
import { Button } from "./ui/Button";

interface PatientLayoutProps {
  name: string;
  onLogout: () => void;
}

const navItems = [
  { to: "/patient", label: "Home", icon: "\u2302" },
  { to: "/patient/calibration", label: "New Session", icon: "\u25B6" },
  { to: "/patient/progress", label: "Progress", icon: "\u2191" },
];

export function PatientLayout({ name, onLogout }: PatientLayoutProps) {
  const { pathname, search } = useLocation();
  // Hide sidebar only during active session or active calibration (signalled by ?active=1)
  const hideSidebar =
    pathname === "/patient/session" ||
    (pathname === "/patient/calibration" && search.includes("active=1"));

  return (
    <div className="flex h-screen bg-bg">
      {!hideSidebar ? (
        <Sidebar
          title="ExoHand"
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
      ) : (
        <div className="absolute bottom-4 left-4 z-10">
          <img src="/logo.png" alt="ExoHand" className="h-6 object-contain opacity-50" />
        </div>
      )}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}

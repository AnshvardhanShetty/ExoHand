import React from "react";
import { NavLink } from "react-router-dom";

interface NavItem {
  to: string;
  label: string;
  icon: string;
}

interface SidebarProps {
  title: string;
  items: NavItem[];
  footer?: React.ReactNode;
}

export function Sidebar({ title, items, footer }: SidebarProps) {
  return (
    <aside className="w-56 h-screen bg-panel2 border-r border-border flex flex-col">
      <div className="px-5 py-4">
        <a href="/" className="block">
          <img src="/logo.png" alt="ExoHand" className="w-24 object-contain" />
        </a>
      </div>
      <nav className="flex-1 px-3 mt-2">
        {items.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/patient" || item.to === "/therapist"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-small font-medium mb-0.5 transition-colors ${
                isActive
                  ? "bg-accent/15 text-text border-l-2 border-accent"
                  : "text-muted hover:text-text hover:bg-white/[0.03]"
              }`
            }
          >
            <span className="text-sm">{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>
      {footer && <div className="p-4 border-t border-border">{footer}</div>}
    </aside>
  );
}

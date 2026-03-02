import React from "react";

interface TopBarProps {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}

export function TopBar({ title, subtitle, actions }: TopBarProps) {
  return (
    <header className="h-14 bg-panel border-b border-border flex items-center justify-between px-6">
      <div>
        <h2 className="text-body font-semibold text-text font-mono tracking-wide">{title}</h2>
        {subtitle && (
          <p className="text-small text-muted">{subtitle}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-3">{actions}</div>}
    </header>
  );
}

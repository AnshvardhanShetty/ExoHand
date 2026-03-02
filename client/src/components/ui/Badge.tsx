import React from "react";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "default" | "success" | "warning" | "danger" | "info";
  className?: string;
}

const variants = {
  default: "bg-white/[0.06] text-muted",
  success: "bg-success/10 text-success",
  warning: "bg-warn/10 text-warn",
  danger: "bg-danger/10 text-danger",
  info: "bg-accent/10 text-accent-light",
};

export function Badge({
  children,
  variant = "default",
  className = "",
}: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-small font-medium ${variants[variant]} ${className}`}
    >
      {children}
    </span>
  );
}

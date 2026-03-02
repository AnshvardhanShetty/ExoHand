import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "sm" | "md" | "lg";
}

const variants = {
  primary:
    "bg-transparent text-white border border-white/20 hover:bg-white/[0.06] active:bg-white/[0.10]",
  secondary:
    "bg-transparent text-text border border-[rgba(148,163,184,0.15)] hover:bg-[rgba(148,163,184,0.05)] active:bg-[rgba(148,163,184,0.08)]",
  danger: "bg-danger text-white hover:bg-danger/90 active:bg-danger/80",
  ghost: "text-muted hover:text-text hover:bg-white/[0.04] active:bg-white/[0.06]",
};

const sizes = {
  sm: "px-3 py-1.5 text-small",
  md: "px-4 py-2 text-small font-medium",
  lg: "px-5 py-2.5 text-body font-medium",
};

export function Button({
  variant = "primary",
  size = "md",
  className = "",
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={`rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${variants[variant]} ${sizes[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

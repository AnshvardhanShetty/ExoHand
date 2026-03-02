import React from "react";

interface CardProps {
  children: React.ReactNode;
  className?: string;
  padding?: boolean;
}

export function Card({ children, className = "", padding = true }: CardProps) {
  return (
    <div
      className={`bg-panel rounded-xl border border-border ${padding ? "p-5" : ""} ${className}`}
    >
      {children}
    </div>
  );
}

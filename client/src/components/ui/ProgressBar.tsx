import React from "react";

interface ProgressBarProps {
  value: number; // 0-100
  label?: string;
  color?: string;
  size?: "sm" | "md" | "lg";
}

const sizes = {
  sm: "h-1",
  md: "h-1.5",
  lg: "h-2.5",
};

export function ProgressBar({
  value,
  label,
  color = "bg-white",
  size = "md",
}: ProgressBarProps) {
  const clamped = Math.max(0, Math.min(100, value));

  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between text-small mb-1.5">
          <span className="text-muted">{label}</span>
          <span className="font-mono font-medium text-text">
            {Math.round(clamped)}%
          </span>
        </div>
      )}
      <div className={`w-full ${sizes[size]} bg-white/[0.06] rounded-full overflow-hidden`}>
        <div
          className={`${sizes[size]} ${color} rounded-full transition-all duration-500`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

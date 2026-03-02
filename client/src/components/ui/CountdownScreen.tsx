import React, { useEffect, useRef, useState } from "react";

interface CountdownScreenProps {
  title: string;
  subtitle?: string;
  seconds?: number;
  onComplete: () => void;
}

export function CountdownScreen({
  title,
  subtitle,
  seconds = 3,
  onComplete,
}: CountdownScreenProps) {
  const [count, setCount] = useState(seconds);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    if (count <= 0) {
      onCompleteRef.current();
      return;
    }
    const id = setTimeout(() => setCount((c) => c - 1), 1000);
    return () => clearTimeout(id);
  }, [count]);

  return (
    <div className="h-full flex items-center justify-center bg-bg">
      <div className="text-center space-y-6">
        <div>
          <p className="text-h3 font-semibold text-text">{title}</p>
          {subtitle && (
            <p className="text-small text-muted mt-2">{subtitle}</p>
          )}
        </div>
        {count > 0 && (
          <p
            key={count}
            className="text-[96px] font-bold font-mono text-text leading-none animate-countdown"
          >
            {count}
          </p>
        )}
      </div>

      <style>{`
        @keyframes countdownPop {
          0% { transform: scale(1.5); opacity: 0; }
          40% { transform: scale(1); opacity: 1; }
          100% { transform: scale(1); opacity: 1; }
        }
        .animate-countdown {
          animation: countdownPop 0.6s ease-out both;
        }
      `}</style>
    </div>
  );
}

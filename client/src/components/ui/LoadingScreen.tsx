import React from "react";

interface LoadingScreenProps {
  message: string;
  submessage?: string;
}

export function LoadingScreen({ message, submessage }: LoadingScreenProps) {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center space-y-6">
        {/* Circular spinner */}
        <div className="relative w-16 h-16 mx-auto">
          <div
            className="absolute inset-0 rounded-full border-2 border-border"
          />
          <div
            className="absolute inset-0 rounded-full border-2 border-transparent border-t-white animate-spin"
          />
        </div>
        <div>
          <p className="text-h3 font-semibold text-text">{message}</p>
          {submessage && (
            <p className="text-small text-muted mt-2">{submessage}</p>
          )}
        </div>
      </div>
    </div>
  );
}

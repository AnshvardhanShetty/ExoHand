import React from "react";

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
}

export function Modal({ open, onClose, title, children }: ModalProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative bg-panel border border-border rounded-xl shadow-2xl max-w-lg w-full mx-4 p-6">
        {title && (
          <h2 className="text-h3 font-semibold text-text mb-4">{title}</h2>
        )}
        {children}
      </div>
    </div>
  );
}

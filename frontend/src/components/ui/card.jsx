import * as React from "react";

export function Card({ className = "", children, ...props }) {
  return (
    <div
      className={`rounded-xl border border-[#30363d] bg-[#0d1117] text-[#e6edf3] shadow-sm ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({ className = "", children, ...props }) {
  return (
    <div
      className={`flex flex-col space-y-1.5 p-4 border-b border-[#30363d] ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardContent({ className = "", children, ...props }) {
  return (
    <div className={`p-4 ${className}`} {...props}>
      {children}
    </div>
  );
}

export function CardFooter({ className = "", children, ...props }) {
  return (
    <div
      className={`flex items-center p-4 border-t border-[#30363d] ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

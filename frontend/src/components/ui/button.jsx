import * as React from "react";

export function Button({
  children,
  className = "",
  variant = "default",
  size = "sm",
  ...props
}) {
  const base =
    "inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#58a6ff] disabled:opacity-50 disabled:pointer-events-none";
  const variants = {
    default: "bg-[#21262d] hover:bg-[#30363d] text-[#e6edf3]",
    accent: "bg-[#238636] hover:bg-[#2ea043] text-white",
    outline: "border border-[#30363d] hover:bg-[#161b22]",
  };
  const sizes = {
    sm: "h-7 px-2 text-[10px]",
    md: "h-8 px-3 text-xs",
    lg: "h-9 px-4 text-sm",
  };

  return (
    <button
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

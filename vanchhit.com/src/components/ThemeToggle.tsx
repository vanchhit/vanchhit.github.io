import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "@/lib/theme";

export function ThemeToggle() {
  const { mode, toggle } = useTheme();
  const Icon = mode === "system" ? Monitor : mode === "dark" ? Moon : Sun;
  const label =
    mode === "system" ? "System theme (click for light)" :
    mode === "light" ? "Light theme (click for dark)" :
    "Dark theme (click for system)";
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={label}
      title={label}
      className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
    >
      <Icon size={16} />
    </button>
  );
}

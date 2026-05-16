import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export type ThemeMode = "light" | "dark" | "system";

type Ctx = {
  mode: ThemeMode;
  resolved: "light" | "dark";
  setMode: (m: ThemeMode) => void;
  toggle: () => void;
};

const ThemeContext = createContext<Ctx | null>(null);

const STORAGE_KEY = "vanchhit-theme";

function getSystem(): "light" | "dark" {
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyClass(resolved: "light" | "dark") {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  if (resolved === "dark") root.classList.add("dark");
  else root.classList.remove("dark");
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<ThemeMode>("system");
  const [resolved, setResolved] = useState<"light" | "dark">("light");

  // Load persisted preference once on mount
  useEffect(() => {
    const stored = (typeof localStorage !== "undefined"
      ? (localStorage.getItem(STORAGE_KEY) as ThemeMode | null)
      : null) ?? "system";
    setModeState(stored);
  }, []);

  // Recompute resolved theme on mode change or system pref change
  useEffect(() => {
    const compute = () => (mode === "system" ? getSystem() : mode);
    const next = compute();
    setResolved(next);
    applyClass(next);

    if (mode === "system" && typeof window !== "undefined") {
      const mq = window.matchMedia("(prefers-color-scheme: dark)");
      const handler = () => {
        const r = getSystem();
        setResolved(r);
        applyClass(r);
      };
      mq.addEventListener("change", handler);
      return () => mq.removeEventListener("change", handler);
    }
  }, [mode]);

  const setMode = (m: ThemeMode) => {
    setModeState(m);
    if (typeof localStorage !== "undefined") localStorage.setItem(STORAGE_KEY, m);
  };

  const toggle = () => {
    // Cycle: system -> light -> dark -> system
    setMode(mode === "system" ? "light" : mode === "light" ? "dark" : "system");
  };

  return (
    <ThemeContext.Provider value={{ mode, resolved, setMode, toggle }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}

// Inline script to apply the saved theme before React hydrates,
// preventing a light/dark flash on first paint.
export const themeBootScript = `
(function(){try{
  var k='${STORAGE_KEY}';
  var s=localStorage.getItem(k)||'system';
  var d=s==='dark'||(s==='system'&&window.matchMedia('(prefers-color-scheme: dark)').matches);
  if(d)document.documentElement.classList.add('dark');
}catch(e){}})();
`;

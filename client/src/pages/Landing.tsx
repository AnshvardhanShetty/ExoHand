import React, { useEffect, useRef, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";

/* ── Scroll-reveal hook ── */
function useReveal<T extends HTMLElement>() {
  const ref = useRef<T>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.classList.add("revealed");
          observer.unobserve(el);
        }
      },
      { threshold: 0.15 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return ref;
}

function Reveal({ children, className = "", delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  const ref = useReveal<HTMLDivElement>();
  return (
    <div
      ref={ref}
      className={`reveal-up ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

/* ── Full-page scroll hook ── */
function useFullPageScroll(sectionCount: number) {
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    let index = 0;
    let cooldown = false;
    let touchStartY = 0;

    const go = (dir: number) => {
      if (cooldown) return;
      const next = index + dir;
      if (next < 0 || next >= sectionCount) return;
      cooldown = true;
      index = next;
      setCurrent(next);
      // Single cooldown covers animation (600ms) + trackpad inertia (~400ms)
      setTimeout(() => { cooldown = false; }, 1000);
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (Math.abs(e.deltaY) < 15) return;
      go(e.deltaY > 0 ? 1 : -1);
    };

    const onTouchStart = (e: TouchEvent) => { touchStartY = e.touches[0].clientY; };
    const onTouchEnd = (e: TouchEvent) => {
      const d = touchStartY - e.changedTouches[0].clientY;
      if (Math.abs(d) > 50) go(d > 0 ? 1 : -1);
    };

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown" || e.key === " ") { e.preventDefault(); go(1); }
      if (e.key === "ArrowUp") { e.preventDefault(); go(-1); }
    };

    window.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("touchstart", onTouchStart, { passive: true });
    window.addEventListener("touchend", onTouchEnd, { passive: true });
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("wheel", onWheel);
      window.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchend", onTouchEnd);
      window.removeEventListener("keydown", onKey);
    };
  }, [sectionCount]);

  return current;
}

/* ── Data ── */
const technologyCards = [
  {
    number: "01",
    title: "EMG Acquisition",
    body: "Four-channel surface electromyography sampled at 1 kHz. Raw signal preprocessing with bandpass filtering and feature extraction in real time.",
  },
  {
    number: "02",
    title: "Neural Classification",
    body: "Patient-calibrated convolutional model classifies intent from EMG features. Online fine-tuning adapts to signal drift across sessions.",
  },
  {
    number: "03",
    title: "Motor Execution",
    body: "Classified intent maps to graded assist profiles. The motor controller interpolates target angles with configurable gain and cooldown constraints.",
  },
];

const clinicalSteps = [
  { title: "Calibration", body: "Per-patient EMG baseline and gesture model training. Full 66-trial protocol or quick session recalibration." },
  { title: "Exercise Selection", body: "Patient selects from close, open, hold, and combined movement protocols. Configurable reps, angles, and hold durations." },
  { title: "Guided Session", body: "Real-time assist with locked state transitions, rep counting, and stability tracking." },
  { title: "Analytics", body: "Per-session and per-exercise scoring. Longitudinal trend analysis across accuracy, stability, and completion." },
  { title: "Adaptation", body: "Model recalibration and assist-level recommendations driven by longitudinal performance data." },
];

const metrics = [
  { value: "95.9%", label: "Classification Accuracy" },
  { value: "<200ms", label: "Intent-to-Motion Latency" },
  { value: "98.8%", label: "Movement Detection Accuracy" },
];

const SECTION_COUNT = 6;

interface LandingProps {
  onSimulation?: () => void | Promise<void>;
}

export function Landing({ onSimulation }: LandingProps) {
  const navigate = useNavigate();
  const current = useFullPageScroll(SECTION_COUNT);

  return (
    <div className="h-screen overflow-hidden bg-bg">

      {/* ── Fixed top nav ── */}
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 py-5">
        <img src="/logo.png" alt="ExoHand" className="h-8 object-contain cursor-pointer" onClick={() => window.location.reload()} />
        <Button size="md" onClick={() => navigate("/login")}>
          Login
        </Button>
      </nav>

      {/* ── Sliding container ── */}
      <div
        className="transition-transform duration-[600ms] ease-[cubic-bezier(0.16,1,0.3,1)]"
        style={{ transform: `translateY(-${current * 100}vh)` }}
      >

        {/* ── Hero ── */}
        <section className="relative h-screen flex flex-col items-center justify-center px-6 text-center">
          <Reveal>
            <h1 className="text-[clamp(1.8rem,4.5vw,3rem)] font-medium text-text tracking-[0.35em] uppercase pl-[0.45em]">
              E<span className="text-muted/40">·</span>X<span className="text-muted/40">·</span>O<span className="text-muted/40">·</span>H<span className="text-muted/40">·</span>A<span className="text-muted/40">·</span>N<span className="text-muted/40">·</span>D
            </h1>
          </Reveal>

          <div className="absolute bottom-10 text-muted/30 animate-bounce">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </section>

        {/* ── Technology ── */}
        <section className="h-screen flex flex-col justify-center px-6">
          <div className="max-w-5xl mx-auto w-full">
            <Reveal>
              <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-4">
                Technology
              </p>
              <h2 className="text-h2 font-medium text-text mb-12 max-w-xl">
                From surface signal to motor response in under 200 milliseconds.
              </h2>
            </Reveal>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {technologyCards.map((card, i) => (
                <Reveal key={card.title} delay={i * 100}>
                  <Card className="flex flex-col h-full">
                    <span className="font-mono text-[11px] text-muted/30 mb-4">{card.number}</span>
                    <h3 className="font-mono text-small uppercase tracking-wider text-text mb-3">
                      {card.title}
                    </h3>
                    <p className="text-small text-muted leading-relaxed">{card.body}</p>
                  </Card>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        {/* ── Clinical Flow ── */}
        <section className="h-screen flex flex-col justify-center px-6">
          <div className="max-w-5xl mx-auto w-full">
            <Reveal>
              <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-4">
                Protocol
              </p>
              <h2 className="text-h2 font-medium text-text mb-12 max-w-xl">
                Five-stage session architecture.
              </h2>
            </Reveal>

            <div className="space-y-0">
              {clinicalSteps.map((step, i) => (
                <Reveal key={step.title} delay={i * 80}>
                  <div className="grid grid-cols-[3rem_1fr] md:grid-cols-[3rem_12rem_1fr] gap-x-6 py-5 border-t border-border items-baseline">
                    <span className="font-mono text-small text-muted/30">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <h3 className="font-mono text-small uppercase tracking-wider text-text">
                      {step.title}
                    </h3>
                    <p className="text-small text-muted col-start-2 md:col-start-3 mt-1 md:mt-0">{step.body}</p>
                  </div>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        {/* ── Metrics ── */}
        <section className="h-screen flex flex-col justify-center px-6">
          <div className="max-w-5xl mx-auto w-full">
            <Reveal>
              <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-4">
                Performance
              </p>
              <h2 className="text-h2 font-medium text-text mb-12 max-w-xl">
                Measurable recovery outcomes.
              </h2>
            </Reveal>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              {metrics.map((m, i) => (
                <Reveal key={m.label} delay={i * 120}>
                  <div className="py-4">
                    <p className="text-[clamp(2.5rem,5vw,3.5rem)] font-medium font-mono text-text leading-none mb-3">
                      {m.value}
                    </p>
                    <p className="text-small text-muted">{m.label}</p>
                  </div>
                </Reveal>
              ))}
            </div>
            <Reveal delay={400}>
              <p className="text-[11px] text-muted/40 font-mono">
                Representative metrics from internal pilot data.
              </p>
            </Reveal>
          </div>
        </section>

        {/* ── Simulation ── */}
        <section className="h-screen flex flex-col justify-center px-6">
          <div className="max-w-5xl mx-auto w-full">
            <Reveal>
              <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-4">
                Simulation
              </p>
              <h2 className="text-h2 font-medium text-text mb-6 max-w-xl">
                Experience the system firsthand.
              </h2>
              <p className="text-small text-muted max-w-2xl leading-relaxed mb-10">
                Run the exact same calibration pipeline, real EMG datasets, and exercise sessions used in the clinical flow — identical to a live patient session, just without a physical device. No account required.
              </p>
            </Reveal>
            <Reveal delay={200}>
              <Button
                size="md"
                onClick={async () => {
                  await onSimulation?.();
                  navigate("/welcome");
                }}
              >
                Try Simulation
              </Button>
            </Reveal>
          </div>
        </section>

        {/* ── About + Footer ── */}
        <section className="h-screen flex flex-col justify-center px-6">
          <div className="max-w-5xl mx-auto w-full">
            <Reveal>
              <p className="font-mono text-[11px] tracking-[0.3em] uppercase text-muted/50 mb-4">
                About
              </p>
              <h2 className="text-h2 font-medium text-text mb-6 max-w-xl">
                Restoring movement through signal intelligence.
              </h2>
              <p className="text-small text-muted max-w-2xl leading-relaxed mb-20">
                ExoHand operates at the intersection of motor neuroscience, embedded control, and machine learning. Surface EMG captures intent. A patient-calibrated neural network classifies it. A graded motor controller executes it. Therapists monitor progress, tune assist levels, and track safety events through a clinical dashboard. Each session generates data that refines the next.
              </p>
            </Reveal>

            <Reveal delay={200}>
              <div className="border-t border-border pt-8 flex items-center justify-between">
                <p className="text-[11px] text-muted/30 font-mono">&copy; 2026 ExoHand</p>
                <Button size="sm" variant="ghost" onClick={() => navigate("/login")}>
                  Access Platform
                </Button>
              </div>
            </Reveal>
          </div>
        </section>

      </div>
    </div>
  );
}

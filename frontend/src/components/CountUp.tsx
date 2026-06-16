import { useEffect, useRef, useState } from "react";

export function CountUp({
  to,
  decimals = 4,
  duration = 1400,
  delay = 0,
}: {
  to: number;
  decimals?: number;
  duration?: number;
  delay?: number;
}) {
  const [val, setVal] = useState(0);
  const raf = useRef<number>(0);

  useEffect(() => {
    let start: number | null = null;
    const tick = (ts: number) => {
      if (!start) start = ts;
      const t = Math.min((ts - start) / duration, 1);
      const eased = t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
      setVal(to * eased);
      if (t < 1) raf.current = requestAnimationFrame(tick);
    };
    const id = setTimeout(() => {
      raf.current = requestAnimationFrame(tick);
    }, delay);
    return () => {
      clearTimeout(id);
      cancelAnimationFrame(raf.current);
    };
  }, [to, duration, delay]);

  return <>{val.toFixed(decimals)}</>;
}

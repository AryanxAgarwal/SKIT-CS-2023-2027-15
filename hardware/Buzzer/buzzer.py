"""Buzzer driver for Raspberry Pi.

Hardware (active buzzer, recommended):
    GPIO18 (physical pin 12) --[1k ohm]--> base of NPN (2N2222 / BC547)
    NPN emitter  -> GND (pin 6)
    NPN collector-> buzzer negative (-)
    buzzer positive (+) -> 5V (pin 2)   (or 3.3V for 3.3V buzzers)

Features
    * on() / off()            - direct control
    * beep(n)                 - blocking short beeps
    * play(pattern)           - non-blocking background pattern (own thread)
    * stop()                  - immediate silence, thread-safe
    * self_test()             - quick hardware check at start-up
    * simulation mode         - runs on a laptop with no GPIO (for tests/dev)
    * safety auto-silence     - never buzzes continuously longer than limit
    * context manager         - always releases the pin on exit
"""
import logging
import threading
import time

log = logging.getLogger(__name__)

# name -> list of (on_seconds, off_seconds); pattern repeats until stop()
PATTERNS = {
    "warning": [(0.15, 0.85)],                 # short tick every second
    "yawn":    [(0.10, 0.10), (0.10, 0.70)],   # double chirp
    "drowsy":  [(0.30, 0.20)],                 # fast beeping
    "alarm":   [(0.70, 0.10)],                 # almost continuous
}


class BuzzerController:
    def __init__(self, pin=18, active_high=True, simulate=False, max_continuous_s=30.0):
        self.pin = pin
        self.max_continuous_s = max_continuous_s
        self._lock = threading.RLock()      # protects the GPIO pin state (on/off)
        self._ctl = threading.RLock()       # serialises play()/stop(); never taken by the pattern thread
        self._stop_evt = threading.Event()
        self._thread = None
        self._current_pattern = None
        self._is_on = False
        self._on_since = None
        self._device = None
        self.simulated = simulate

        if not simulate:
            try:
                from gpiozero import Buzzer
                self._device = Buzzer(pin, active_high=active_high, initial_value=False)
                log.info("Buzzer ready on BCM GPIO%d", pin)
            except Exception as exc:  # no GPIO available (PC, missing lib, permission)
                log.warning("GPIO unavailable (%s) -> buzzer SIMULATION mode", exc)
                self.simulated = True
        if self.simulated:
            log.info("Buzzer running in simulation mode (no real GPIO)")

    # ------------------------------------------------------------------ #
    # Low-level control
    # ------------------------------------------------------------------ #
    def on(self):
        """Switch buzzer ON immediately."""
        with self._lock:
            if self._is_on:
                return
            if self._device is not None:
                self._device.on()
            self._is_on = True
            self._on_since = time.monotonic()
            log.debug("Buzzer ON")

    def off(self):
        """Switch buzzer OFF immediately."""
        with self._lock:
            if self._device is not None:
                self._device.off()
            if self._is_on:
                log.debug("Buzzer OFF")
            self._is_on = False
            self._on_since = None

    @property
    def is_on(self):
        return self._is_on

    @property
    def current_pattern(self):
        return self._current_pattern

    # ------------------------------------------------------------------ #
    # Blocking helper
    # ------------------------------------------------------------------ #
    def beep(self, times=1, on_time=0.2, off_time=0.2):
        """Blocking beeps - handy for tests and start-up confirmation."""
        for i in range(times):
            self.on()
            time.sleep(on_time)
            self.off()
            if i < times - 1:
                time.sleep(off_time)

    def self_test(self):
        """Three short beeps at start-up so the driver knows the alarm works."""
        log.info("Buzzer self-test")
        self.beep(times=3, on_time=0.1, off_time=0.1)

    # ------------------------------------------------------------------ #
    # Non-blocking patterns
    # ------------------------------------------------------------------ #
    def play(self, pattern):
        """Start a named pattern in the background. Restarts only if changed."""
        if pattern not in PATTERNS:
            raise ValueError(f"Unknown pattern '{pattern}'. Choose from {list(PATTERNS)}")
        with self._ctl:
            if self._current_pattern == pattern and self._thread and self._thread.is_alive():
                return
            self._halt_thread()
            self._stop_evt.clear()
            self._current_pattern = pattern
            self._thread = threading.Thread(
                target=self._run_pattern, args=(PATTERNS[pattern],),
                name=f"buzzer-{pattern}", daemon=True)
            self._thread.start()
            log.info("Buzzer pattern -> %s", pattern)

    def stop(self):
        """Stop any pattern and make sure the buzzer is silent."""
        with self._ctl:
            self._halt_thread()
            self._current_pattern = None
            self.off()

    def _halt_thread(self):
        self._stop_evt.set()
        t = self._thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.5)
        self._thread = None

    def _run_pattern(self, steps):
        started = time.monotonic()
        try:
            while not self._stop_evt.is_set():
                for on_s, off_s in steps:
                    if time.monotonic() - started > self.max_continuous_s:
                        log.warning("Max continuous buzz time reached - silencing")
                        return
                    self.on()
                    if self._stop_evt.wait(on_s):
                        return
                    self.off()
                    if off_s and self._stop_evt.wait(off_s):
                        return
        finally:
            self.off()

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def close(self):
        self.stop()
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
        log.info("Buzzer released")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    """Command-line control:
        python buzzer.py --test              hardware self-test + all patterns
        python buzzer.py --on                buzzer ON (Ctrl+C to turn OFF)
        python buzzer.py --off               buzzer OFF
        python buzzer.py --pattern alarm     play a pattern for 5 s
        python buzzer.py --pin 17 --test     use another BCM pin
        python buzzer.py --simulate --test   no hardware needed
    """
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pin", type=int, default=18, help="BCM GPIO pin (default 18 = physical pin 12)")
    ap.add_argument("--simulate", action="store_true", help="run without GPIO hardware")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--on", action="store_true")
    ap.add_argument("--off", action="store_true")
    ap.add_argument("--pattern", choices=list(PATTERNS))
    ap.add_argument("--seconds", type=float, default=5.0, help="duration for --pattern")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    with BuzzerController(pin=a.pin, simulate=a.simulate) as bz:
        if a.off:
            bz.off()
            print("Buzzer OFF")
        elif a.on:
            bz.on()
            print("Buzzer ON - press Ctrl+C to turn it off")
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
        elif a.pattern:
            bz.play(a.pattern)
            time.sleep(a.seconds)
        else:  # default / --test
            bz.self_test()
            for name in PATTERNS:
                print(f"Pattern: {name}")
                bz.play(name)
                time.sleep(3)
                bz.stop()
                time.sleep(0.5)
            print("ON for 1 s")
            bz.on(); time.sleep(1); bz.off()
            print("Test complete")


if __name__ == "__main__":
    main()

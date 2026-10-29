"""Buzzer tests.

Run (no hardware needed, uses simulation mode):
    python test_buzzer.py            # plain Python, no extra packages
    pytest -q test_buzzer.py         # if pytest is installed

Hardware test on the Raspberry Pi (you should HEAR the buzzer):
    python test_buzzer.py --hardware
"""
import sys
import time

from buzzer import PATTERNS, BuzzerController


def make():
    return BuzzerController(pin=18, simulate=True)


def test_starts_off():
    b = make()
    assert not b.is_on
    b.close()


def test_on_off():
    b = make()
    b.on()
    assert b.is_on
    b.off()
    assert not b.is_on
    b.close()


def test_beep_ends_off():
    b = make()
    b.beep(times=2, on_time=0.02, off_time=0.02)
    assert not b.is_on
    b.close()


def test_pattern_starts_and_stops():
    b = make()
    b.play("drowsy")
    time.sleep(0.1)
    assert b.current_pattern == "drowsy"
    b.stop()
    assert not b.is_on and b.current_pattern is None
    b.close()


def test_same_pattern_not_restarted():
    b = make()
    b.play("warning")
    first = b._thread
    b.play("warning")
    assert b._thread is first
    b.close()


def test_switch_pattern():
    b = make()
    b.play("warning")
    b.play("alarm")
    assert b.current_pattern == "alarm"
    b.close()


def test_all_patterns_play():
    b = make()
    for name in PATTERNS:
        b.play(name)
        time.sleep(0.05)
        assert b.current_pattern == name
    b.close()


def test_unknown_pattern_raises():
    b = make()
    try:
        b.play("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError expected")
    finally:
        b.close()


def test_auto_silence_after_max_time():
    b = BuzzerController(pin=18, simulate=True, max_continuous_s=0.3)
    b.play("alarm")
    time.sleep(0.7)
    assert not b.is_on
    b.close()


def test_context_manager_releases_buzzer():
    with make() as b:
        b.on()
    assert not b.is_on


def test_stop_is_fast():
    b = make()
    b.play("alarm")
    t = time.monotonic()
    b.stop()
    assert time.monotonic() - t < 1.0
    b.close()


def hardware_test():
    print("HARDWARE TEST - listen for the buzzer")
    with BuzzerController(pin=18) as bz:
        if bz.simulated:
            print("GPIO not available -> nothing audible. Run this on the Pi.")
            return 1
        input("Press Enter: 3 short beeps ")
        bz.self_test()
        for name in PATTERNS:
            input(f"Press Enter: pattern '{name}' for 3 s ")
            bz.play(name)
            time.sleep(3)
            bz.stop()
        print("If you heard every sound, the buzzer wiring is correct.")
    return 0


if __name__ == "__main__":
    if "--hardware" in sys.argv:
        sys.exit(hardware_test())
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:
            failed += 1
            print("FAIL", name, repr(exc))
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

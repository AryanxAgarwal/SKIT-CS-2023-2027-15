# Buzzer Module - Driver Drowsiness Detection (Raspberry Pi)

## 1. About the buzzer
The buzzer is the **alert output** of the system. When the software decides the driver is
drowsy (eyes closed too long, yawning, face lost), it makes the buzzer sound so the driver wakes up.

We use an **active buzzer**: it has a built-in oscillator, so it beeps whenever its input is HIGH
(3.3 V). No tone generation is needed. (A *passive* buzzer needs PWM and is not used here.)

## 2. Wiring (BCM numbering)
| From | To |
|---|---|
| GPIO18 (physical pin 12) | 1 kOhm resistor -> NPN transistor BASE (2N2222 / BC547) |
| Transistor EMITTER | GND (physical pin 6) |
| Transistor COLLECTOR | Buzzer negative (-) |
| Buzzer positive (+) | 5 V (physical pin 2) |

The transistor protects the Pi's GPIO pin from the buzzer's current.
To use another pin: `python buzzer.py --pin 17 --test`.

## 3. Alert patterns
| Pattern | Sound | Used for |
|---|---|---|
| `warning` | short tick every second | yawning, high PERCLOS, face lost |
| `yawn` | double chirp | yawn detected |
| `drowsy` | fast beeping | eyes closed >= 0.8 s |
| `alarm` | almost continuous | eyes closed >= 1.5 s |

## 4. Files
| File | Purpose |
|---|---|
| `buzzer.py` | Buzzer driver (`BuzzerController`) + command-line control |
| `test_buzzer.py` | 11 automated tests + interactive hardware test |
| `BUZZER.md` | This document |

## 5. Install (Raspberry Pi OS 64-bit)
```bash
sudo apt update && sudo apt install -y python3-gpiozero python3-lgpio
# or in a virtualenv:  pip install gpiozero lgpio
```

## 6. Usage from the command line
```bash
python buzzer.py --test               # self-test + all patterns
python buzzer.py --on                 # ON until Ctrl+C
python buzzer.py --off                # OFF
python buzzer.py --pattern alarm --seconds 5
python buzzer.py --simulate --test    # no hardware (laptop)
```

## 7. Usage in your own code
```python
from buzzer import BuzzerController

with BuzzerController(pin=18) as bz:
    bz.on()            # sound
    bz.off()           # silent
    bz.beep(3)         # 3 blocking beeps
    bz.play("alarm")   # background pattern (does not block your camera loop)
    bz.stop()          # silence immediately
```

### Features
- `on()`, `off()`, `beep(n)`, `play(pattern)`, `stop()`, `self_test()`, `close()`
- Patterns run on their own thread, so video processing is never blocked
- `play()` with the same pattern does not restart it (no stutter when called every frame)
- Safety auto-silence after 30 s of continuous buzzing (`max_continuous_s`)
- Simulation mode when GPIO is missing - code runs on any computer
- Context manager always releases the pin on exit or crash

## 8. Testing
**Automated (no hardware, any computer):**
```bash
python test_buzzer.py        # or: pytest -q test_buzzer.py
```
Expected: `11/11 passed`.

**Hardware (on the Pi - you must hear it):**
```bash
python test_buzzer.py --hardware
```

| Test case | Expected result |
|---|---|
| Starts off | buzzer silent after creation |
| on / off | `is_on` True then False |
| beep(n) | ends silent |
| play + stop | pattern runs, stop silences < 1 s |
| Same pattern twice | thread not restarted |
| Switch pattern | new pattern replaces old |
| Unknown pattern | `ValueError` |
| Max continuous time | auto-silence after limit |
| Context manager | silent after `with` block |

**Troubleshooting**
| Problem | Fix |
|---|---|
| No sound | check transistor pins (flat side, E-B-C), buzzer polarity, pin number |
| Always on | transistor collector/emitter swapped |
| "simulation mode" message | gpiozero/lgpio not installed, or not running on a Pi |
| `Permission denied` on GPIO | add user to group: `sudo usermod -aG gpio $USER`, then re-login |

## 9. Push to Git
```bash
git add buzzer.py test_buzzer.py BUZZER.md
git commit -m "feat(sprint3): buzzer driver with alert patterns, tests and docs"
git push origin main
```

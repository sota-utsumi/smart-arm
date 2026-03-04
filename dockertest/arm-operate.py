import curses
from time import sleep
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

CH_NECK = 0
CH_ARM_BIG = 4
CH_ARM_SMALL = 8
#CH_GRIP = 12

LIMITS = {
    CH_NECK:      (30, 170),
    CH_ARM_BIG:   (40, 170),
    CH_ARM_SMALL: (20, 120),
#   CH_GRIP:      (0, 125),
}

TICK = 0.03      # 体感速度
STEP = 2         # 1 tickあたりの角度変化

def clamp(ch, a):
    mn, mx = LIMITS[ch]
    return max(mn, min(mx, int(a)))

def center(ch):
    mn, mx = LIMITS[ch]
    return (mn + mx) // 2

# 現在角
cur = {
    CH_NECK: center(CH_NECK),
    CH_ARM_BIG: center(CH_ARM_BIG),
    CH_ARM_SMALL: center(CH_ARM_SMALL),
   #CH_GRIP: center(CH_GRIP),
}

def apply_all():
    for ch in cur:
        kit.servo[ch].angle = cur[ch]

def apply(ch):
    kit.servo[ch].angle = cur[ch]

HELP = [
    "Controls (press & hold):",
    "  Neck (CH0):     A(left) / D(right)",
    "  Big Arm (CH4):  W(up)   / S(down)",
    "  Small (CH8):    I(up)   / K(down)",
    "  Grip (CH12):    O(open angle-) / P(close angle+)",
    "  H: home(center)   Q: quit",
]

def draw(stdscr):
    stdscr.clear()
    for i, line in enumerate(HELP):
        stdscr.addstr(i, 0, line)

    y = len(HELP) + 1
    stdscr.addstr(y+0, 0, f"STEP={STEP}  TICK={TICK}s")
    stdscr.addstr(y+2, 0, f"CH0  neck :  {cur[CH_NECK]:3d}   limits {LIMITS[CH_NECK]}")
    stdscr.addstr(y+3, 0, f"CH4  big  :  {cur[CH_ARM_BIG]:3d}   limits {LIMITS[CH_ARM_BIG]}")
    stdscr.addstr(y+4, 0, f"CH8  small:  {cur[CH_ARM_SMALL]:3d}   limits {LIMITS[CH_ARM_SMALL]}")
   #stdscr.addstr(y+5, 0, f"CH12 grip :  {cur[CH_GRIP]:3d}   limits {LIMITS[CH_GRIP]}")
    stdscr.addstr(y+7, 0, "Tip: if arm hits, STOP and reduce LIMITS min/max for that axis.")
    stdscr.refresh()

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)   # getch() をブロックしない
    stdscr.keypad(True)

    apply_all()
    draw(stdscr)

    while True:
        key = stdscr.getch()

        if key == -1:
            sleep(TICK)
            continue

        c = chr(key).lower() if 0 <= key < 256 else ""

        if c == "q":
            break

        moved = False

        # Neck
        if c == "a":
            cur[CH_NECK] = clamp(CH_NECK, cur[CH_NECK] - STEP); apply(CH_NECK); moved = True
        elif c == "d":
            cur[CH_NECK] = clamp(CH_NECK, cur[CH_NECK] + STEP); apply(CH_NECK); moved = True

        # Big arm
        elif c == "w":
            cur[CH_ARM_BIG] = clamp(CH_ARM_BIG, cur[CH_ARM_BIG] + STEP); apply(CH_ARM_BIG); moved = True
        elif c == "s":
            cur[CH_ARM_BIG] = clamp(CH_ARM_BIG, cur[CH_ARM_BIG] - STEP); apply(CH_ARM_BIG); moved = True

        # Small arm
        elif c == "i":
            cur[CH_ARM_SMALL] = clamp(CH_ARM_SMALL, cur[CH_ARM_SMALL] + STEP); apply(CH_ARM_SMALL); moved = True
        elif c == "k":
            cur[CH_ARM_SMALL] = clamp(CH_ARM_SMALL, cur[CH_ARM_SMALL] - STEP); apply(CH_ARM_SMALL); moved = True

        # Grip (O=open -> angle down, P=close -> angle up)
        # elif c == "o":
        #     cur[CH_GRIP] = clamp(CH_GRIP, cur[CH_GRIP] - STEP); apply(CH_GRIP); moved = True
        # elif c == "p":
        #     cur[CH_GRIP] = clamp(CH_GRIP, cur[CH_GRIP] + STEP); apply(CH_GRIP); moved = True

        # Home
        elif c == "h":
            for ch in cur:
                cur[ch] = center(ch)
            apply_all()
            moved = True

        if moved:
            draw(stdscr)

        sleep(TICK)

    # 必要なら信号停止（保持トルク消える）
    # for ch in [CH_NECK, CH_ARM_BIG, CH_ARM_SMALL, CH_GRIP]:
    #     kit.servo[ch].angle = None

if __name__ == "__main__":
    curses.wrapper(main)

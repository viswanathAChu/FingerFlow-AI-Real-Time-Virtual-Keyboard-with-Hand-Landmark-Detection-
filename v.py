import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pyttsx3

# ─────────────────────────────────────────────
#  SPEECH ENGINE  — background thread so camera
#  never freezes during speech
# ─────────────────────────────────────────────
_engine   = pyttsx3.init()
_engine.setProperty('rate', 160)
_engine.setProperty('volume', 1.0)
_lock     = threading.Lock()
_speaking = False

def speak(text):
    """Speak text in a daemon thread — non-blocking."""
    global _speaking
    def _run():
        global _speaking
        with _lock:
            _speaking = True
            _engine.say(text)
            _engine.runAndWait()
            _speaking = False
    if not _speaking:
        threading.Thread(target=_run, daemon=True).start()

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
WEBCAM_INDEX   = 0       # change to 1, 2 … if webcam not found
CLICK_DIST     = 35      # pinch threshold in pixels
COOLDOWN_SEC   = 0.35    # seconds between key presses
GAP            = 7       # gap between keys

# ─────────────────────────────────────────────
#  KEYBOARD LAYOUT
# ─────────────────────────────────────────────
ROWS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL") + ["⌫"],   # backspace at end of row 2
    list("ZXCVBNM"),
    ["SPACE"],
]

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ─────────────────────────────────────────────
#  BUILD KEYS  (called every frame — adapts to frame size)
# ─────────────────────────────────────────────
def build_keys(fw, fh):
    # keyboard occupies bottom 35% of frame
    kb_top    = int(fh * 0.65)
    kb_height = fh - kb_top - 12
    n_rows    = len(ROWS)
    key_h     = max((kb_height - GAP * (n_rows - 1)) // n_rows, 30)

    max_keys  = max(len(r) for r in ROWS)
    key_w     = max((fw - GAP * (max_keys + 1)) // max_keys, 40)

    keys = []
    for r, row in enumerate(ROWS):
        y1 = kb_top + r * (key_h + GAP)
        y2 = y1 + key_h

        if row == ["SPACE"]:
            space_w = key_w * 6 + GAP * 5
            x1 = (fw - space_w) // 2
            keys.append(dict(label="SPACE", x1=x1, y1=y1, x2=x1 + space_w, y2=y2))
            continue

        # compute total row width for centering
        total_w = 0
        for lbl in row:
            total_w += (int(key_w * 1.6) if lbl == "⌫" else key_w) + GAP
        total_w -= GAP
        x = (fw - total_w) // 2

        for lbl in row:
            w = int(key_w * 1.6) if lbl == "⌫" else key_w
            keys.append(dict(label=lbl, x1=x, y1=y1, x2=x + w, y2=y2))
            x += w + GAP

    return keys, key_w, key_h

# ─────────────────────────────────────────────
#  DRAW KEYBOARD
# ─────────────────────────────────────────────
def draw_keyboard(frame, keys, hovered_label=None):
    overlay = frame.copy()
    for k in keys:
        is_hover = (k["label"] == hovered_label)
        is_back  = (k["label"] == "⌫")

        if is_hover:
            bg = (255, 255, 255)
        elif is_back:
            bg = (10, 20, 50)
        else:
            bg = (18, 18, 18)
        cv2.rectangle(overlay, (k["x1"], k["y1"]), (k["x2"], k["y2"]), bg, -1)

        if is_hover:
            border = (0, 230, 160)
        elif is_back:
            border = (60, 160, 255)
        else:
            border = (65, 65, 65)
        cv2.rectangle(overlay, (k["x1"], k["y1"]), (k["x2"], k["y2"]), border, 2)

        font_color = (15, 15, 15) if is_hover else (225, 225, 225)
        label      = k["label"]
        kw         = k["x2"] - k["x1"]
        kh         = k["y2"] - k["y1"]

        if label == "SPACE":
            disp, fs, th = "SPACE", 0.65, 2
        elif label == "⌫":
            disp, fs, th = "< DEL", 0.52, 2
        else:
            disp = label
            fs   = min(0.7, kh / 75)
            th   = 2

        (tw, thr), _ = cv2.getTextSize(disp, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        tx = k["x1"] + (kw - tw) // 2
        ty = k["y1"] + (kh + thr) // 2
        cv2.putText(overlay, disp, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, font_color, th)

    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

# ─────────────────────────────────────────────
#  DRAW TEXT BOX  — large, centre-aligned text
# ─────────────────────────────────────────────
def draw_textbox(frame, text, fw, fh):
    margin  = int(fw * 0.07)
    box_x1  = margin
    box_x2  = fw - margin
    box_y1  = 18
    box_y2  = int(fh * 0.22)       # 22% of frame height = large box
    box_w   = box_x2 - box_x1
    box_h   = box_y2 - box_y1

    # background
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (12, 12, 12), -1)
    # glowing border
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 200, 140), 2)
    cv2.rectangle(frame, (box_x1 + 2, box_y1 + 2), (box_x2 - 2, box_y2 - 2),
                  (0, 100, 70), 1)

    # "Typed Text" label top-left
    cv2.putText(frame, "Typed Text", (box_x1 + 14, box_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 130, 100), 1)

    # ── wrap and centre text ──
    fs        = 1.5
    thickness = 2
    inner_w   = box_w - 36
    inner_h   = box_h - 42

    display   = text + "|"
    # estimate chars per line
    char_w    = int(fs * 17)
    cpl       = max(1, inner_w // char_w)

    lines = []
    tmp   = display
    while len(tmp) > cpl:
        lines.append(tmp[:cpl])
        tmp = tmp[cpl:]
    lines.append(tmp)
    lines = lines[-2:]            # show last 2 lines max

    line_h      = int(fs * 34)
    total_h     = len(lines) * line_h
    base_y      = box_y1 + 34 + (inner_h - total_h) // 2

    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, fs, thickness)
        tx = box_x1 + (box_w - tw) // 2     # centre aligned
        ty = base_y + i * line_h + th
        cv2.putText(frame, line, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 235, 175), thickness)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def get_tip(landmarks, idx, w, h):
    lm = landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)

def pinch_distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def key_at(x, y, keys):
    for k in keys:
        if k["x1"] < x < k["x2"] and k["y1"] < y < k["y2"]:
            return k["label"]
    return None

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Virtual Keyboard", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Virtual Keyboard", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    typed_text = ""
    last_click = 0.0
    click_anim = 0
    last_key   = ""   # tracks last typed key for flash display

    print("\n🖐  Virtual Keyboard ready — press  Q  to quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame          = cv2.flip(frame, 1)
        fh, fw         = frame.shape[:2]
        keys, kw, kh   = build_keys(fw, fh)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_model.process(rgb)

        hovered   = None

        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                index_tip = get_tip(hl, 8, fw, fh)
                thumb_tip = get_tip(hl, 4, fw, fh)

                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec((0, 170, 110), 2, 2),
                    mp_draw.DrawingSpec((0, 220, 160), 2, 2))

                hovered = key_at(*index_tip, keys)

                dist = pinch_distance(index_tip, thumb_tip)
                now  = time.time()
                if dist < CLICK_DIST and (now - last_click) > COOLDOWN_SEC:
                    if hovered:
                        if hovered == "SPACE":
                            typed_text += " "
                            speak("space")
                            last_key = "SPACE"
                        elif hovered == "⌫":
                            typed_text = typed_text[:-1]
                            speak("delete")
                            last_key = "DEL"
                        else:
                            typed_text += hovered
                            speak(hovered)
                            last_key = hovered
                        last_click = now
                        click_anim = 8

                cursor_col = (0, 50, 255) if dist < CLICK_DIST else (0, 220, 160)
                cv2.circle(frame, index_tip, 13, cursor_col, -1)
                cv2.circle(frame, thumb_tip,  9, (190, 190, 190), -1)
                cv2.line(frame, index_tip, thumb_tip, (90, 90, 90), 1)

        # ── draw UI ──────────────────────────
        draw_keyboard(frame, keys, hovered)
        draw_textbox(frame, typed_text, fw, fh)

        # hint — centred between textbox and keyboard
        hint     = "Pinch index + thumb to type    |    Q = quit"
        hint_fs  = 0.52
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, hint_fs, 1)
        hint_y   = int(fh * 0.30)
        cv2.putText(frame, hint, ((fw - hw) // 2, hint_y),
                    cv2.FONT_HERSHEY_SIMPLEX, hint_fs, (90, 90, 90), 1)

        # click flash — shows the key that was typed
        if click_anim > 0:
            msg = last_key if last_key else "CLICK!"
            fs  = 3.5 if len(msg) == 1 else 1.8
            (cw, ch), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, fs, 4)
            cv2.putText(frame, msg, ((fw - cw) // 2, int(fh * 0.48)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 160), 4)
            click_anim -= 1

        cv2.imshow("Virtual Keyboard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📝 Final typed text:\n{typed_text}\n")

if __name__ == "__main__":
    main()
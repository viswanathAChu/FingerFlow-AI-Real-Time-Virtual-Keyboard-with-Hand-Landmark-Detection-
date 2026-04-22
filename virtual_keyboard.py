import cv2
import mediapipe as mp
import numpy as np
import time

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
WEBCAM_INDEX   = 0          # change to 1, 2 … if your webcam isn't found
CLICK_DIST     = 35         # pixels — how close index tip must be to thumb tip to "click"
COOLDOWN_SEC   = 0.35       # seconds between repeated key presses
KEY_W, KEY_H   = 58, 52     # width / height of each key in pixels
START_X        = 20         # left margin of keyboard
START_Y        = 200        # top margin of keyboard (moved up for fullscreen visibility)
GAP            = 6          # gap between keys

# ─────────────────────────────────────────────
#  KEYBOARD LAYOUT
# ─────────────────────────────────────────────
ROWS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["SPACE", "⌫"],           # bottom row
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
#  HELPER – build key rectangles once
# ─────────────────────────────────────────────
def build_keys():
    """Return list of dicts: {label, x1, y1, x2, y2}"""
    keys = []
    for r, row in enumerate(ROWS):
        # centre-align each row
        total_w = len(row) * (KEY_W + GAP) - GAP
        row_x   = START_X + (640 - 2 * START_X - total_w) // 2
        y1 = START_Y + r * (KEY_H + GAP)
        y2 = y1 + KEY_H
        for c, label in enumerate(row):
            w = KEY_W * 3 + GAP * 2 if label == "SPACE" else KEY_W
            x1 = row_x + c * (KEY_W + GAP)
            x2 = x1 + w
            keys.append(dict(label=label, x1=x1, y1=y1, x2=x2, y2=y2))
            if label == "SPACE":
                # skip extra slots consumed by wide key
                row_x += (KEY_W + GAP) * 2
    return keys

ALL_KEYS = build_keys()

# ─────────────────────────────────────────────
#  DRAW KEYBOARD
# ─────────────────────────────────────────────
def draw_keyboard(frame, hovered_label=None):
    overlay = frame.copy()
    for k in ALL_KEYS:
        is_hover = (k["label"] == hovered_label)
        # background
        color = (255, 255, 255) if is_hover else (30, 30, 30)
        cv2.rectangle(overlay, (k["x1"], k["y1"]), (k["x2"], k["y2"]),
                      color, -1)
        # border
        border = (0, 220, 160) if is_hover else (80, 80, 80)
        cv2.rectangle(overlay, (k["x1"], k["y1"]), (k["x2"], k["y2"]),
                      border, 2)
        # label
        font_color = (20, 20, 20) if is_hover else (220, 220, 220)
        label = k["label"]
        font_scale = 0.55 if label not in ("SPACE", "⌫") else 0.45
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, 2)
        tx = k["x1"] + (k["x2"] - k["x1"] - tw) // 2
        ty = k["y1"] + (k["y2"] - k["y1"] + th) // 2
        cv2.putText(overlay, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 2)

    # blend for glass effect
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

# ─────────────────────────────────────────────
#  DRAW TYPED TEXT BOX
# ─────────────────────────────────────────────
def draw_textbox(frame, text):
    cv2.rectangle(frame, (30, 20), (610, 90), (20, 20, 20), -1)
    cv2.rectangle(frame, (30, 20), (610, 90), (0, 220, 160), 2)
    display = text[-36:] if len(text) > 36 else text   # show last 36 chars
    cv2.putText(frame, display + "|", (42, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 160), 2)

# ─────────────────────────────────────────────
#  LANDMARK HELPERS
# ─────────────────────────────────────────────
def get_tip(landmarks, idx, w, h):
    lm = landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)

def pinch_distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def key_at(x, y):
    for k in ALL_KEYS:
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
    cv2.setWindowProperty("Virtual Keyboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    typed_text  = ""
    last_click  = 0.0
    click_anim  = 0          # frames to show click flash

    print("\n🖐  Virtual Keyboard ready — press  Q  to quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        # Dynamically position keyboard based on actual frame height
        fh, fw = frame.shape[:2]
        global START_Y, ALL_KEYS
        START_Y = int(fh * 0.42)
        ALL_KEYS = build_keys()
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands_model.process(rgb)

        hovered = None
        index_tip = None

        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                h, w = frame.shape[:2]
                index_tip = get_tip(hl, 8,  w, h)   # index finger tip
                thumb_tip = get_tip(hl, 4,  w, h)   # thumb tip

                # draw skeleton
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec((0,180,120), 2, 2),
                    mp_draw.DrawingSpec((0,220,160), 2, 2))

                # hover key under index tip
                hovered = key_at(*index_tip)

                # pinch = click
                dist = pinch_distance(index_tip, thumb_tip)
                now  = time.time()
                if dist < CLICK_DIST and (now - last_click) > COOLDOWN_SEC:
                    if hovered:
                        if hovered == "SPACE":
                            typed_text += " "
                        elif hovered == "⌫":
                            typed_text = typed_text[:-1]
                        else:
                            typed_text += hovered
                        last_click = now
                        click_anim = 6   # flash for 6 frames

                # draw cursor dot
                color = (0, 80, 255) if dist < CLICK_DIST else (0, 220, 160)
                cv2.circle(frame, index_tip, 10, color, -1)
                cv2.circle(frame, thumb_tip,  8, (200, 200, 200), -1)

                # draw pinch distance arc
                cv2.line(frame, index_tip, thumb_tip, (120, 120, 120), 1)

        # ── draw UI ──────────────────────────────
        draw_keyboard(frame, hovered)
        draw_textbox(frame, typed_text)

        # click flash
        if click_anim > 0:
            cv2.putText(frame, "CLICK!", (260, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 160), 3)
            click_anim -= 1

        # instruction hint
        cv2.putText(frame, "Pinch index+thumb to type  |  Q to quit",
                    (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (140, 140, 140), 1)

        cv2.imshow("Virtual Keyboard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📝 Final typed text:\n{typed_text}\n")

if __name__ == "__main__":
    main()
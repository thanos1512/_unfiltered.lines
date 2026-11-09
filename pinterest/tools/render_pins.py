"""Render 1000x1500 (2:3) Pinterest pins in the Unfiltered Lines style.

Usage: python3 render_pins.py OUT_DIR [key ...]
"""
import math
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pins_data import PINS  # noqa: E402

S = 2                      # supersampling factor
W, H = 1000, 1500
SW, SH = W * S, H * S
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
SERIF = f"{FONT_DIR}/DMSerifDisplay-Regular.ttf"
BEBAS = f"{FONT_DIR}/BebasNeue-Regular.ttf"
INTER = f"{FONT_DIR}/Inter-Medium.otf"

THEMES = {
    "embers": dict(
        stops=[(0, "#0C0807"), (0.44, "#0C0807"), (0.56, "#110806"), (0.67, "#1C0907"),
               (0.78, "#300C07"), (0.86, "#4C1207"), (0.93, "#6A1B08"), (1.0, "#8A2A0B")],
        text="#FFF3E8", accent="#FFB547", header="#FFB547", handle="#FDE6DD", line="#FEB33F"),
    "cream": dict(
        stops=[(0, "#F8F1E6"), (0.19, "#F5EDE1"), (0.33, "#F2EADA"), (0.56, "#EFE3D0"),
               (0.78, "#EDDEC9"), (1.0, "#E9D8BF")],
        text="#221B15", accent="#B03F0A", header="#AF3E06", handle="#29221B", line="#B4400B"),
    "water": dict(
        stops=[(0, "#2F93AB"), (0.11, "#237D97"), (0.26, "#145E7A"), (0.44, "#0C425C"),
               (0.56, "#0B3450"), (0.67, "#092B46"), (0.78, "#07213A"), (0.93, "#051427"),
               (1.0, "#050E1D")],
        text="#F2F8FB", accent="#8AECF9", header="#8AEEFC", handle="#E1E9EE", line="#84ECF8"),
    "aurora": dict(
        stops=[(0, "#03101A"), (0.06, "#0A1628"), (0.2, "#082231"), (0.33, "#051E29"),
               (0.44, "#04212F"), (0.56, "#052530"), (0.67, "#062832"), (0.78, "#062C35"),
               (0.93, "#083038"), (1.0, "#073239")],
        text="#F1FAF7", accent="#7EF3C9", header="#F4FFFF", handle="#E1F2F0", line="#84F6CD"),
    "galaxy": dict(
        stops=[(0, "#100820"), (0.19, "#140B2A"), (0.33, "#180C32"), (0.44, "#1D0E3A"),
               (0.56, "#261047"), (0.67, "#311356"), (0.78, "#401964"), (0.86, "#4A1C6C"),
               (0.93, "#511E71"), (1.0, "#531E76")],
        text="#F7F0FF", accent="#FF90DB", header="#FF93DD", handle="#EDE2F8", line="#FF8ED9"),
    "sunset": dict(
        stops=[(0, "#1D0B3B"), (0.11, "#290D40"), (0.26, "#39104A"), (0.33, "#42124E"),
               (0.44, "#5D1A4F"), (0.56, "#7C2550"), (0.67, "#9D344D"), (0.78, "#BE4441"),
               (0.85, "#D45440"), (0.93, "#E47540"), (1.0, "#F09848")],
        text="#FFF6EF", accent="#FFD470", header="#FFD574", handle="#FFF2EA", line="#FDD36A"),
}


def rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def rgba(h, a):
    c = rgb(h).astype(int)
    return (int(c[0]), int(c[1]), int(c[2]), int(a))


# ---------------------------------------------------------------- backgrounds
_YY, _XX = np.mgrid[0:SH, 0:SW].astype(np.float32)


def vgrad(stops):
    ys = np.linspace(0, 1, SH, dtype=np.float32)
    pos = np.array([p for p, _ in stops], dtype=np.float32)
    cols = np.stack([rgb(c) for _, c in stops])
    col = np.stack([np.interp(ys, pos, cols[:, i]) for i in range(3)], axis=1)
    return np.broadcast_to(col[:, None, :], (SH, SW, 3)).astype(np.float32).copy()


def glow(img, cx, cy, rx, ry, color, strength, power=2.0):
    d = np.sqrt(((_XX - cx * S) / (rx * S)) ** 2 + ((_YY - cy * S) / (ry * S)) ** 2)
    a = (np.clip(1 - d, 0, 1) ** power) * strength
    img += (rgb(color) - img) * a[..., None]
    return img


def screen(img, mask, color, strength):
    """Screen-blend a colour into img through a 0..1 mask (mask at 1x, upscaled)."""
    if mask.shape != (SH, SW):
        m = Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8)).resize((SW, SH), Image.BICUBIC)
        mask = np.asarray(m).astype(np.float32) / 255.0
    c = rgb(color) / 255.0
    base = img / 255.0
    a = (mask * strength)[..., None]
    out = base * (1 - a) + (1 - (1 - base) * (1 - c)) * a
    return out * 255.0


def dots_layer(rng, n, yfun, rfun, colors, afun, glow_px=0, ring=False):
    layer = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range(n):
        x = rng.uniform(0, W)
        y = yfun()
        r = rfun()
        a = afun(y)
        col = rgba(colors[rng.integers(len(colors))], a)
        box = [(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S]
        if ring:
            d.ellipse(box, outline=col, width=max(2, int(1.2 * S)))
        else:
            d.ellipse(box, fill=col)
    if glow_px:
        g = layer.filter(ImageFilter.GaussianBlur(glow_px * S))
        layer = Image.alpha_composite(g, layer)
    return layer


def build_background(theme, seed):
    rng = np.random.default_rng(seed)
    t = THEMES[theme]
    img = vgrad(t["stops"])
    overlays = []

    if theme == "embers":
        img = glow(img, 500, 1610, 760, 600, "#D9500C", 0.80)
        img = glow(img, 500, 1560, 430, 330, "#FF7A1F", 0.45)
        overlays.append(dots_layer(
            rng, 170,
            yfun=lambda: H * (1 - (rng.random() ** 1.7) * 0.62),
            rfun=lambda: rng.choice([1.0, 1.3, 1.6, 2.0, 2.6, 3.2], p=[.25, .25, .2, .15, .1, .05]),
            colors=["#FFB547", "#FF8A2A", "#FFD98A", "#FF6A1A"],
            afun=lambda y: int(np.clip(70 + 185 * (y / H) ** 1.5 * rng.uniform(0.5, 1.0), 40, 255)),
            glow_px=3))
    elif theme == "cream":
        img = glow(img, 120, 40, 760, 640, "#FFFBF6", 0.65)
        img = glow(img, 900, 1500, 700, 500, "#E6D2B6", 0.25)
    elif theme == "water":
        img = glow(img, 520, -90, 640, 560, "#74C8DA", 0.55)
        rays = Image.new("L", (W, H), 0)
        rd = ImageDraw.Draw(rays)
        for _ in range(7):
            x0 = rng.uniform(120, 880)
            ang = math.radians(rng.uniform(-16, 16))
            w0 = rng.uniform(18, 60)
            length = rng.uniform(700, 1050)
            spread = rng.uniform(40, 120)
            x1 = x0 + math.tan(ang) * length
            rd.polygon([(x0 - w0, -10), (x0 + w0, -10), (x1 + w0 + spread, length), (x1 - w0 - spread, length)],
                       fill=int(rng.uniform(60, 120)))
        rays = rays.filter(ImageFilter.GaussianBlur(26))
        m = np.asarray(rays).astype(np.float32) / 255.0
        fade = np.clip(1 - np.linspace(0, 1, H) / 0.62, 0, 1)[:, None] ** 1.4
        img = screen(img, m * fade, "#CFF6FF", 0.55)
        overlays.append(dots_layer(
            rng, 16,
            yfun=lambda: rng.uniform(1080, 1470),
            rfun=lambda: rng.uniform(3.5, 9.0),
            colors=["#8AECF9", "#BFF4FB"],
            afun=lambda y: int(rng.uniform(55, 110)),
            ring=True))
        overlays.append(dots_layer(
            rng, 30,
            yfun=lambda: rng.uniform(300, 1480),
            rfun=lambda: rng.uniform(0.8, 1.6),
            colors=["#BFF4FB"],
            afun=lambda y: int(rng.uniform(40, 110))))
    elif theme == "aurora":
        mask = np.zeros((H, W), np.float32)
        purple = np.zeros((H, W), np.float32)
        ys = np.arange(H, dtype=np.float32)[:, None]
        xs = np.arange(W, dtype=np.float32)[None, :]

        def blur(a, r):
            im = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
            return np.asarray(im.filter(ImageFilter.GaussianBlur(r))).astype(np.float32) / 255

        for k in range(4):
            yc0 = rng.uniform(150, 300)
            a1, l1, p1 = rng.uniform(25, 70), rng.uniform(600, 1300), rng.uniform(0, 6.28)
            a2, l2, p2 = rng.uniform(8, 22), rng.uniform(200, 420), rng.uniform(0, 6.28)
            thick = rng.uniform(16, 34)
            reach = rng.uniform(70, 170)
            yc = yc0 + a1 * np.sin(2 * np.pi * xs / l1 + p1) + a2 * np.sin(2 * np.pi * xs / l2 + p2)
            env = (0.5 + 0.5 * np.sin(2 * np.pi * xs / rng.uniform(500, 1100) + rng.uniform(0, 6.28)))
            env = np.clip(0.15 + 0.85 * env * (0.65 + 0.35 * np.sin(2 * np.pi * xs / rng.uniform(180, 320) + rng.uniform(0, 6.28))), 0, 1)
            streak = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(xs / rng.uniform(5, 9) + rng.uniform(0, 6.28))) ** 3
            core = np.exp(-((ys - yc) / thick) ** 2)
            up = np.where(ys < yc, np.exp(-(yc - ys) / reach), 0) * streak
            mask += (core * 0.9 + up * 0.8) * env * rng.uniform(0.6, 1.0)
            purple += np.where(ys < yc - thick, np.exp(-(yc - thick - ys) / 120), 0) * env * 0.5
        mask = np.clip(mask, 0, 1.3) / 1.3
        sharp = blur(mask, 5)
        haze = blur(mask, 30)
        purple = blur(np.clip(purple, 0, 1), 36)
        img = screen(img, purple, "#6E5BD0", 0.50)
        img = screen(img, np.clip(haze * 0.9 + sharp * 0.5, 0, 1), "#26D99B", 0.78)
        img = screen(img, sharp ** 2, "#C6FFEA", 0.30)
        overlays.append(dots_layer(
            rng, 150,
            yfun=lambda: rng.uniform(0, H),
            rfun=lambda: rng.choice([0.7, 0.9, 1.2, 1.6], p=[.4, .3, .2, .1]),
            colors=["#FFFFFF", "#D9FFF2"],
            afun=lambda y: int(rng.uniform(50, 200))))
    elif theme == "galaxy":
        img = glow(img, 840, 1430, 430, 390, "#B8459E", 0.55)
        img = glow(img, 160, 1500, 520, 300, "#5B2397", 0.35)
        overlays.append(dots_layer(
            rng, 230,
            yfun=lambda: H * (rng.random() ** 1.25),
            rfun=lambda: rng.choice([0.7, 0.9, 1.2, 1.7], p=[.4, .3, .2, .1]),
            colors=["#FFFFFF", "#FFD6F3", "#E6DCFF"],
            afun=lambda y: int(np.clip(rng.uniform(60, 230) * (1.15 - y / H * 0.6), 30, 240))))
    elif theme == "sunset":
        img = glow(img, 500, 1600, 640, 560, "#FFD27A", 0.95)
        img = glow(img, 500, 1540, 980, 430, "#FF9A52", 0.45)
        streaks = Image.new("L", (W, H), 0)
        sd = ImageDraw.Draw(streaks)
        for _ in range(7):
            cy = rng.uniform(720, 1180)
            cx = rng.uniform(150, 850)
            sw = rng.uniform(260, 520)
            sh = rng.uniform(7, 16)
            sd.ellipse([cx - sw, cy - sh, cx + sw, cy + sh], fill=int(rng.uniform(80, 160)))
        streaks = streaks.filter(ImageFilter.GaussianBlur(14))
        img = screen(img, np.asarray(streaks).astype(np.float32) / 255, "#FFB3A0", 0.30)

    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    for ov in overlays:
        out = Image.alpha_composite(out, ov)
    return out.convert("RGB")


# ---------------------------------------------------------------- typography
def parse_quote(q):
    tokens, accent = [], False
    for w in q.split(" "):
        if w.startswith("["):
            accent, w = True, w[1:]
        end = w.endswith("]")
        if end:
            w = w[:-1]
        tokens.append((w, accent))
        if end:
            accent = False
    return tokens


def wrap(tokens, font, maxw):
    """Balanced line breaking (minimum raggedness) instead of greedy wrapping."""
    n = len(tokens)
    widths = [font.getlength(w) for w, _ in tokens]
    space = font.getlength(" ")
    INF = float("inf")
    best = [INF] * (n + 1)
    best[n] = 0.0
    nxt = [n] * (n + 1)
    for i in range(n - 1, -1, -1):
        lw = 0.0
        for j in range(i, n):
            lw = widths[j] if j == i else lw + space + widths[j]
            if lw > maxw and j > i:
                break
            slack = max(0.0, maxw - lw) / maxw
            words = j - i + 1
            if j == n - 1:
                cost = (max(0.0, slack - 0.45) ** 2) * 2
            else:
                cost = slack ** 2
            if words == 1 and n > 3:
                cost += 0.6          # avoid one-word lines
            total = cost + best[j + 1] + 0.05   # small per-line cost keeps line count low
            if total < best[i]:
                best[i] = total
                nxt[i] = j + 1
    lines, i = [], 0
    while i < n:
        j = nxt[i]
        lines.append(tokens[i:j])
        i = j
    return lines


def ink_bbox(font, text):
    """Real ink bbox of text drawn at origin with anchor 'ls' (left-baseline)."""
    pad = int(font.size * 2)
    im = Image.new("L", (pad * 3, pad * 3), 0)
    ImageDraw.Draw(im).text((pad, pad * 2), text, font=font, fill=255, anchor="ls")
    a = np.asarray(im)
    ys, xs = np.nonzero(a > 8)
    return xs.min() - pad, ys.min() - pad * 2, xs.max() - pad, ys.max() - pad * 2  # l, t, r, b


def render_pin(pin, seed, out_path):
    t = THEMES[pin["theme"]]
    img = build_background(pin["theme"], seed)
    d = ImageDraw.Draw(img)

    margin = 92
    maxw = (W - 2 * margin) * S
    tokens = parse_quote(pin["quote"])

    chosen = None
    for size in [100, 96, 92, 88, 84, 80, 76, 72, 68, 64, 60]:
        f = ImageFont.truetype(SERIF, size * S)
        lines = wrap(tokens, f, maxw)
        lh = 1.2 * size
        cap = 0.67 * size
        mark_h = 0.84 * size
        gap = 0.70 * size
        block = mark_h + gap + cap + (len(lines) - 1) * lh + 0.24 * size
        if len(lines) <= 6 and block <= 840:
            chosen = (size, f, lines, lh, cap, mark_h, gap, block)
            break
    if chosen is None:
        raise SystemExit(f"quote too long: {pin['quote']}")
    size, f, lines, lh, cap, mark_h, gap, block = chosen

    center_y = 752
    top = center_y - block / 2

    # opening quote mark, scaled so its ink height == mark_h
    probe = ImageFont.truetype(SERIF, 400)
    l, tt, r, b = ink_bbox(probe, "“")
    mfont = ImageFont.truetype(SERIF, int(400 * mark_h / (b - tt) * S))
    l, tt, r, b = ink_bbox(mfont, "“")
    d.text((margin * S - l, top * S - tt), "“", font=mfont, fill=t["accent"], anchor="ls")

    # quote lines
    space = f.getlength(" ")
    base = top + mark_h + gap + cap
    for i, line in enumerate(lines):
        x = margin * S
        y = (base + i * lh) * S
        for w, a in line:
            d.text((x, y), w, font=f, fill=t["accent"] if a else t["text"], anchor="ls")
            x += f.getlength(w) + space

    # header: UNFILTERED LINES (tracked caps)
    hsize = 38
    hf = ImageFont.truetype(BEBAS, hsize * S)
    word = "UNFILTERED LINES"
    track = 0.14 * hsize * S
    widths = [hf.getlength(c) for c in word]
    total = sum(widths) + track * (len(word) - 1)
    x = (SW - total) / 2
    for c, cw in zip(word, widths):
        d.text((x, 132 * S), c, font=hf, fill=t["header"], anchor="ls")
        x += cw + track

    # handle with two short lines
    isize = 30
    inf = ImageFont.truetype(INTER, isize * S)
    handle = "@_unfiltered.lines"
    hw = inf.getlength(handle)
    cy = 1382
    hx = (SW - hw) / 2
    d.text((hx, (cy + 0.36 * isize) * S), handle, font=inf, fill=t["handle"], anchor="ls")
    ll, lg, th = 80 * S, 22 * S, 2 * S
    d.rectangle([hx - lg - ll, cy * S - th / 2, hx - lg, cy * S + th / 2], fill=t["line"])
    d.rectangle([hx + hw + lg, cy * S - th / 2, hx + hw + lg + ll, cy * S + th / 2], fill=t["line"])

    final = img.resize((W, H), Image.LANCZOS)
    arr = np.asarray(final).astype(np.float32)
    arr += np.random.default_rng(seed + 7).normal(0, 1.3, arr.shape)
    final = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    final.save(out_path, "JPEG", quality=92, subsampling=0, optimize=True)
    return size, len(lines)


def slug(text):
    s = text.split("–")[0].lower()
    s = s.replace("’", "").replace("'", "")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60].strip("-")


def pin_filename(pin):
    key = pin["src"].split("/")[-1].replace("2026-11-08-", "p-")
    return f"{key}-{slug(pin['title'])}.jpg"


if __name__ == "__main__":
    out_dir = sys.argv[1]
    only = set(sys.argv[2:])
    os.makedirs(out_dir, exist_ok=True)
    for i, pin in enumerate(PINS):
        name = pin_filename(pin)
        if only and not any(name.startswith(k) for k in only):
            continue
        size, n = render_pin(pin, seed=1000 + i, out_path=os.path.join(out_dir, name))
        print(f"{name}  size={size} lines={n}")

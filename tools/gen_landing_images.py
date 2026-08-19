"""
Regenerates the landing page imagery (static/landing-hero.jpg and
static/landing-process.jpg).

    python3 tools/gen_landing_images.py

These are composed programmatically rather than being screenshots: the real
result-view screenshot showed a black video element (a stopped <video> with
no poster frame), which is the least persuasive possible image for a video
tool. The frames here are synthesised to read as plausible broadcast footage
stills -- built from layered gradients, radial light sources, silhouette
geometry with real depth cues, and a grain/vignette pass -- and then composed
into UI mockups of this app's actual result view and pipeline.

They are illustrative, not claims about specific real footage. If real
broadcast frames (or generated stills) become available, dropping them in as
the `frames` list and re-running is the intended upgrade path.
"""

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import math


def vertical_gradient(w, h, stops):
    """stops: list of (position 0..1, (r,g,b))."""
    arr = np.zeros((h, w, 3), dtype=np.float64)
    positions = [s[0] for s in stops]
    colors = [np.array(s[1], dtype=np.float64) for s in stops]
    for y in range(h):
        t = y / max(1, h - 1)
        # find surrounding stops
        for i in range(len(positions) - 1):
            if positions[i] <= t <= positions[i + 1]:
                span = positions[i + 1] - positions[i]
                local = 0 if span == 0 else (t - positions[i]) / span
                arr[y, :] = colors[i] * (1 - local) + colors[i + 1] * local
                break
        else:
            arr[y, :] = colors[-1]
    return arr


def add_radial_light(arr, cx, cy, radius, color, strength):
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - cx) / radius) ** 2 + ((yy - cy) / radius) ** 2)
    falloff = np.clip(1 - d, 0, 1) ** 2
    for c in range(3):
        arr[:, :, c] += falloff * color[c] * strength
    return arr


def add_grain_and_vignette(img, grain=6, vignette=0.32):
    arr = np.array(img).astype(np.float64)
    h, w = arr.shape[:2]
    noise = np.random.normal(0, grain, (h, w, 1))
    arr += noise
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    d = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    v = np.clip(1 - vignette * (d ** 2), 0, 1)
    arr *= v[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def frame_city_dusk(w, h):
    """Wide city skyline at dusk -- warm sky, cool silhouettes, window lights."""
    arr = vertical_gradient(w, h, [
        (0.0, (28, 34, 74)), (0.32, (86, 62, 108)),
        (0.55, (214, 118, 88)), (0.70, (247, 176, 96)), (1.0, (34, 30, 46)),
    ])
    arr = add_radial_light(arr, w * 0.62, h * 0.66, w * 0.45, (255, 190, 120), 0.55)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    horizon = int(h * 0.72)
    rng = np.random.default_rng(7)
    x = -20
    while x < w + 20:
        bw = int(rng.integers(w * 0.05, w * 0.13))
        bh = int(rng.integers(h * 0.10, h * 0.34))
        top = horizon - bh
        d.rectangle([x, top, x + bw, horizon], fill=(20, 22, 38))
        for _ in range(int(bw * bh / 900)):
            wx = int(rng.integers(x + 3, max(x + 4, x + bw - 3)))
            wy = int(rng.integers(top + 3, max(top + 4, horizon - 3)))
            if rng.random() < 0.55:
                d.rectangle([wx, wy, wx + 2, wy + 3], fill=(255, 214, 148))
        x += bw + int(rng.integers(2, 10))
    d.rectangle([0, horizon, w, h], fill=(16, 16, 28))
    # ground light spill
    arr2 = np.array(img).astype(np.float64)
    arr2 = add_radial_light(arr2, w * 0.62, horizon, w * 0.5, (255, 170, 100), 0.22)
    img = Image.fromarray(np.clip(arr2, 0, 255).astype(np.uint8))
    return add_grain_and_vignette(img)


def frame_studio_interview(w, h):
    """Close-ish studio shot -- teal/orange grade, shallow-depth feel."""
    arr = vertical_gradient(w, h, [
        (0.0, (18, 42, 54)), (0.5, (24, 62, 78)), (1.0, (12, 28, 38)),
    ])
    arr = add_radial_light(arr, w * 0.38, h * 0.42, w * 0.42, (120, 220, 230), 0.42)
    arr = add_radial_light(arr, w * 0.78, h * 0.30, w * 0.30, (255, 150, 80), 0.40)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # subject silhouette (head + shoulders), soft-edged
    sil = Image.new('L', (w, h), 0)
    ds = ImageDraw.Draw(sil)
    hx, hy, hr = w * 0.42, h * 0.44, h * 0.17
    ds.ellipse([hx - hr, hy - hr * 1.15, hx + hr, hy + hr * 1.15], fill=255)
    ds.ellipse([hx - hr * 2.3, hy + hr * 0.85, hx + hr * 2.3, hy + hr * 4.2], fill=255)
    sil = sil.filter(ImageFilter.GaussianBlur(2))
    dark = Image.new('RGB', (w, h), (8, 18, 26))
    img = Image.composite(dark, img, sil)
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    return add_grain_and_vignette(img)


def frame_field_action(w, h):
    """Outdoor daylight wide -- greens/sky, motion-blur streaks."""
    arr = vertical_gradient(w, h, [
        (0.0, (120, 180, 224)), (0.42, (168, 208, 236)),
        (0.52, (96, 138, 82)), (1.0, (38, 68, 44)),
    ])
    arr = add_radial_light(arr, w * 0.30, h * 0.18, w * 0.40, (255, 250, 214), 0.50)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(21)
    horizon = int(h * 0.52)
    # distant treeline
    for i in range(70):
        tx = int(rng.integers(-10, w + 10))
        th = int(rng.integers(h * 0.03, h * 0.09))
        d.ellipse([tx - 12, horizon - th, tx + 12, horizon + 4], fill=(46, 78, 52))
    # motion streaks suggesting action
    for i in range(9):
        sy = int(rng.integers(int(h * 0.56), int(h * 0.95)))
        sx = int(rng.integers(0, w))
        ln = int(rng.integers(w * 0.10, w * 0.30))
        d.line([sx, sy, sx + ln, sy + rng.integers(-3, 4)], fill=(150, 190, 140), width=2)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return add_grain_and_vignette(img)


def frame_night_drama(w, h):
    """Moody night exterior -- deep blues, single practical light, rain feel."""
    arr = vertical_gradient(w, h, [
        (0.0, (10, 14, 32)), (0.55, (18, 26, 56)), (1.0, (8, 10, 22)),
    ])
    arr = add_radial_light(arr, w * 0.72, h * 0.34, w * 0.26, (180, 200, 255), 0.55)
    arr = add_radial_light(arr, w * 0.20, h * 0.78, w * 0.30, (255, 120, 90), 0.28)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(33)
    for _ in range(220):
        rx = int(rng.integers(0, w))
        ry = int(rng.integers(0, h))
        ln = int(rng.integers(6, 16))
        d.line([rx, ry, rx - 2, ry + ln], fill=(150, 170, 210), width=1)
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    return add_grain_and_vignette(img, grain=8, vignette=0.42)

from PIL import Image, ImageDraw, ImageFilter
import numpy as np



def frame_studio_desk(w, h):
    """News/studio desk two-shot: lit backdrop, desk foreground, two figures
    at different depths -- the layering is what makes it read as a real set
    rather than a floating silhouette."""
    arr = vertical_gradient(w, h, [
        (0.0, (16, 46, 62)), (0.45, (26, 86, 104)), (0.72, (18, 54, 70)), (1.0, (10, 24, 34)),
    ])
    # backdrop wash + key light
    arr = add_radial_light(arr, w * 0.50, h * 0.30, w * 0.55, (90, 200, 220), 0.40)
    arr = add_radial_light(arr, w * 0.84, h * 0.22, w * 0.26, (255, 168, 96), 0.45)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)

    # background: vertical light panels (studio wall) -- receding, low contrast
    for i in range(7):
        px = int(w * (0.06 + i * 0.135))
        d.rectangle([px, int(h * 0.10), px + int(w * 0.045), int(h * 0.60)],
                    fill=(38, 108, 128))
    img = img.filter(ImageFilter.GaussianBlur(3.2))  # background is defocused
    d = ImageDraw.Draw(img)

    # midground: two seated figures, different sizes = different depths
    def figure(cx, cy, scale, tone):
        hr = h * 0.085 * scale
        d.ellipse([cx - hr, cy - hr * 1.2, cx + hr, cy + hr * 1.2], fill=tone)
        d.ellipse([cx - hr * 2.1, cy + hr * 0.9, cx + hr * 2.1, cy + hr * 4.6], fill=tone)

    figure(w * 0.34, h * 0.50, 1.0, (14, 40, 54))
    figure(w * 0.62, h * 0.53, 0.92, (12, 34, 48))

    # foreground: desk edge, darker + sharper than everything behind it
    d.rectangle([0, int(h * 0.78), w, h], fill=(10, 26, 36))
    d.rectangle([0, int(h * 0.78), w, int(h * 0.80)], fill=(52, 140, 158))
    # desk sheen
    arr2 = np.array(img).astype(np.float64)
    arr2 = add_radial_light(arr2, w * 0.45, h * 0.86, w * 0.40, (60, 150, 170), 0.16)
    img = Image.fromarray(np.clip(arr2, 0, 255).astype(np.uint8))
    return add_grain_and_vignette(img, grain=5, vignette=0.30)


def frame_field_match(w, h):
    """Daylight pitch wide: proper depth via a lit field plane, receding
    crowd band, and figures scaled by distance -- the earlier version was
    empty because it had a horizon and nothing else."""
    arr = vertical_gradient(w, h, [
        (0.0, (128, 186, 228)), (0.30, (176, 214, 240)),
        (0.40, (150, 176, 150)), (0.46, (104, 152, 88)), (1.0, (54, 104, 56)),
    ])
    arr = add_radial_light(arr, w * 0.26, h * 0.14, w * 0.42, (255, 252, 220), 0.48)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(11)

    horizon = int(h * 0.40)
    # crowd band (background, defocused later)
    d.rectangle([0, int(horizon - h * 0.10), w, horizon], fill=(58, 62, 88))
    for _ in range(900):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(int(horizon - h * 0.10), horizon))
        c = tuple(int(v) for v in rng.integers(70, 190, size=3))
        d.point((cx, cy), fill=c)

    # pitch markings -- perspective lines converging slightly
    d.line([int(w * 0.02), h, int(w * 0.30), horizon], fill=(190, 220, 190), width=2)
    d.line([int(w * 0.98), h, int(w * 0.70), horizon], fill=(190, 220, 190), width=2)
    d.arc([int(w * 0.30), int(h * 0.62), int(w * 0.70), int(h * 0.95)], 200, 340,
          fill=(190, 220, 190), width=2)

    img = img.filter(ImageFilter.GaussianBlur(1.4))
    d = ImageDraw.Draw(img)

    # players: scaled by depth, sharper the closer they are
    def player(cx, cy, scale, kit):
        hh = h * 0.05 * scale
        d.ellipse([cx - hh * 0.42, cy - hh, cx + hh * 0.42, cy - hh * 0.2], fill=(58, 44, 38))
        d.rectangle([cx - hh * 0.5, cy - hh * 0.25, cx + hh * 0.5, cy + hh * 0.9], fill=kit)
        d.line([cx - hh * 0.3, cy + hh * 0.9, cx - hh * 0.55, cy + hh * 1.9], fill=(30, 34, 40), width=max(1, int(hh * 0.16)))
        d.line([cx + hh * 0.3, cy + hh * 0.9, cx + hh * 0.6, cy + hh * 1.85], fill=(30, 34, 40), width=max(1, int(hh * 0.16)))

    player(w * 0.30, h * 0.56, 0.85, (222, 226, 232))
    player(w * 0.47, h * 0.62, 1.05, (196, 42, 52))
    player(w * 0.63, h * 0.58, 0.90, (222, 226, 232))
    player(w * 0.78, h * 0.70, 1.25, (196, 42, 52))

    return add_grain_and_vignette(img, grain=4, vignette=0.26)


def frame_night_street(w, h):
    """Night exterior with actual subject matter: wet street, streetlamp
    pools, a figure. The earlier version was just rain over a gradient --
    no ground plane, no anchor, so it read as static."""
    arr = vertical_gradient(w, h, [
        (0.0, (10, 16, 38)), (0.40, (20, 30, 64)), (0.58, (14, 20, 44)), (1.0, (6, 8, 20)),
    ])
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)

    horizon = int(h * 0.58)
    # buildings receding into haze
    rng = np.random.default_rng(5)
    x = -10
    while x < w + 10:
        bw = int(rng.integers(w * 0.07, w * 0.16))
        bh = int(rng.integers(h * 0.14, h * 0.34))
        d.rectangle([x, horizon - bh, x + bw, horizon], fill=(16, 22, 46))
        for _ in range(int(bw * bh / 1400)):
            wx = int(rng.integers(x + 2, max(x + 3, x + bw - 2)))
            wy = int(rng.integers(horizon - bh + 2, horizon - 2))
            d.rectangle([wx, wy, wx + 2, wy + 3], fill=(230, 190, 130))
        x += bw + 4

    # street plane
    d.rectangle([0, horizon, w, h], fill=(12, 14, 30))

    # lamp pools on wet ground -- the strongest single depth cue here
    arr2 = np.array(img).astype(np.float64)
    for lx, strength in [(w * 0.22, 0.55), (w * 0.58, 0.75), (w * 0.86, 0.45)]:
        arr2 = add_radial_light(arr2, lx, horizon + h * 0.02, w * 0.16, (255, 196, 120), strength)
        arr2 = add_radial_light(arr2, lx, horizon + h * 0.30, w * 0.10, (255, 176, 96), strength * 0.5)
    img = Image.fromarray(np.clip(arr2, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)

    # vertical reflection streaks on wet asphalt
    for lx in [w * 0.22, w * 0.58, w * 0.86]:
        for i in range(14):
            yy = horizon + int(i * h * 0.030)
            ww = int(w * 0.035 * (1 - i / 18))
            if ww < 1:
                continue
            d.line([lx - ww, yy, lx + ww, yy], fill=(120, 92, 58), width=1)

    # foreground figure, backlit
    fx, fy = w * 0.40, h * 0.66
    hr = h * 0.055
    d.ellipse([fx - hr, fy - hr * 1.25, fx + hr, fy + hr * 1.1], fill=(6, 8, 18))
    d.ellipse([fx - hr * 1.9, fy + hr * 0.85, fx + hr * 1.9, fy + hr * 4.6], fill=(6, 8, 18))

    img = img.filter(ImageFilter.GaussianBlur(0.7))
    d = ImageDraw.Draw(img)
    # rain, finer and less uniform than before
    rng2 = np.random.default_rng(99)
    for _ in range(140):
        rx = int(rng2.integers(0, w))
        ry = int(rng2.integers(0, h))
        ln = int(rng2.integers(5, 11))
        d.line([rx, ry, rx - 1, ry + ln], fill=(120, 140, 180), width=1)

    return add_grain_and_vignette(img, grain=6, vignette=0.40)

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np


def load_font(size, bold=False):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold
        else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rounded_thumb(src, w, h, radius=6):
    t = src.resize((w, h), Image.LANCZOS).convert('RGB')
    mask = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    out = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    out.paste(t, (0, 0), mask)
    return out


def build_process_strip(frames, out_path, scale=2):
    """frames: list of PIL Images used as the source footage."""
    W, H = 1000 * scale, 460 * scale
    BG = (14, 20, 33)
    INK = (231, 237, 246)
    DIM = (139, 152, 173)
    LINE = (38, 49, 73)
    ACCENT = (239, 68, 68)
    GREEN = (46, 204, 113)

    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    f_label = load_font(13 * scale, bold=True)
    f_small = load_font(11 * scale)
    f_score = load_font(11 * scale, bold=True)
    f_step = load_font(10 * scale, bold=True)

    pad = 28 * scale
    y = pad

    # ---------- Stage 1: source footage ----------
    d.text((pad, y), 'STEP 01', font=f_step, fill=ACCENT)
    d.text((pad + 62 * scale, y), 'Full episode in', font=f_label, fill=INK)
    y += 22 * scale

    strip_h = 62 * scale
    strip_w = W - pad * 2
    n_src = 10
    tw = strip_w // n_src
    for i in range(n_src):
        src = frames[i % len(frames)]
        # vary the crop so repeated source frames don't read as identical tiles
        cw = int(src.width * 0.72)
        off = int((src.width - cw) * (i / max(1, n_src - 1)))
        crop = src.crop((off, 0, off + cw, src.height))
        t = rounded_thumb(crop, tw - 3 * scale, strip_h, radius=4 * scale)
        img.paste(t, (pad + i * tw, y), t)
    d.text((pad, y + strip_h + 6 * scale), '48 min source \u00b7 scanned end to end',
           font=f_small, fill=DIM)

    y += strip_h + 30 * scale

    # ---------- Stage 2: scored scenes ----------
    d.text((pad, y), 'STEP 02', font=f_step, fill=ACCENT)
    d.text((pad + 62 * scale, y), 'Every scene detected and scored', font=f_label, fill=INK)
    y += 22 * scale

    n_sc = 6
    gap = 12 * scale
    cw = (strip_w - gap * (n_sc - 1)) // n_sc
    ch = 74 * scale
    scores = [9.2, 4.1, 8.7, 3.4, 9.6, 5.2]
    picked = [True, False, True, False, True, False]
    for i in range(n_sc):
        x = pad + i * (cw + gap)
        src = frames[i % len(frames)]
        t = rounded_thumb(src, cw, ch, radius=6 * scale)
        if not picked[i]:
            # unpicked scenes read as visually rejected, not just unlabelled
            t = t.convert('RGB')
            t = Image.blend(t, Image.new('RGB', t.size, BG), 0.55).convert('RGBA')
            mask = Image.new('L', (cw, ch), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, cw - 1, ch - 1], radius=6 * scale, fill=255)
            t.putalpha(mask)
        img.paste(t, (x, y), t)

        col = GREEN if picked[i] else (90, 100, 120)
        d.rounded_rectangle([x, y, x + cw - 1, y + ch - 1], radius=6 * scale,
                            outline=col, width=2 * scale)
        # score badge
        bw, bh = 34 * scale, 17 * scale
        d.rounded_rectangle([x + cw - bw - 5 * scale, y + 5 * scale,
                             x + cw - 5 * scale, y + 5 * scale + bh],
                            radius=4 * scale, fill=(10, 14, 24))
        d.text((x + cw - bw + 4 * scale, y + 8 * scale), f'{scores[i]:.1f}',
               font=f_score, fill=col)
        if picked[i]:
            d.text((x + 7 * scale, y + ch - 19 * scale), 'KEEP', font=f_score, fill=GREEN)

    d.text((pad, y + ch + 6 * scale),
           'Vision + dialogue scoring \u00b7 highest-scoring moments kept',
           font=f_small, fill=DIM)

    y += ch + 30 * scale

    # ---------- Stage 3: assembled promo ----------
    d.text((pad, y), 'STEP 03', font=f_step, fill=ACCENT)
    d.text((pad + 62 * scale, y), 'Cut, mixed and delivered', font=f_label, fill=INK)
    y += 22 * scale

    out_h = 76 * scale
    kept = [i for i, p in enumerate(picked) if p]
    seg_w = strip_w // len(kept)
    for j, i in enumerate(kept):
        src = frames[i % len(frames)]
        t = rounded_thumb(src, seg_w - 3 * scale, out_h, radius=6 * scale)
        img.paste(t, (pad + j * seg_w, y), t)
    d.rounded_rectangle([pad, y, pad + strip_w - 1, y + out_h - 1],
                        radius=6 * scale, outline=ACCENT, width=2 * scale)
    # timeline ticks under the assembled cut
    ty = y + out_h + 8 * scale
    d.line([pad, ty, pad + strip_w, ty], fill=LINE, width=2 * scale)
    for j in range(len(kept) + 1):
        tx = pad + j * seg_w
        tx = min(tx, pad + strip_w)
        d.line([tx, ty - 4 * scale, tx, ty + 4 * scale], fill=ACCENT, width=2 * scale)
    d.text((pad, ty + 8 * scale),
           '15s promo \u00b7 music + narration mixed \u00b7 MP4 / ProRes / AVC-Intra out',
           font=f_small, fill=DIM)

    img = img.resize((W // scale, H // scale), Image.LANCZOS)  # supersampled for crisp text
    img.save(out_path, optimize=True)
    return img

from PIL import Image, ImageDraw, ImageFont
import numpy as np



def build_hero(frames, out_path, scale=2):
    W, H = 760 * scale, 500 * scale
    PANEL = (18, 26, 43)
    CARD = (26, 34, 51)
    INK = (231, 237, 246)
    DIM = (139, 152, 173)
    LINE = (38, 49, 73)
    ACCENT = (239, 68, 68)
    GREEN = (46, 204, 113)

    img = Image.new('RGB', (W, H), PANEL)
    d = ImageDraw.Draw(img)
    f_title = load_font(13 * scale, bold=True)
    f_small = load_font(10 * scale)
    f_mono = load_font(10 * scale, bold=True)
    f_badge = load_font(9 * scale, bold=True)

    pad = 18 * scale

    # ---- window chrome ----
    d.rectangle([0, 0, W, 30 * scale], fill=(12, 18, 30))
    for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        cx = pad + i * 16 * scale
        d.ellipse([cx, 11 * scale, cx + 9 * scale, 20 * scale], fill=c)
    d.text((pad + 62 * scale, 11 * scale), 'Promo ready \u00b7 15s \u00b7 3 of 6 scenes used',
           font=f_small, fill=DIM)

    y = 30 * scale + pad

    # ---- the player, with REAL colourful footage in it ----
    meter_col = 42 * scale
    vid_w = int(W - pad * 2 - meter_col)
    vid_h = int(vid_w * 9 / 16 * 0.62)
    frame = frames[0]
    t = rounded_thumb(frame, vid_w, vid_h, radius=8 * scale)
    img.paste(t, (pad, y), t)

    # transport bar across the bottom of the video
    bar_y = y + vid_h - 22 * scale
    d.rectangle([pad, bar_y, pad + vid_w, y + vid_h], fill=(8, 12, 20))
    d.ellipse([pad + 10 * scale, bar_y + 6 * scale, pad + 20 * scale, bar_y + 16 * scale], fill=INK)
    track_x0 = pad + 30 * scale
    track_x1 = pad + vid_w - 54 * scale
    d.line([track_x0, bar_y + 11 * scale, track_x1, bar_y + 11 * scale], fill=(70, 82, 104), width=2 * scale)
    played_to = track_x0 + int((track_x1 - track_x0) * 0.42)
    d.line([track_x0, bar_y + 11 * scale, played_to, bar_y + 11 * scale], fill=ACCENT, width=2 * scale)
    d.ellipse([played_to - 4 * scale, bar_y + 7 * scale, played_to + 4 * scale, bar_y + 15 * scale], fill=ACCENT)
    d.text((pad + vid_w - 46 * scale, bar_y + 6 * scale), '0:06', font=f_small, fill=DIM)

    # ---- live peak meter beside the video (a real feature of this app) ----
    m_w = 32 * scale
    mx = pad + vid_w + 10 * scale
    my = y
    m_h = vid_h
    d.rounded_rectangle([mx, my, mx + m_w, my + m_h], radius=5 * scale, fill=(10, 16, 26))
    for ch, level in enumerate([0.72, 0.61]):
        bx = mx + 6 * scale + ch * 10 * scale
        bw = 6 * scale
        d.rectangle([bx, my + 5 * scale, bx + bw, my + m_h - 5 * scale], fill=(22, 30, 44))
        lit_h = int((m_h - 10 * scale) * level)
        top = my + m_h - 5 * scale - lit_h
        for yy in range(top, my + m_h - 5 * scale, 2 * scale):
            frac = 1 - (yy - (my + 5 * scale)) / max(1, (m_h - 10 * scale))
            col = GREEN if frac < 0.6 else ((241, 196, 15) if frac < 0.85 else ACCENT)
            d.rectangle([bx, yy, bx + bw, yy + 1 * scale], fill=col)
        d.rectangle([bx, top - 2 * scale, bx + bw, top - 1 * scale], fill=INK)

    y += vid_h + pad

    # ---- delivery package row ----
    d.rounded_rectangle([pad, y, W - pad, y + 74 * scale], radius=8 * scale, fill=CARD)
    d.text((pad + 12 * scale, y + 10 * scale), 'DELIVERY PACKAGE', font=f_badge, fill=ACCENT)
    labels = [('MP4 MASTER', True), ('PRORES 29.97', False), ('PRORES 23.976', False), ('AVC-INTRA', False)]
    bx = pad + 12 * scale
    by = y + 28 * scale
    for text, primary in labels:
        tw = int(d.textlength(text, font=f_mono)) + 20 * scale
        if primary:
            d.rounded_rectangle([bx, by, bx + tw, by + 24 * scale], radius=5 * scale, fill=ACCENT)
            d.text((bx + 10 * scale, by + 7 * scale), text, font=f_mono, fill=(255, 255, 255))
        else:
            d.rounded_rectangle([bx, by, bx + tw, by + 24 * scale], radius=5 * scale,
                                outline=LINE, width=1 * scale)
            d.text((bx + 10 * scale, by + 7 * scale), text, font=f_mono, fill=DIM)
        bx += tw + 8 * scale

    y += 74 * scale + 10 * scale

    # ---- scene table ----
    rows = [('1', '0s', '12s', '9.2'), ('2', '18s', '24s', '8.7'), ('3', '30s', '36s', '9.6')]
    d.text((pad + 2 * scale, y), '#     START     END     SCORE', font=f_badge, fill=DIM)
    y += 16 * scale
    for r in rows:
        d.line([pad, y - 3 * scale, W - pad, y - 3 * scale], fill=LINE, width=1)
        d.text((pad + 2 * scale, y), r[0], font=f_small, fill=INK)
        d.text((pad + 34 * scale, y), r[1], font=f_small, fill=INK)
        d.text((pad + 84 * scale, y), r[2], font=f_small, fill=INK)
        d.text((pad + 132 * scale, y), r[3], font=f_small, fill=GREEN)
        y += 18 * scale

    img = img.resize((W // scale, H // scale), Image.LANCZOS)
    img.save(out_path, optimize=True)
    return img

if __name__ == '__main__':
    import os, sys, tempfile
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    static = os.path.join(repo, 'static')
    tmp = tempfile.mkdtemp()

    # Base frames. frame_city_dusk comes from the first pass; the other three
    # are the reworked versions -- the originals were flat and structureless
    # (a silhouette on a gradient reads as an avatar, not a studio shot).
    specs = [
        ('frame-city', frame_city_dusk),
        ('frame-studio', frame_studio_desk),
        ('frame-field', frame_field_match),
        ('frame-night', frame_night_street),
    ]
    frames = []
    for name, fn in specs:
        img = fn(640, 360)
        img.save(os.path.join(tmp, name + '.png'))
        frames.append(img)
        print('frame:', name)

    build_process_strip(frames, os.path.join(tmp, 'process.png'))
    build_hero([frames[0], frames[2], frames[1], frames[3]], os.path.join(tmp, 'hero.png'))

    # JPEG, not PNG: these are photographic-style images with gradients and
    # grain, which PNG compresses badly (~340KB each vs ~60KB here).
    for src, dst, w in [('process.png', 'landing-process.jpg', 1100),
                        ('hero.png', 'landing-hero.jpg', 900)]:
        img = Image.open(os.path.join(tmp, src)).convert('RGB')
        if img.width != w:
            img = img.resize((w, round(img.height * w / img.width)), Image.LANCZOS)
        out = os.path.join(static, dst)
        img.save(out, quality=86, optimize=True, progressive=True)
        print('wrote', out, img.size, os.path.getsize(out), 'bytes')

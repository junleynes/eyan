"""Broadcast-style text graphics for promo plugs.

Renders lower-thirds/kickers/tune-in cards as real RGBA images (Pillow), which
ffmpeg then composites with `overlay`. Deliberately NOT ffmpeg's `drawtext`:
drawtext can only stamp flat glyphs with a hard shadow, and it looks it --
no letter-spacing control, no gradient scrim, no rule/accent bar, no
multi-weight type. Everything below is the stuff that actually makes a
graphic read as broadcast rather than as a burned-in caption.

Self-contained aside from Pillow and a font file -- no app imports -- so it
can be exercised (and its output eyeballed) on its own.
"""
import os

# Title-safe: broadcast convention is to keep text inside the middle 80% so
# nothing is lost to overscan on older sets or a station's own bug/DOG.
# Expressed as a fraction of frame width/height.
TITLE_SAFE = 0.10

# Font lookup is a list, not one hardcoded path, because the font a given
# server actually has installed varies. Poppins first (clean geometric sans,
# reads well small and at speed, which is what a 1-2s kicker needs), then
# progressively more universal fallbacks. A server with none of these gets a
# clear error rather than a silently ugly default bitmap font.
_FONT_CANDIDATES = {
    'bold': [
        '/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    ],
    'medium': [
        '/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    ],
}

def find_font(weight='bold'):
    for path in _FONT_CANDIDATES.get(weight, []):
        if os.path.exists(path):
            return path
    return None

def _hex_to_rgb(hex_color, default=(52, 230, 197)):
    h = (hex_color or '').strip().lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    if len(h) != 6:
        return default
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default

def _draw_tracked_text(draw, xy, text, font, fill, tracking=0):
    """Pillow has no letter-spacing, so draw glyph by glyph. Tracking is the
    single biggest difference between type that reads as 'designed' and type
    that reads as 'default' -- broadcast kickers are almost always set with
    generous positive tracking."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking
    return x

def _tracked_width(draw, text, font, tracking=0):
    if not text:
        return 0
    w = sum(draw.textlength(ch, font=font) for ch in text)
    return w + tracking * max(0, len(text) - 1)

def render_kicker(text, width, height, accent='#34e6c5', kind='kicker',
                  subtitle=None):
    """A lower-third kicker: accent bar + tracked uppercase headline over a
    gradient scrim that fades to nothing at the top.

    The scrim is the part people skip and shouldn't: white text straight over
    footage is illegible the moment a bright frame comes up, and a hard black
    box looks like a caption burn-in. A vertical gradient reads as designed
    and guarantees contrast regardless of what's underneath.

    Returns an RGBA Pillow Image sized to the full frame, so ffmpeg can
    overlay it at 0,0 with no positioning maths.
    """
    from PIL import Image, ImageDraw, ImageFont

    bold_path = find_font('bold')
    med_path = find_font('medium')
    if not bold_path:
        raise RuntimeError('No usable font found for text graphics '
                           '(looked for Poppins/DejaVu/Liberation/FreeSans).')

    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    accent_rgb = _hex_to_rgb(accent)

    safe_x = int(width * TITLE_SAFE)
    safe_bottom = int(height * TITLE_SAFE)

    # Type scale off frame height so a 720p and a 1080p render look identical
    # rather than the text being half the size on one of them.
    head_size = max(14, int(height * 0.058))
    sub_size = max(11, int(height * 0.030))
    head_font = ImageFont.truetype(bold_path, head_size)
    sub_font = ImageFont.truetype(med_path or bold_path, sub_size)
    tracking = max(1, int(head_size * 0.06))

    headline = (text or '').strip().upper()
    sub = (subtitle or '').strip()

    head_h = head_size * 1.15
    sub_h = (sub_size * 1.5) if sub else 0
    bar_h = int(head_h + sub_h)

    # --- gradient scrim ---
    # Drawn as its own layer then alpha-composited, so the gradient is a real
    # per-row alpha ramp rather than a stack of translucent rectangles.
    scrim_top = int(height - safe_bottom - bar_h - height * 0.06)
    scrim = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(scrim)
    scrim_depth = height - scrim_top
    for i in range(scrim_depth):
        y = scrim_top + i
        # Ease-in curve, not linear: a linear ramp still shows a visible
        # "edge" where it starts. Squared falloff makes the top of the scrim
        # genuinely invisible against the picture.
        t = i / max(1, scrim_depth - 1)
        alpha = int(200 * (t ** 1.6))
        sdraw.line([(0, y), (width, y)], fill=(6, 10, 18, alpha))
    img = Image.alpha_composite(img, scrim)
    draw = ImageDraw.Draw(img)

    text_y = height - safe_bottom - bar_h

    # --- accent bar ---
    # Thin vertical rule to the left of the type. Cheap, and it's the single
    # most recognisable "this is a broadcast graphic" cue there is.
    bar_w = max(3, int(width * 0.0035))
    draw.rectangle([safe_x, text_y, safe_x + bar_w, text_y + bar_h],
                   fill=accent_rgb + (255,))
    text_x = safe_x + bar_w + int(width * 0.014)

    # --- headline ---
    # Soft drop shadow first (offset copy at low alpha) so the type holds up
    # even where the scrim is thinnest.
    shadow_off = max(1, int(head_size * 0.05))
    shadow_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    sh_draw = ImageDraw.Draw(shadow_layer)
    _draw_tracked_text(sh_draw, (text_x + shadow_off, text_y + shadow_off),
                       headline, head_font, (0, 0, 0, 150), tracking)
    img = Image.alpha_composite(img, shadow_layer)
    draw = ImageDraw.Draw(img)
    _draw_tracked_text(draw, (text_x, text_y), headline, head_font,
                       (255, 255, 255, 255), tracking)

    # --- optional subtitle (tune-in line, air date, etc.) ---
    if sub:
        sub_y = text_y + head_h + int(sub_size * 0.25)
        _draw_tracked_text(draw, (text_x, sub_y), sub.upper(), sub_font,
                           accent_rgb + (255,), max(1, int(sub_size * 0.10)))

    return img

def render_endcard(title, subtitle, width, height, accent='#34e6c5',
                   logo_path=None):
    """Full-frame tune-in card: centred title + accent rule + subtitle over a
    near-opaque wash, optionally with the station/show logo above it. Used at
    the tail of a promo rather than over picture, so it can be much heavier
    than the kicker above."""
    from PIL import Image, ImageDraw, ImageFont

    bold_path = find_font('bold')
    med_path = find_font('medium')
    if not bold_path:
        raise RuntimeError('No usable font found for text graphics.')

    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    accent_rgb = _hex_to_rgb(accent)

    wash = Image.new('RGBA', (width, height), (6, 10, 18, 232))
    img = Image.alpha_composite(img, wash)
    draw = ImageDraw.Draw(img)

    title_size = max(18, int(height * 0.085))
    sub_size = max(12, int(height * 0.036))
    title_font = ImageFont.truetype(bold_path, title_size)
    sub_font = ImageFont.truetype(med_path or bold_path, sub_size)
    t_track = max(1, int(title_size * 0.07))
    s_track = max(1, int(sub_size * 0.16))

    title = (title or '').strip().upper()
    subtitle = (subtitle or '').strip().upper()

    t_w = _tracked_width(draw, title, title_font, t_track)
    s_w = _tracked_width(draw, subtitle, sub_font, s_track) if subtitle else 0

    block_h = title_size * 1.2 + (sub_size * 2.4 if subtitle else 0)
    logo_img = None
    logo_h = 0
    if logo_path and os.path.exists(logo_path):
        try:
            logo_img = Image.open(logo_path).convert('RGBA')
            logo_h = int(height * 0.16)
            ratio = logo_h / logo_img.height
            logo_img = logo_img.resize((max(1, int(logo_img.width * ratio)), logo_h),
                                       Image.LANCZOS)
            block_h += logo_h + height * 0.04
        except Exception:
            logo_img = None
            logo_h = 0

    y = int((height - block_h) / 2)

    if logo_img is not None:
        img.alpha_composite(logo_img, (int((width - logo_img.width) / 2), y))
        y += logo_h + int(height * 0.04)

    _draw_tracked_text(draw, (int((width - t_w) / 2), y), title, title_font,
                       (255, 255, 255, 255), t_track)
    y += int(title_size * 1.2)

    if subtitle:
        rule_w = int(width * 0.06)
        rule_y = y + int(sub_size * 0.5)
        draw.rectangle([int((width - rule_w) / 2), rule_y,
                        int((width + rule_w) / 2), rule_y + max(2, int(height * 0.004))],
                       fill=accent_rgb + (255,))
        y = rule_y + int(sub_size * 1.1)
        _draw_tracked_text(draw, (int((width - s_w) / 2), y), subtitle, sub_font,
                           accent_rgb + (255,), s_track)

    return img

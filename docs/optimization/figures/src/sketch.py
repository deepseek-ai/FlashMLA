"""Tiny hand-drawn-style SVG generator (no dependencies; Pillow is used for text metrics when available).

Every stroke is drawn twice as a slightly bowed, jittered cubic (the "rough" look); fills are either flat pastel
polygons with wobbly corners or hachure lines.  The random generator is seeded, so the output is reproducible.

    s = Sketch(1600, 1000, seed=7)
    s.box(40, 60, 300, 120, fill=FILL["blue"], title="qkv_a", lines=["skinny GEMM", "14.9 MB"])
    s.arrow([(340, 120), (420, 120)])
    s.save("fig.svg")
"""
import math
import random
from xml.sax.saxutils import escape

INK = "#1e1e1e"
STROKE = {"blue": "#1864ab", "green": "#2b8a3e", "orange": "#d9480f", "red": "#c92a2a", "purple": "#5f3dc4",
          "gray": "#495057", "teal": "#0b7285", "yellow": "#e67700", "pink": "#a61e4d", "ink": INK}
FILL = {"blue": "#d0ebff", "green": "#d3f9d8", "orange": "#ffe8cc", "red": "#ffe3e3", "purple": "#e5dbff",
        "gray": "#f1f3f5", "teal": "#c5f6fa", "yellow": "#fff3bf", "pink": "#ffdeeb", "white": "#ffffff",
        "paper": "#fffdf6"}
FONT = "'Comic Sans MS','Segoe Print','KaiTi','STKaiti','Kaiti SC','AR PL UKai CN',cursive"

try:  # text metrics (optional)
    from PIL import ImageFont
    _F = {}

    def _font(kind, size):
        key = (kind, size)
        if key not in _F:
            path = {"lat": "C:/Windows/Fonts/comic.ttf", "latb": "C:/Windows/Fonts/comicbd.ttf",
                    "cjk": "C:/Windows/Fonts/simkai.ttf"}[kind]
            _F[key] = ImageFont.truetype(path, size)
        return _F[key]
    _font("lat", 16), _font("cjk", 16)
    _HAVE_PIL = True
except Exception:  # pragma: no cover
    _HAVE_PIL = False


def text_width(t, size, bold=False):
    """Approximate rendered width of `t` in px."""
    w = 0.0
    for ch in t:
        o = ord(ch)
        if _HAVE_PIL:
            if 0x20 <= o < 0x7f or o in (0xb5, 0xd7, 0xb7, 0xb1):
                w += _font("latb" if bold else "lat", size).getlength(ch)
            else:
                w += max(_font("cjk", size).getlength(ch), size * (1.0 if o >= 0x2e80 else 0.55))
        else:
            w += size * (1.0 if o >= 0x2e80 else 0.56)
    return w * (1.03 if bold else 1.0)


class Sketch:
    def __init__(self, w, h, seed=1, bg=FILL["paper"], rough=1.0):
        self.w, self.h, self.rough = w, h, rough
        self.rnd = random.Random(seed)
        self.out = []
        self.warnings = []
        self.out.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{bg}"/>')

    # ------------------------------------------------------------------ low level
    def _j(self, r):
        return (self.rnd.random() * 2 - 1) * r

    def _seg(self, x1, y1, x2, y2, rough):
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 1e-6:
            return ""
        r = rough * self.rough * min(1.6, 0.5 + L / 260.0)
        nx, ny = -(y2 - y1) / L, (x2 - x1) / L
        bow = self._j(r * 1.6) + self._j(L * 0.004)
        sx, sy = x1 + self._j(r), y1 + self._j(r)
        ex, ey = x2 + self._j(r), y2 + self._j(r)
        t1, t2 = 0.30 + self._j(0.08), 0.70 + self._j(0.08)
        c1x, c1y = x1 + (x2 - x1) * t1 + nx * bow + self._j(r * 0.5), y1 + (y2 - y1) * t1 + ny * bow + self._j(r * 0.5)
        c2x, c2y = x1 + (x2 - x1) * t2 + nx * bow + self._j(r * 0.5), y1 + (y2 - y1) * t2 + ny * bow + self._j(r * 0.5)
        return f"M{sx:.1f},{sy:.1f} C{c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} {ex:.1f},{ey:.1f}"

    def _path(self, d, stroke, width, dash=None, opacity=1.0, fill="none"):
        if not d:
            return
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        if opacity < 1:
            extra += f' opacity="{opacity}"'
        self.out.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" '
                        f'stroke-linejoin="round"{extra}/>')

    def line(self, x1, y1, x2, y2, stroke=INK, width=1.8, rough=1.0, dash=None, passes=2, opacity=1.0):
        d = " ".join(self._seg(x1, y1, x2, y2, rough) for _ in range(passes))
        self._path(d, stroke, width, dash, opacity)

    def polyline(self, pts, stroke=INK, width=1.8, rough=1.0, dash=None, passes=2, closed=False, opacity=1.0):
        pts = list(pts)
        if closed:
            pts = pts + [pts[0]]
        d = []
        for _ in range(passes):
            for (a, b) in zip(pts[:-1], pts[1:]):
                d.append(self._seg(a[0], a[1], b[0], b[1], rough))
        self._path(" ".join(d), stroke, width, dash, opacity)

    def _smooth(self, pts, closed, jit):
        """Catmull-Rom spline through jittered points, as an SVG path."""
        p = [(x + self._j(jit), y + self._j(jit)) for x, y in pts]
        n = len(p)
        if closed:
            get = lambda i: p[i % n]
            segs = range(n)
        else:
            get = lambda i: p[min(max(i, 0), n - 1)]
            segs = range(n - 1)
        d = [f"M{get(0)[0]:.1f},{get(0)[1]:.1f}"]
        for i in segs:
            p0, p1, p2, p3 = get(i - 1), get(i), get(i + 1), get(i + 2)
            c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
            c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
            d.append(f"C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}")
        if closed:
            d.append("Z")
        return " ".join(d)

    def curve(self, pts, stroke=INK, width=1.8, rough=1.0, dash=None, passes=2, closed=False, fill="none", opacity=1.0):
        jit = 1.1 * rough * self.rough
        if fill != "none":
            self.out.append(f'<path d="{self._smooth(pts, closed, jit * 0.6)}" fill="{fill}" stroke="none"/>')
        for _ in range(passes):
            self._path(self._smooth(pts, closed, jit), stroke, width, dash, opacity)

    # ------------------------------------------------------------------ shapes
    def _round_pts(self, x, y, w, h, r):
        """Outline points of a rounded rectangle: dense near the corners (so the spline stays local), sparse on the edges."""
        r = min(r, w / 2, h / 2)
        k = r * 0.2929  # the 45-degree point of each corner

        def edge(L):
            if L - 2 * r < 44:
                return [r, L / 2, L - r] if L - 2 * r > 6 else [L / 2]
            ds = [r, r + 16]
            n = int((L - 2 * r - 32) // 150)
            ds += [r + 16 + (L - 2 * r - 32) * i / (n + 1) for i in range(1, n + 1)]
            return ds + [L - r - 16, L - r]
        pts = [(x + d, y) for d in edge(w)] + [(x + w - k, y + k)]
        pts += [(x + w, y + d) for d in edge(h)] + [(x + w - k, y + h - k)]
        pts += [(x + w - d, y + h) for d in edge(w)] + [(x + k, y + h - k)]
        pts += [(x, y + h - d) for d in edge(h)] + [(x + k, y + k)]
        return pts

    def hachure(self, x, y, w, h, color, gap=8, width=1.1, rough=0.8):
        d = []
        t = gap * 0.5
        while t < w + h:
            ax, ay = x + min(t, w), y + max(0.0, t - w)
            bx, by = x + max(0.0, t - h), y + min(t, h)
            d.append(self._seg(ax, ay, bx, by, rough))
            t += gap + self._j(gap * 0.12)
        self._path(" ".join(d), color, width, opacity=0.85)

    def rect(self, x, y, w, h, stroke=INK, fill=None, width=1.8, r=10, rough=1.0, dash=None, hatch=None, hatch_gap=8,
             passes=2):
        """Rounded sketchy rectangle.  fill = flat colour (under the outline); hatch = hachure colour."""
        if r > 0:
            pts = self._round_pts(x, y, w, h, r)
            if fill:
                self.out.append(f'<path d="{self._smooth(pts, True, 0.8)}" fill="{fill}" stroke="none"/>')
            if hatch:
                self.hachure(x + 3, y + 3, w - 6, h - 6, hatch, hatch_gap)
            for _ in range(passes):
                self._path(self._smooth(pts, True, 1.0 * rough * self.rough), stroke, width, dash)
        else:
            if fill:
                q = [(x + self._j(1), y + self._j(1)), (x + w + self._j(1), y + self._j(1)),
                     (x + w + self._j(1), y + h + self._j(1)), (x + self._j(1), y + h + self._j(1))]
                self.out.append('<path d="M' + " L".join(f"{a:.1f},{b:.1f}" for a, b in q) + f' Z" fill="{fill}" stroke="none"/>')
            if hatch:
                self.hachure(x + 2, y + 2, w - 4, h - 4, hatch, hatch_gap)
            self.polyline([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], stroke, width, rough, dash, passes, closed=True)

    def ellipse(self, cx, cy, rx, ry, stroke=INK, fill=None, width=1.8, rough=1.0, dash=None):
        n = 14
        pts = [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]
        if fill:
            self.out.append(f'<path d="{self._smooth(pts, True, 0.6)}" fill="{fill}" stroke="none"/>')
        for _ in range(2):
            self._path(self._smooth(pts, True, 1.0 * rough * self.rough), stroke, width, dash)

    def arrow(self, pts, stroke=INK, width=2.0, rough=1.0, dash=None, head=11, smooth=False, both=False, passes=2,
              opacity=1.0):
        """Arrow along a polyline (or a smooth curve through the points when smooth=True)."""
        pts = list(pts)
        if smooth and len(pts) > 2:
            for _ in range(passes):
                self._path(self._smooth(pts, False, 0.9 * rough * self.rough), stroke, width, dash, opacity)
        else:
            self.polyline(pts, stroke, width, rough, dash, passes, opacity=opacity)

        def tip(p_from, p_to):
            ang = math.atan2(p_to[1] - p_from[1], p_to[0] - p_from[0])
            for da in (0.42, -0.42):
                a = ang + math.pi + da + self._j(0.05)
                self.line(p_to[0], p_to[1], p_to[0] + head * math.cos(a), p_to[1] + head * math.sin(a), stroke, width,
                          rough * 0.5, passes=1, opacity=opacity)
        tip(pts[-2], pts[-1])
        if both:
            tip(pts[1], pts[0])

    # ------------------------------------------------------------------ text
    def text(self, x, y, t, size=15, color=INK, anchor="start", bold=False, italic=False, rot=0.0, maxw=None, family=None):
        """Single line; y is the baseline.  maxw: warn when the line is wider than this."""
        if maxw is not None:
            wpx = text_width(t, size, bold)
            if wpx > maxw + 1:
                self.warnings.append(f"overflow {wpx:.0f} > {maxw:.0f}: {t}")
        attr = f' font-weight="bold"' if bold else ""
        if italic:
            attr += ' font-style="italic"'
        if rot:
            attr += f' transform="rotate({rot:.2f} {x:.1f} {y:.1f})"'
        self.out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family or FONT}" font-size="{size}" fill="{color}" '
                        f'text-anchor="{anchor}" xml:space="preserve"{attr}>{escape(t)}</text>')

    def lines(self, x, y, ls, size=15, color=INK, anchor="start", lh=1.32, bold=False, maxw=None):
        """Multi-line text; returns the y of the baseline after the last line.  A line may be (text, colour[, bold])."""
        for l in ls:
            c, b = color, bold
            if isinstance(l, tuple):
                c = l[1] if len(l) > 1 and l[1] else color
                b = l[2] if len(l) > 2 else bold
                l = l[0]
            if l:
                self.text(x, y, l, size, c, anchor, b, maxw=maxw)
            y += size * lh
        return y

    # ------------------------------------------------------------------ composites
    def box(self, x, y, w, h, title=None, lines=None, fill=FILL["gray"], stroke=INK, tsize=17, size=14, align="center",
            tcolor=None, color=INK, r=10, dash=None, hatch=None, width=1.8, pad=10, lh=1.3, valign="top"):
        """Box with a bold title and body lines."""
        self.rect(x, y, w, h, stroke, fill, width, r, dash=dash, hatch=hatch)
        lines = lines or []
        total = (tsize * 1.25 if title else 0) + len(lines) * size * lh
        cy = y + pad + (tsize if title else size) if valign == "top" else y + (h - total) / 2 + (tsize if title else size)
        ax = x + w / 2 if align == "center" else x + pad
        anchor = "middle" if align == "center" else "start"
        if title:
            self.text(ax, cy, title, tsize, tcolor or stroke, anchor, bold=True, maxw=w - 2 * pad + 4)
            cy += tsize * 0.35 + size * lh * 0.95
        self.lines(ax, cy, lines, size, color, anchor, lh, maxw=w - 2 * pad + 4)

    def badge(self, cx, cy, t, fill=FILL["yellow"], stroke=INK, size=14, padx=9, h=None, bold=True, color=None):
        """Pill with a short label, centred on (cx, cy)."""
        w = text_width(t, size, bold) + 2 * padx
        h = h or size + 11
        self.rect(cx - w / 2, cy - h / 2, w, h, stroke, fill, 1.6, r=h / 2, rough=0.7)
        self.text(cx, cy + size * 0.36, t, size, color or stroke, "middle", bold=bold)
        return w

    def note(self, x, y, w, h, lines, fill="#fff9b1", stroke="#b08900", size=14, rot=-1.2, title=None, color=INK, lh=1.3):
        """Sticky note (slightly rotated)."""
        self.out.append(f'<g transform="rotate({rot:.2f} {x + w / 2:.1f} {y + h / 2:.1f})">')
        self.out.append(f'<path d="M{x + 3:.1f},{y + 4:.1f} h{w} v{h} h{-w} Z" fill="#000" opacity="0.10"/>')
        self.rect(x, y, w, h, stroke, fill, 1.4, r=0, rough=0.7)
        cy = y + 12 + size
        if title:
            self.text(x + 12, cy, title, size + 1, stroke, bold=True, maxw=w - 20)
            cy += (size + 1) * 1.35
        self.lines(x + 12, cy, lines, size, color, lh=lh, maxw=w - 20)
        self.out.append("</g>")

    def group(self, x, y, w, h, label=None, stroke=STROKE["gray"], fill=None, size=16, dash="9 7", label_fill=FILL["paper"]):
        """Dashed container with a label sitting on its top edge."""
        self.rect(x, y, w, h, stroke, fill, 1.6, r=16, dash=dash, rough=0.9, passes=1)
        if label:
            lw = text_width(label, size, True) + 18
            self.out.append(f'<rect x="{x + 22:.1f}" y="{y - size * 0.75:.1f}" width="{lw:.1f}" height="{size * 1.5:.1f}" fill="{label_fill}"/>')
            self.text(x + 31, y + size * 0.36, label, size, stroke, bold=True)

    def brace(self, x1, y1, x2, y2, depth=10, stroke=INK, width=1.6):
        """Curly-ish bracket from (x1,y1) to (x2,y2), bulging to the left of the direction of travel."""
        L = math.hypot(x2 - x1, y2 - y1)
        nx, ny = (y2 - y1) / L * depth, -(x2 - x1) / L * depth
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        pts = [(x1, y1), (x1 + nx * 0.8 + (mx - x1) * 0.08, y1 + ny * 0.8 + (my - y1) * 0.08),
               (mx - (mx - x1) * 0.12 + nx, my - (my - y1) * 0.12 + ny), (mx + nx * 2, my + ny * 2),
               (mx + (x2 - mx) * 0.12 + nx, my + (y2 - my) * 0.12 + ny),
               (x2 + nx * 0.8 - (x2 - mx) * 0.08, y2 + ny * 0.8 - (y2 - my) * 0.08), (x2, y2)]
        self.curve(pts, stroke, width, rough=0.5, passes=1)

    def raw(self, svg):
        self.out.append(svg)

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
                f'font-family="{FONT}">')
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(head + "\n" + "\n".join(self.out) + "\n</svg>\n")
        for wmsg in self.warnings:
            print("  [warn]", wmsg)
        return path

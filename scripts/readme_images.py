"""Regenerate the README images in docs/assets from the made-up demo export.

    uv run python scripts/readme_images.py

Needs Google Chrome or Chromium for the screenshots, and Pillow
(`uv run --with pillow ...`). The images show insights.html in light and dark,
cropped to whole cards and framed with rounded corners.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
WIDTH, SCALE, PAD, RADIUS = 900, 2, 32, 28
BORDER = {"light": (208, 215, 222, 255), "dark": (48, 54, 61, 255)}


def _chrome() -> str:
    for name in ("google-chrome-stable", "google-chrome", "chromium", "chromium-browser", "chrome"):
        if found := shutil.which(name):
            return found
    sys.exit("Chrome or Chromium is needed for the screenshots.")


def _screenshot(page: Path, out: Path, *, dark: bool) -> None:
    args = [
        _chrome(),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--force-device-scale-factor={SCALE}",
        f"--window-size={WIDTH},4800",
        f"--screenshot={out}",
    ]
    if dark:
        args += ["--force-dark-mode", "--blink-settings=preferredColorScheme=0"]
    subprocess.run([*args, page.as_uri()], check=True, capture_output=True, timeout=120)


def _cards(image: Image.Image) -> list[tuple[int, int]]:
    """Vertical extents of the page's blocks (title, tiles, each card...)."""
    px, (w, h) = image.load(), image.size
    background = px[5, 5]
    spans, start = [], None
    for y in range(h):
        blank = all(px[x, y] == background for x in range(0, w, 4))
        if not blank and start is None:
            start = y
        elif blank and start is not None:
            if y - start > 20:
                spans.append((start, y))
            start = None
    return spans


def _stack(image: Image.Image, spans: list[tuple[int, int]]) -> Image.Image:
    background = image.getpixel((5, 5))
    left, right = 16 * SCALE - PAD // 2, image.size[0] - 16 * SCALE + PAD // 2
    parts = [image.crop((left, a, right, b)) for a, b in spans]
    out = Image.new(
        "RGB",
        (right - left + PAD, sum(p.size[1] for p in parts) + PAD * (len(parts) + 1)),
        background,
    )
    y = PAD
    for part in parts:
        out.paste(part, (PAD // 2, y))
        y += part.size[1] + PAD
    return out


def _save(image: Image.Image, theme: str, path: Path) -> None:
    framed = image.convert("RGBA")
    w, h = framed.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), RADIUS, fill=255)
    framed.putalpha(mask)
    ImageDraw.Draw(framed).rounded_rectangle(
        (0, 0, w - 1, h - 1), RADIUS, outline=BORDER[theme], width=SCALE
    )
    framed.quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(
        path, optimize=True
    )


def main() -> None:
    """Build the demo archive, screenshot it, and write the images."""
    sys.path.insert(0, str(ROOT / "tests"))
    import demo

    with tempfile.TemporaryDirectory(prefix="convoviz-readme-") as tmp:
        work = Path(tmp)
        export = demo.build(work / "export.zip")
        archive = work / "ChatGPT Archive"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "convoviz",
                str(export),
                "-o",
                str(archive),
                "-y",
                "-q",
                "--timezone",
                "Europe/Paris",
            ],
            check=True,
        )
        ASSETS.mkdir(parents=True, exist_ok=True)
        for theme in ("light", "dark"):
            shot = work / f"{theme}.png"
            _screenshot(archive / "insights.html", shot, dark=theme == "dark")
            image = Image.open(shot).convert("RGB")
            title, _subtitle, tiles, calendar, week, _monthly, models, *_rest, words = _cards(image)
            _save(
                _stack(image, [(title[0] - 24, tiles[1]), week]),
                theme,
                ASSETS / f"insights-{theme}.png",
            )
            _save(_stack(image, [calendar, models]), theme, ASSETS / f"charts-{theme}.png")
            _save(_stack(image, [words]), theme, ASSETS / f"wordcloud-{theme}.png")
    print(f"Wrote images to {ASSETS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

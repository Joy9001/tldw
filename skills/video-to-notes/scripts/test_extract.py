"""Self-check for extract.py: `python test_extract.py`. Needs ffmpeg; no network, no whisper."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import extract  # noqa: E402

ROLLING_AUTO = """WEBVTT
Kind: captions
Language: en

00:00:00.100 --> 00:00:02.000 align:start position:0%

open<00:00:00.500><c> the</c><00:00:00.800><c> settings</c>

00:00:02.000 --> 00:00:02.010 align:start position:0%
open the settings


00:00:02.010 --> 00:00:04.000 align:start position:0%
open the settings
then<00:00:02.500><c> click</c><00:00:03.000><c> save</c>
"""
MANUAL = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nfirst&nbsp;\nline &amp; more\n\n01:02:03.500 --> 01:02:05.000\nlater\n"
SRT = "1\n00:00:01,000 --> 00:00:02,000\nhello\n\n2\n00:00:03,500 --> 00:00:04,000\nworld\n"


def test_parse_subs():
    assert extract.parse_subs(ROLLING_AUTO) == [(0.1, "open the settings"), (2.01, "then click save")]
    assert extract.parse_subs(MANUAL) == [(1.0, "first"), (1.0, "line & more"), (3723.5, "later")]
    assert extract.paragraphs(extract.parse_subs(SRT)) == "[00:01] hello world\n"


def test_pick_times():
    scores = [(t / 10, 0.0) for t in range(300)] + [(5.0, 0.9), (5.5, 0.8), (12.0, 0.05)]
    assert extract.pick_times(scores, 10) == [0.0, 5.0, 12.0]  # 5.5 too close to 5.0; zero scores ignored
    assert extract.pick_times(scores, 2) == [0.0, 5.0]


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        video = d / "demo.mp4"
        colors = [a for c in ("red", "blue", "green") for a in ("-f", "lavfi", "-i", f"color={c}:s=320x240:d=4")]
        subprocess.run(["ffmpeg", "-v", "error", *colors, "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0",
                        "-pix_fmt", "yuv420p", str(video)], check=True)
        (d / "demo.srt").write_text(SRT, encoding="utf-8")
        extract.main([str(video), "--out", str(d / "out")])
        meta = json.loads((d / "out" / "meta.json").read_text(encoding="utf-8"))
        assert meta["transcript"] == "captions (demo.srt)", meta["transcript"]
        assert [f["time"] for f in meta["frames"]] == ["00:00", "00:04", "00:08"], meta["frames"]
        assert all((d / "out" / f["file"]).exists() for f in meta["frames"])
        assert meta["sheets"] == ["sheets/sheet_01.jpg"]
        assert (d / "out" / "transcript.txt").read_text(encoding="utf-8") == "[00:01] hello world\n"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

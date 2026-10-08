"""Checks for extract.py.

  python test_extract.py            offline: synthetic videos, needs only ffmpeg
  uv run --no-project --python 3.12 --with faster-whisper python test_extract.py --online
                                    also downloads from real sites (slow; URLs can rot)
"""
import json
import os
import subprocess
import sys
import tempfile
import time
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
# BOM, a NOTE block, named cue ids, and a cue whose spoken text is just a number.
VTT_IDS = "﻿WEBVTT\n\nNOTE exported by a tool\n\nintro\n00:00:01.000 --> 00:00:02.000\nhello\n\nc2\n00:00:03.000 --> 00:00:04.000\n42\n"


def ffmpeg(*args, **kw):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True, **kw)


def make_video(path, size="320x240", colors=("red", "blue", "green"), seconds=4, audio=True, pipe=False):
    """Color blocks that change every `seconds`, optionally with a tone."""
    n = len(colors)
    cmd = [a for c in colors for a in ("-f", "lavfi", "-i", f"color={c}:s={size}:d={seconds}")]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=duration={n * seconds}"]
    cmd += ["-filter_complex", "".join(f"[{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]", "-map", "[v]"]
    if audio:
        cmd += ["-map", f"{n}:a"]
    cmd += ["-pix_fmt", "yuv420p"]
    if pipe:  # like a browser MediaRecorder file: no duration, no seek index
        with open(path, "wb") as f:
            ffmpeg(*cmd, "-f", "webm", "pipe:1", stdout=f)
    else:
        ffmpeg(*cmd, path)


def run(src, out=None):
    """extract.main on src -> (exit code, meta or None)."""
    try:
        extract.main([str(src)] + (["--out", str(out)] if out else []))
    except SystemExit as e:
        return e.code, None
    return 0, json.loads((Path(out) / "meta.json").read_text(encoding="utf-8")) if out else None


def times(meta):
    return [f["time"] for f in meta["frames"]]


def sheet_size(path):
    s = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                                  capture_output=True, check=True).stdout)["streams"][0]
    return s["width"], s["height"]


def test_parse_subs():
    assert extract.parse_subs(ROLLING_AUTO) == [(0.1, "open the settings"), (2.01, "then click save")]
    assert extract.parse_subs(MANUAL) == [(1.0, "first"), (1.0, "line & more"), (3723.5, "later")]
    assert extract.paragraphs(extract.parse_subs(SRT)) == "[00:01] hello world\n"
    assert extract.parse_subs(VTT_IDS) == [(1.0, "hello"), (3.0, "42")]


def test_pick_times():
    scores = [(t / 10, 0.0) for t in range(300)] + [(5.0, 0.9), (5.5, 0.8), (12.0, 0.05)]
    assert extract.pick_times(scores, 10) == [0.0, 5.0, 12.0]  # 5.5 too close to 5.0; zero scores ignored
    assert extract.pick_times(scores, 2) == [0.0, 5.0]


def test_containers():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "demo.srt").write_text(SRT, encoding="utf-8")
        for ext in ("mp4", "mkv", "webm", "mov"):
            make_video(d / f"demo.{ext}")
            code, meta = run(d / f"demo.{ext}", d / ext)
            assert code == 0, ext
            assert meta["transcript"] == "captions (demo.srt)", (ext, meta["transcript"])
            assert times(meta) == ["00:00", "00:04", "00:08"], (ext, times(meta))
            assert meta["duration"] == "00:12" and meta["sheets"] == ["sheets/sheet_01.jpg"], ext
            assert all((d / ext / f["file"]).exists() for f in meta["frames"]), ext
            assert (d / ext / "transcript.txt").read_text(encoding="utf-8") == "[00:01] hello world\n", ext


def test_portrait_sheets_stay_readable():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_video(d / "phone.mp4", size="180x320")
        (d / "phone.srt").write_text(SRT, encoding="utf-8")
        code, meta = run(d / "phone.mp4", d / "out")
        w, h = sheet_size(d / "out" / meta["sheets"][0])
        assert code == 0 and h > w and h <= 3 * 640 + 4 * 6, (w, h)  # tiles fit a 640px box


def test_no_audio_track():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_video(d / "silent.mp4", audio=False)
        code, meta = run(d / "silent.mp4", d / "out")
        assert code == 0, code  # must not ask for Whisper
        assert meta["transcript"] == "none (no audio track)", meta["transcript"]
        assert times(meta) == ["00:00", "00:04", "00:08"], times(meta)
        assert (d / "out" / "transcript.txt").read_text(encoding="utf-8").strip() == ""


def test_webm_without_duration():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_video(d / "browser.webm", pipe=True)
        (d / "browser.srt").write_text(SRT, encoding="utf-8")
        code, meta = run(d / "browser.webm", d / "out")
        assert code == 0, code
        assert meta["duration"] == "00:11", meta["duration"]  # last decoded frame, not container metadata
        assert times(meta) == ["00:00", "00:04", "00:08"], times(meta)


def test_audio_only():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ffmpeg("-f", "lavfi", "-i", "sine=duration=4", d / "tone.m4a")
        ffmpeg("-f", "lavfi", "-i", "sine=duration=4", "-f", "lavfi", "-i", "color=red:s=64x64:d=1",
               "-map", "0:a", "-map", "1:v", "-frames:v", "1", "-c:v", "mjpeg", "-disposition:v", "attached_pic",
               d / "cover.mp3")
        for name in ("tone.m4a", "cover.mp3"):  # cover art is not a video
            (d / name).with_suffix(".srt").write_text(SRT, encoding="utf-8")
            code, meta = run(d / name, d / name.replace(".", "_"))
            assert code == 0, name
            assert meta["frames"] == [] and meta["sheets"] == [], (name, meta["frames"])
            assert meta["transcript"].startswith("captions"), (name, meta["transcript"])


def test_unicode_and_spaces_in_path():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d) / "テスト フォルダ"
        d.mkdir()
        make_video(d / "デモ 録画 #1.mp4")
        (d / "デモ 録画 #1.srt").write_text(SRT, encoding="utf-8")
        cwd = os.getcwd()
        os.chdir(d)
        try:
            code, _ = run(d / "デモ 録画 #1.mp4")  # default output folder
        finally:
            os.chdir(cwd)
        meta = json.loads((d / "tldw" / "デモ-録画-1" / "meta.json").read_text(encoding="utf-8"))
        assert code == 0 and meta["title"] == "デモ 録画 #1" and len(meta["frames"]) == 3


def test_very_short_clip():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_video(d / "blip.mp4", colors=("red",), seconds=1)
        (d / "blip.srt").write_text(SRT, encoding="utf-8")
        code, meta = run(d / "blip.mp4", d / "out")
        assert code == 0 and times(meta) == ["00:00"] and meta["sheets"] == ["sheets/sheet_01.jpg"], meta


def test_bad_inputs():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "notes.txt").write_text("not a video", encoding="utf-8")
        assert run(d / "missing.mp4", d / "a")[0] == 2
        assert run(d / "notes.txt", d / "b")[0] == 2


# --- online: real sites. Each URL was checked when added; replace any that rot. -------------------------
PLAYLIST = "UU-KqnO3ez7vF-kyIQ_22rdA"  # a channel's uploads playlist
ONLINE = [
    ("youtube, human captions + chapters", "https://youtu.be/lz5OuKzvadQ",
     lambda m: m["transcript"] == "captions (en)" and len(m["chapters"]) == 8),
    ("youtube, auto captions only", "https://www.youtube.com/watch?v=0o7vULw57y8",
     lambda m: m["transcript"] == "captions (en-orig)"),
    ("youtube shorts, portrait", "https://www.youtube.com/shorts/fwBIZRq-vzY",
     lambda m: m["frames"] and m["transcript"].startswith("captions")),
    ("youtube, Japanese, 22 min", "https://www.youtube.com/watch?v=E7PZcIhKNgA",
     lambda m: m["transcript"] == "captions (ja-orig)" and len(m["chapters"]) == 15 and len(m["frames"]) >= 30),
    ("youtube, playlist params ignored", f"https://www.youtube.com/watch?v=lz5OuKzvadQ&list={PLAYLIST}&t=30s",
     lambda m: m["title"] == "Resolve Merge Conflict in Visual Studio Code"),
    ("loom, no captions -> whisper", "https://www.loom.com/share/43d05f362f734614a2e81b4694a3a523",
     lambda m: m["transcript"].startswith("whisper") and m["frames"]),
    ("direct .mp4 URL", "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
     lambda m: m["frames"]),
]
ONLINE_ERRORS = [
    ("playlist URL rejected", f"https://www.youtube.com/playlist?list={PLAYLIST}"),
    ("vimeo needs a login (yt-dlp, 2026)", "https://vimeo.com/76979871"),
    ("unavailable video", "https://www.youtube.com/watch?v=aaaaaaaaaaa"),
]


def online():
    failed = 0
    with tempfile.TemporaryDirectory() as d:
        for i, (name, url, check) in enumerate(ONLINE):
            start = time.time()
            code, meta = run(url, Path(d) / str(i))
            ok = code == 0 and bool(check(meta))
            failed += not ok
            summary = f"{time.time() - start:.0f}s, {meta['duration']}, {meta['transcript']}, {len(meta['frames'])} frames" if meta else ""
            print("ok " if ok else "FAIL", name, f"({summary})" if ok else f"(exit {code}, {summary})")
        for name, url in ONLINE_ERRORS:
            code, _ = run(url, Path(d) / "err")
            failed += code != 2
            print("ok " if code == 2 else "FAIL", name, f"(exit {code})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    if "--online" in sys.argv:
        online()

#!/usr/bin/env python3
"""Turn a video (URL or local file) into a transcript, candidate frames and contact sheets.

Usage: python extract.py <url-or-file> [--out DIR] [--model small] [--max-frames N]
Writes <out>/{meta.json, transcript.txt, frames/, sheets/} and prints a summary.
Exit codes: 0 ok, 2 bad input or missing tool, 3 no captions and faster-whisper not installed.
"""
import argparse
import glob
import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FONTS = ["C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
CUE = re.compile(r"^(?:(\d+):)?(\d+):(\d+)[.,](\d+)\s+-->")
SETTLE = 0.5  # grab frames slightly after a scene change so menus/animations finish


def die(code, msg):
    print(msg, file=sys.stderr)
    sys.exit(code)


def run(cmd):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True)


def ts(t):
    t = int(t)
    h, m, s = t // 3600, t // 60 % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_subs(text):
    """VTT/SRT -> [(seconds, line)]. Drops cue ids, tags and the repeated lines of rolling auto-captions."""
    out, last, t = [], None, None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = CUE.match(line)
        if m:
            h, mi, s, ms = m.groups()
            t = int(h or 0) * 3600 + int(mi) * 60 + int(s) + int(ms) / 10 ** len(ms)
            continue
        if i + 1 < len(lines) and CUE.match(lines[i + 1]):
            continue  # a VTT cue id or SRT index sits right above its timestamp
        line = html.unescape(re.sub(r"<[^>]+>", "", line)).replace("\xa0", " ").strip()
        if t is None or not line or line == last:
            continue
        out.append((t, line))
        last = line
    return out


def paragraphs(lines, span=20):
    paras = []
    for t, text in lines:
        if paras and t - paras[-1][0] < span:
            paras[-1][1].append(text)
        else:
            paras.append((t, [text]))
    return "\n".join(f"[{ts(t)}] {' '.join(words)}" for t, words in paras) + "\n"


def pick_subs(info):
    """Human captions in the video's language, then original-language auto captions, then anything."""
    lang = (info.get("language") or "").split("-")[0]
    subs = [k for k in info.get("subtitles") or {} if k != "live_chat"]
    auto = list(info.get("automatic_captions") or {})
    cands = ([(k, False) for k in subs if k.split("-")[0] == lang]
             + [(k, True) for k in auto if k.endswith("-orig")]
             + [(k, True) for k in auto if k.split("-")[0] == lang]
             + [(k, False) for k in subs])
    return cands[0] if cands else (None, False)


def download(url, info, out):
    key, auto = pick_subs(info)
    info_file = out / "info.json"  # reuse the -J metadata instead of extracting the page a second time
    info_file.write_text(json.dumps(info), encoding="utf-8")
    cmd = ["yt-dlp", "-q", "--no-warnings", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
           "-o", out / "video.%(ext)s", "--load-info-json", info_file]
    if key:
        cmd += ["--write-auto-subs" if auto else "--write-subs", "--sub-langs", key,
                "--sub-format", "vtt/best", "--convert-subs", "vtt"]
    try:
        run(cmd)
    finally:
        info_file.unlink(missing_ok=True)
    video = next(p for p in out.glob("video.*") if p.suffix not in (".vtt", ".part", ".ytdl", ".json"))
    sub = out / f"video.{key}.vtt"
    meta = {"title": info.get("title"), "source": url, "uploader": info.get("uploader"),
            "description": (info.get("description") or "")[:2000],
            "chapters": [{"start": c["start_time"], "title": c["title"]} for c in info.get("chapters") or []]}
    return video, meta, (sub.read_text(encoding="utf-8"), f"captions ({key})") if sub.exists() else None


def sidecar(video):
    """A .vtt/.srt next to a local file (Zoom, Loom, OBS exports) counts as captions."""
    stem = glob.escape(str(video.with_suffix("")))
    for p in sorted(glob.glob(stem + "*.vtt") + glob.glob(stem + "*.srt")):
        return Path(p).read_text(encoding="utf-8", errors="replace"), f"captions ({Path(p).name})"
    return None


def load_whisper():
    """faster-whisper's classes, or exit 3 with the command that installs it."""
    # NVIDIA's pip wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12) keep their DLLs where Windows won't look.
    spec = importlib.util.find_spec("nvidia")
    dirs = [d for root in (spec.submodule_search_locations if spec else []) for d in glob.glob(os.path.join(root, "*", "bin"))]
    os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError:
        args = " ".join(f'"{a}"' for a in sys.argv)
        gpu = ' --with nvidia-cublas-cu12 --with "nvidia-cudnn-cu12==9.*"' if shutil.which("nvidia-smi") else ""
        die(3, "No captions found and faster-whisper is not installed. Re-run with:\n"
               f"  uv run --no-project --python 3.12 --with faster-whisper{gpu} python {args}")
    return WhisperModel, BatchedInferencePipeline


def transcribe(video, model, whisper):
    import numpy as np  # installed alongside faster-whisper
    WhisperModel, BatchedInferencePipeline = whisper
    # Decode with ffmpeg ourselves: faster-whisper's PyAV path breaks on newer PyAV releases.
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, np.float32)
    for device in ("auto", "cpu"):
        try:
            pipeline = BatchedInferencePipeline(WhisperModel(model, device=device, compute_type="int8"))
            segs, info = pipeline.transcribe(audio, vad_filter=True, batch_size=8)  # ~1.8x faster than sequential
            return [(s.start, s.text.strip()) for s in segs], f"whisper {model} ({info.language})"
        except RuntimeError as e:  # e.g. CUDA picked but cuBLAS missing
            if device == "cpu":
                raise
            print(f"GPU transcription failed ({e}); using CPU", file=sys.stderr)


def scene_scores(video):
    out = run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-an",
               "-vf", "scale=640:-2,select='gte(scene,0)',metadata=print:file=-", "-f", "null", "-"]).stdout
    return [(float(t), float(s)) for t, s in re.findall(r"pts_time:([\d.]+)\s+lavfi\.scene_score=([\d.]+)", out)]


def pick_times(scores, k, gap=3.0):
    """Top-k scene changes at least `gap` seconds apart, plus the opening frame.
    Ranking beats a fixed threshold: screen recordings score ~0.05 where slide decks score ~0.5."""
    chosen = [0.0]
    for t, s in sorted(scores, key=lambda x: -x[1]):
        if len(chosen) >= k or s < 0.01:
            break
        if all(abs(t - c) >= gap for c in chosen):
            chosen.append(t)
    return sorted(chosen)


def grab(video, times, duration, out):
    frames, thumbs, sheets = out / "frames", out / "thumbs", out / "sheets"
    for d in (frames, thumbs, sheets):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    font = next((f"fontfile='{f.replace(':', chr(92) + ':')}':" for f in FONTS if Path(f).exists()), "")

    def one(i, t):
        at = max(0.0, min(t + SETTLE, duration - 0.1))
        name = f"{i:03d}_{ts(at).replace(':', '-')}.jpg"
        label = f"#{i} " + ts(at).replace(":", "\\:")
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", video,
             "-frames:v", "1", "-q:v", "2", frames / name,
             "-vf", f"scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2,drawtext={font}text='{label}':x=10:y=10:fontsize=30:fontcolor=white"
                    ":box=1:boxcolor=black@0.7:boxborderw=6",
             "-frames:v", "1", thumbs / f"{i:03d}.jpg"])
        return {"n": i, "time": ts(at), "seconds": round(at, 2), "file": f"frames/{name}"}

    with ThreadPoolExecutor(4) as pool:  # ponytail: 4 seek-and-decode jobs measured 1.5x; 8 gained nothing
        result = list(pool.map(one, range(1, len(times) + 1), times))
    run(["ffmpeg", "-v", "error", "-y", "-i", thumbs / "%03d.jpg", "-vf", "tile=3x3:padding=6:color=white",
         sheets / "sheet_%02d.jpg"])
    shutil.rmtree(thumbs)
    return result, sorted(f"sheets/{p.name}" for p in sheets.glob("*.jpg"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="video URL (anything yt-dlp supports) or local file")
    ap.add_argument("--out", help="output folder (default: ./tldw/<id>)")
    ap.add_argument("--model", default="small", help="faster-whisper model when there are no captions")
    ap.add_argument("--max-frames", type=int, help="candidate frames (default: ~1 per 12s, 6..45)")
    a = ap.parse_args(argv)

    is_url = re.match(r"https?://", a.source) is not None
    for tool in ("ffmpeg", "ffprobe") + (("yt-dlp",) if is_url else ()):
        if not shutil.which(tool):
            die(2, f"{tool} not found on PATH. Install it (yt-dlp: `pip install yt-dlp`, ffmpeg: https://ffmpeg.org/download.html).")
    try:
        if is_url:
            info = json.loads(run(["yt-dlp", "-J", "--no-playlist", "--flat-playlist", a.source]).stdout)
            if info.get("_type") == "playlist":
                die(2, "That URL is a playlist or channel; pass a single video URL.")
            if info.get("is_live"):
                die(2, "Live streams aren't supported; try again once the stream has ended.")
            out = Path(a.out or Path("tldw") / info["id"])
            out.mkdir(parents=True, exist_ok=True)
            video, meta, subs = download(a.source, info, out)
        else:
            video = Path(a.source).resolve()
            if not video.is_file():
                die(2, f"Not a URL or an existing file: {a.source}")
            out = Path(a.out or Path("tldw") / re.sub(r"[^\w.-]+", "-", video.stem))
            out.mkdir(parents=True, exist_ok=True)
            meta, subs = {"title": video.stem, "source": str(video), "chapters": []}, sidecar(video)

        probe = json.loads(run(["ffprobe", "-v", "error", "-print_format", "json",
                                "-show_format", "-show_streams", "-show_chapters", video]).stdout)
        streams = probe.get("streams", [])
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        has_video = any(s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
                        for s in streams)  # an mp3's cover art is a "video" stream too
        meta["chapters"] = meta["chapters"] or [{"start": float(c["start_time"]), "title": c.get("tags", {}).get("title", "")}
                                                for c in probe.get("chapters", [])]

        lines, source = (parse_subs(subs[0]), subs[1]) if subs else ([], None)
        if not lines and not has_audio:
            source = "none (no audio track)"
        whisper = load_whisper() if not lines and has_audio else None  # exits 3 before any slow work
        with ThreadPoolExecutor(1) as pool:  # Whisper runs on the GPU while the scene pass uses the CPU
            job = pool.submit(transcribe, video, a.model, whisper) if whisper else None
            scores = scene_scores(video) if has_video else []
            if job:
                lines, source = job.result()
        (out / "transcript.txt").write_text(paragraphs(lines), encoding="utf-8")

        # Browser-recorded webm often carries no duration; the last decoded frame stands in.
        duration = float(probe["format"].get("duration") or 0) or (scores[-1][0] if scores else 0.0)
        k = a.max_frames or max(6, min(45, round(duration / 12)))  # ponytail: naive density heuristic
        frames, sheets = grab(video, pick_times(scores, k), duration, out) if has_video else ([], [])
    except subprocess.CalledProcessError as e:
        die(2, f"{Path(str(e.cmd[0])).name} failed:\n{(e.stderr or '')[-2000:]}")

    meta.update(video=str(video.resolve()), duration=ts(duration), transcript=source, frames=frames, sheets=sheets)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out: {out.resolve()}\ntitle: {meta['title']}\nduration: {ts(duration)} | chapters: {len(meta['chapters'])}"
          f" | transcript: {source} | frames: {len(frames)} | sheets: {len(sheets)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

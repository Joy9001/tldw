---
name: tldw
description: Turn a video into a polished write-up with screenshots (blog post, study notes, how-to, or PR description) from a YouTube or any yt-dlp-supported URL, or a local recording (screen recording, Loom/Zoom/OBS export). Use when the user gives a video link or file and asks for notes, an article, a summary, docs, a tutorial, or a PR description, optionally published to Notion, Google Docs, or another connected tool.
---

# tldw: too long; didn't watch

## 1. Extract

```
python "<skill dir>/scripts/extract.py" "<url-or-file>" [--out DIR]
```

`<skill dir>` is this skill's base directory. Output goes to `./tldw/<id>/` and the script prints its `out:` path.

- **Exit 0**: continue.
- **Exit 3**: no captions and faster-whisper isn't installed. Tell the user it's a one-time setup (about 500 MB for the model, plus about 1 GB of CUDA libraries when an NVIDIA GPU is detected), then run the `uv run ...` command the script printed. Without `uv`: `pip install faster-whisper` into Python 3.10–3.12, plus `nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"` for GPU.
- **Exit 2**: show the error (missing tool, bad input, download failure, playlist or live-stream URL) and stop.

Before transcribing a video over 30 minutes with no captions, warn that CPU transcription takes roughly half the video's length, and offer `--model base` for speed.

A local file with a matching `.vtt`/`.srt` next to it uses those captions instead of Whisper.

Two cases change what you can write. `transcript: "none (no audio track)"` is a silent recording: work from the frames alone and tell the user there was no narration. Empty `frames` means an audio-only file: write without screenshots.

## 2. Understand

Read `meta.json` (title, description, chapters, `video` path, `frames` list) and `transcript.txt` (paragraphs prefixed with `[mm:ss]`). Chapters are the default outline when present.

## 3. Choose the format

The user's explicit request wins. Otherwise infer and state the choice in one line:

- Someone operating software step by step (screen recording, demo, tutorial) → **How-to**. If it presents a change or fix to a codebase → **PR description**.
- Lecture, talk, course, explainer → **Study notes**.
- Anything else → **Blog post**.

## 4. Pick screenshots

View `sheets/sheet_*.jpg`. Each tile is labeled `#n mm:ss` and matches `frames/NNN_mm-ss.jpg` (also listed in `meta.json`). Pick frames that show what the text describes: the UI after a step, the slide being discussed, the error message. Skip talking heads, transitions, near-duplicates and blank frames. Aim for about one per section or step, ten at most.

If the moment you need isn't a candidate, grab it at the transcript timestamp and look at it before using it:

```
ffmpeg -v error -y -ss <seconds> -i "<meta.video>" -frames:v 1 -q:v 2 "<out>/frames/extra_<mm-ss>.jpg"
```

## 5. Write

Write `<out>/note.md` with relative image links: `![what it shows](frames/007_02-15.jpg)`.

- Write in the language of the user's request unless told otherwise.
- Rewrite, don't transcribe: cut filler, merge repetition, keep exact names, commands, code, numbers and UI labels. When the transcript garbles code or a command, read it off the frame.
- Never invent steps, flags or results that aren't in the video.
- Link timestamps where useful (YouTube: add `t=<seconds>` to the URL's query string).
- Someone else's video: write it as notes with credit and a link, without long verbatim quotes.

### Blog post
```
# <Title that states the payoff>
<2–3 sentence hook: the problem and what the reader gets>
## <One section per chapter/topic>
<prose; an image where it helps>
## Key takeaways
- ...
Source: [<video title>](<url>)
```

### Study notes
```
# <Topic>: notes
**Source:** [<title>](<url>) · <speaker>
## TL;DR
- 3–5 bullets
## <Section per chapter> (mm:ss)
- key points; **terms** in bold; code and formulas in blocks
## Review questions
- ...
```

### How-to
```
# How to <task>
**You'll need:** <prereqs>
## Steps
1. <Imperative step>. <What you should see.>
   ![...](frames/...)
## Notes and pitfalls
- ...
```

### PR description
```
## What
## Why
## How to test
1. ...
## Screenshots
```

## 6. Deliver

- **No destination named**: `note.md` is the deliverable. Give its path.
- **Destination named** (Notion, Google Docs, Confluence, ...): publish with the connected tool for it. Don't write integration code or ask for API keys. If no such tool is connected, say so and deliver `note.md`.
- **Images**: upload them with the connector's file or attachment tool if it has one. Otherwise put `[Screenshot: frames/007_02-15.jpg: <caption>]` at each spot and tell the user which files to drag in.
- Publish only where the user asked.

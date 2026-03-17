"""
Text-to-Video Pipeline Module
Uses edge-tts for natural Microsoft voices (requires internet for TTS only)
"""

import os
import json
import asyncio
import textwrap
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mpe

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

LLM_MODEL       = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
OUTPUT_DIR      = "./videos"

# Edge TTS voices — natural Microsoft voices
VOICES = {
    "fr": "fr-FR-DeniseNeural",      # French female voice
    "en": "en-US-JennyNeural",       # English female voice
    "ar": "ar-SA-ZariyahNeural",     # Arabic female voice
}

SCRIPT_PROMPT = """Create a 5-slide educational video script in French.
Return ONLY a JSON array. Keep each field SHORT.

[
  {{"slide":1,"title":"Short title","points":["point1","point2","point3"],"narration":"One sentence only."}},
  {{"slide":2,"title":"Short title","points":["point1","point2","point3"],"narration":"One sentence only."}},
  {{"slide":3,"title":"Short title","points":["point1","point2","point3"],"narration":"One sentence only."}},
  {{"slide":4,"title":"Short title","points":["point1","point2","point3"],"narration":"One sentence only."}},
  {{"slide":5,"title":"Short title","points":["point1","point2","point3"],"narration":"One sentence only."}}
]

Topic summary (use this to fill in the content):
{context}

JSON array only, no explanation:"""


def repair_json(raw: str) -> list:
    """Try to repair truncated JSON by closing open structures."""
    import re
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    start = raw.find("[")
    if start == -1:
        return []
    raw = raw[start:]

    # Try parsing as-is first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Count open objects to repair
    depth       = 0
    in_string   = False
    escape_next = False
    last_good   = 0

    for i, ch in enumerate(raw):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_good = i + 1

    # Truncate to last complete object
    if last_good > 0:
        truncated = raw[:last_good]
        # Close the array
        truncated = truncated.rstrip().rstrip(",") + "]"
        try:
            result = json.loads(truncated)
            print(f"  ✅ Repaired JSON: {len(result)} slides")
            return result
        except json.JSONDecodeError:
            pass

    return []


def generate_script(context: str, model: str = LLM_MODEL, base_url: str = OLLAMA_BASE_URL) -> list:
    """Generate a 5-slide video script from course content."""
    # Limit context to avoid JSON truncation
    context = context[:1500]

    llm   = ChatOllama(model=model, base_url=base_url, temperature=0.1, num_predict=1200)
    prompt = ChatPromptTemplate.from_template(SCRIPT_PROMPT)
    chain  = prompt | llm | StrOutputParser()

    raw = chain.invoke({"context": context})

    slides = repair_json(raw)

    if not slides:
        print(f"⚠️ JSON parse failed, raw output:\n{raw[:400]}")
        # Return fallback slides so video still generates
        slides = [
            {"slide": i+1, "title": f"Partie {i+1}", "points": ["Point 1", "Point 2", "Point 3"], "narration": f"Voici la partie {i+1} du cours."}
            for i in range(5)
        ]
        print("  ↩️  Using fallback slides")

    print(f"✅ Generated {len(slides)} slides")
    return slides


def create_slide_image(slide_data: dict, slide_num: int, output_path: str,
                       width: int = 1280, height: int = 720) -> str:
    """Create a beautiful slide image."""

    BG_COLOR     = (12, 14, 28)
    ACCENT_COLOR = (99, 88, 255)
    TITLE_COLOR  = (255, 255, 255)
    POINT_COLOR  = (190, 200, 225)
    DIM_COLOR    = (60, 65, 90)

    img  = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Background gradient effect (horizontal bands)
    for y in range(height):
        alpha = int(8 * (y / height))
        draw.line([(0, y), (width, y)], fill=(12 + alpha, 14 + alpha, 40 + alpha))

    # Left accent bar
    draw.rectangle([0, 0, 6, height], fill=ACCENT_COLOR)

    # Top accent line
    draw.rectangle([0, 0, width, 3], fill=ACCENT_COLOR)

    # Slide number circle
    draw.ellipse([40, 35, 90, 85], fill=ACCENT_COLOR)

    # Load fonts
    try:
        font_num    = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 26)
        font_title  = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 52)
        font_point  = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 28)
        font_footer = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
    except Exception:
        font_num = font_title = font_point = font_footer = ImageFont.load_default()

    # Slide number
    draw.text((54, 48), str(slide_num), fill="white", font=font_num)

    # Title
    title = slide_data.get("title", "")
    draw.text((30, 110), title, fill=TITLE_COLOR, font=font_title)

    # Separator
    draw.rectangle([30, 180, width - 30, 183], fill=ACCENT_COLOR)

    # Bullet points
    y = 210
    for point in slide_data.get("points", [])[:3]:
        # Arrow bullet
        draw.polygon([(30, y+10), (30, y+22), (42, y+16)], fill=ACCENT_COLOR)
        wrapped = textwrap.fill(point, width=72)
        draw.text((55, y), wrapped, fill=POINT_COLOR, font=font_point)
        lines = wrapped.count('\n') + 1
        y += 60 + (lines - 1) * 30

    # Footer bar
    draw.rectangle([0, height - 45, width, height], fill=(20, 22, 45))
    draw.text((30, height - 30), "🧠 DocMind — AI Course System", fill=DIM_COLOR, font=font_footer)
    draw.text((width - 200, height - 30), f"Slide {slide_num} / 5", fill=DIM_COLOR, font=font_footer)

    img.save(output_path, quality=95)
    return output_path


async def tts_edge(text: str, output_path: str, lang: str = "fr") -> bool:
    """Generate natural voice using edge-tts (Microsoft Neural voices)."""
    try:
        import edge_tts
        voice = VOICES.get(lang, VOICES["fr"])
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        print(f"    🔊 TTS OK [{voice}]")
        return True
    except Exception as e:
        print(f"    ⚠️ Edge TTS failed: {e}")
        return False


def text_to_speech(text: str, output_path: str, lang: str = "fr") -> bool:
    """Run async edge-tts in sync context."""
    try:
        asyncio.run(tts_edge(text, output_path, lang))
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        print(f"    ⚠️ TTS error: {e}")
        return False


def fallback_tts_espeak(text: str, output_path: str, lang: str = "fr") -> bool:
    """Fallback to espeak if edge-tts fails."""
    wav_path = output_path.replace(".mp3", ".wav")
    try:
        subprocess.run([
            "espeak", "-v", lang, "-s", "120", "-a", "160",
            "-g", "10", "-w", wav_path, text[:400]
        ], check=True, capture_output=True)

        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path,
            "-af", "apad=pad_dur=1",
            "-codec:a", "libmp3lame", "-qscale:a", "2",
            output_path
        ], check=True, capture_output=True)

        if os.path.exists(wav_path):
            os.remove(wav_path)
        return True
    except Exception as e:
        print(f"    ⚠️ espeak fallback failed: {e}")
        return False


def create_silent_audio(output_path: str, duration: float = 8.0):
    """Create silent audio as last resort fallback."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(duration),
            "-codec:a", "libmp3lame", output_path
        ], check=True, capture_output=True)
    except Exception as e:
        print(f"    ⚠️ Silent audio failed: {e}")


def detect_language(text: str) -> str:
    """Detect language for TTS voice selection."""
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    if arabic_chars > len(text) * 0.3:
        return "ar"
    french_words = ["le", "la", "les", "un", "une", "est", "sont", "pour", "avec", "dans"]
    if any(f" {w} " in text.lower() for w in french_words):
        return "fr"
    return "fr"


def create_video(
    chapter_name: str,
    context: str,
    output_dir: str = OUTPUT_DIR,
    model: str = LLM_MODEL,
    base_url: str = OLLAMA_BASE_URL
) -> Optional[str]:
    """Full pipeline: context → script → slides + audio → video."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(output_dir) / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"📝 Generating script for: {chapter_name}...")
    slides = generate_script(context, model=model, base_url=base_url)

    if not slides:
        print("❌ No slides generated")
        return None

    video_clips    = []
    MIN_DURATION   = 8.0

    for i, slide in enumerate(slides, 1):
        print(f"  🖼️  Slide {i}: {slide.get('title', '')[:40]}...")

        # Create slide image
        slide_path = str(tmp_dir / f"slide_{i:02d}.png")
        create_slide_image(slide, i, slide_path)

        # Generate audio
        narration  = slide.get("narration", slide.get("title", f"Slide {i}"))
        lang       = detect_language(narration)
        audio_path = str(tmp_dir / f"audio_{i:02d}.mp3")

        # Try edge-tts first, fallback to espeak
        ok = text_to_speech(narration, audio_path, lang=lang)
        if not ok:
            print(f"  ↩️  Trying espeak fallback...")
            ok = fallback_tts_espeak(narration, audio_path, lang=lang)
        if not ok:
            print(f"  🔇 Using silence for slide {i}")
            create_silent_audio(audio_path, MIN_DURATION)

        # Build video clip
        try:
            audio_clip = mpe.AudioFileClip(audio_path)
            duration   = max(audio_clip.duration + 1.5, MIN_DURATION)
        except Exception:
            duration   = MIN_DURATION
            audio_clip = None

        image_clip = mpe.ImageClip(slide_path).set_duration(duration)
        if audio_clip:
            image_clip = image_clip.set_audio(audio_clip)
        video_clips.append(image_clip)

    if not video_clips:
        return None

    print("🎬 Combining into final video...")
    final      = mpe.concatenate_videoclips(video_clips, method="compose")
    safe_name  = chapter_name.replace(" ", "_").replace("/", "-")
    out_path   = str(Path(output_dir) / f"{safe_name}_course.mp4")

    final.write_videofile(out_path, fps=24, codec="libx264",
                          audio_codec="aac", logger=None)

    # Cleanup
    try:
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()
    except Exception:
        pass

    print(f"✅ Video saved: {out_path}")
    return out_path
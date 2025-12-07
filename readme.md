
# ShortFor

https://github.com/user-attachments/assets/997e1048-90a4-4ec8-8e7c-c12646143a5b

mTopic


ShortFormTopic is a tooling project for generating short-form videos from Reddit posts and AI-generated stories. It supports two major paths:

1. **Real-video workflow** – Stitching satisfying/loop videos + captions + TTS.  
2. **AI-video workflow** – Generating full video clips from Stable Diffusion / Mochi / ComfyUI pipelines, interpolating them with RIFE, upscaling them, and assembling narrative videos with synchronized captions.

This README explains:
- What the application does  
- How it works end-to-end  
- Detailed summaries of **all scripts** in the project  
- Newly added explanations extracted from project files  

> ⚠️ This repository calls external services (Azure Cognitive Services, ElevenLabs, Ollama, ffmpeg, Practical-RIFE, ComfyUI). You must configure API keys and local model paths before running.

---

## **Quick Overview**

**Purpose:** Produce short vertical videos by combining narration, captions, traditional clips, or fully AI-generated scenes.

**Inputs:**
- Scraped Reddit posts (video/text)
- AI-generated stories (Ollama or OpenAI)
- Local video assets (`data/satisfying_videos/`)
- AI models / ComfyUI workflows

**Outputs:**
- Final MP4 short videos (`data/out/`)
- Intermediate assets:
  - TTS audio + SRT (`data/TTS/`)
  - Generated stories (`data/stories/`)
  - AI-generated video frames/videos (`data/out/<id>/`)

---

# **Architecture & Workflow (Complete, Including Details From Uploaded Files)**

## **1. Reddit Scraping**
- `scripts/reddit.py` scrapes text & video posts.
- Writes `data/reddit_posts.csv`.
- Downloads Reddit videos to:
  - `data/satisfying_videos/regular`
  - `data/satisfying_videos/loops`

---

## **2. Story Generation**

Story generation is handled by `scripts/llama_generation.py` or `scripts/gpt_generation.py`.

`main.py` randomly chooses:
- a mode (AITA, AMA, SS)
- a theme / tone
- optional source novel or topic

Then it:
- Builds a complex system prompt using scraped Reddit stories
- Generates a **single block story + title**
- Saves it into `data/stories/<uuid>.json`

---

## **3. Text-to-Speech + Word Timing**

### **Azure TTS (`azure_synth.py`)**
- Synthesizes narration audio  
- Requests **word-level timestamps**  
- Uses **stable_whisper** to generate SRT/VTT  
- Provides helpers:
  - `get_tts()`
  - `create_transcript()`
  - `merge_srt_words_to_sentences()`

### **ElevenLabs (`eleven_labs.py`)**
- Lightweight ElevenLabs client  
- Returns audio + timing metadata  
- Hard-coded API key must be replaced with environment variable  

---

# **4. Real-Video Workflow (MoviePy Pipeline)**

### **Core file: `scripts/video_generator.py`**
Responsible for **everything** involving real video assets:

### **Video Selection Logic**
- Pulls from `regular/` and `loops/` folders  
- Per-frame **black pixel detection** to reject low-quality clips  
- Automatically loops short videos  
- Ensures total duration ≥ narration duration  

### **Caption Generation**
- Merges punctuation + hyphenated words  
- Groups words into time chunks (target duration)  
- Handles abbreviations like `AITA`  
- Provides animated effects:
  - Zoom-in  
  - Zoom-out  
  - Bounce/jump text  
  - Directional movement + compensation zoom  

### **Color Handling**
- Random complementary color pairs per scene  

### **Final Composition**
- Combines:
  - Background video clips  
  - Animated text clips  
  - Narration audio  
- Builds the final MP4 file  

---

# **5. AI Video Workflow (ComfyUI, Mochi, SDXL, Interpolation, Upscaling)**

The AI workflow is significantly more complex and is spread across:

| Step | Script |
|------|--------|
| Mochi/ComfyUI generation | `video_generation_workflow.py` |
| Story → Multi-part orchestration | `ai_video_creation.py` |
| RIFE interpolation | `video_interpolation.py` |
| RRDB upscaling | `video_upscaler.py` |

---

## **5.1 Mochi Video Generation (`video_generation_workflow.py`)**

This script runs *inside the ComfyUI folder*.

**Key operations:**
- Dynamically locates and loads:
  - ComfyUI core  
  - Custom nodes  
  - Model paths (via `extra_model_paths.yaml`)  
- Initializes:
  - CLIP text encoders  
  - Mochi UNet  
  - Mochi VAE  
  - Latent-video generator node  
- Builds a full graph:
  - `EmptyMochiLatentVideo`  
  - `KSampler`  
  - `MochiDecode`  
  - `VHS_VideoCombine`  

**CLI Arguments:**

--cuda-device
--id
--prompt
--part-index
--fps
--frames
--steps

This script outputs **raw low-FPS AI-generated clips**.

---

## **5.2 High-Level Orchestration (`ai_video_creation.py`)**

### **Mochi / ComfyUI Workflow**
- Splits the story into prompt parts (up to 6)  
- Runs two processes at a time using multiprocessing  
- Assigns GPUs `(0, 1)`  
- Calls `video_generation_workflow.py` with the part-specific prompt  
- Each part renders a 2 FPS clip  

### **Stable Diffusion XL Workflow**
- Using HuggingFace pipelines:
  - `AutoPipelineForText2Image`
  - `StableVideoDiffusionPipeline`
- Generates:
  - A seed image  
  - Video frames per story part  
- Saves everything into `data/out/<id>/`

### **Interpolation is triggered after generation**  
The script calls the global interpolation function when appropriate.

---

## **5.3 Multi-Stage FPS Interpolation (`video_interpolation.py`)**

This script wraps **Practical-RIFE** using a subprocess call.

### **Transformation sequence**
Often:
`[2, 4, 4]`
Meaning:
- 2 FPS → 4 FPS  
- 4 FPS → 16 FPS  
- 16 FPS → 64 FPS  

### **Behavior**
- Each stage writes to:
data/out/<id>/interpolized

- Intermediate files are deleted  
- Final file is renamed:

*interpolated<fps>fps.mp4


---

## **5.4 Frame Upscaling (`video_upscaler.py`)**

This script defines and runs a **full RRDBNet** architecture (BSRGAN/ESRGAN style).

### **Pipeline**
1. Load RRDBNet model  
2. Read input frames  
3. Run upscaling inference  
4. Write final HD MP4  

### **Why after interpolation?**
Because interpolation produces smooth motion but blurry intermediate frames — RRDBNet restores detail.

---

# **6. Global Orchestrator (`main.py`)**

`main.py` brings every component together.

### Workflow:
1. Optional Reddit scraping  
2. Story generation (LLM)  
3. TTS + timing generation  
4. Decide workflow:
 - **Real video → video_generator.py**
 - **AI video → ai_video_creation.py**
5. Optional interpolation  
6. Optional upscaling  
7. Caption overlay + audio mix  
8. Export MP4  
9. Mark post as generated in CSV  

---

# **Script Reference Summary**

| Script | Purpose |
|--------|---------|
| `azure_synth.py` | Azure TTS, Whisper alignment |
| `eleven_labs.py` | ElevenLabs TTS |
| `llama_generation.py` | Ollama story generator |
| `gpt_generation.py` | OpenAI story generator |
| `main.py` | End-to-end orchestrator |
| `video_generator.py` | Real video stitching & captioning |
| `ai_video_creation.py` | AI generation manager |
| `video_generation_workflow.py` | Mochi latent video generator |
| `video_interpolation.py` | Progressive RIFE interpolation |
| `video_upscaler.py` | RRDBNet upscaling |

---

# **Installation & Running**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Install:

ffmpeg

ImageMagick

Chrome + ChromeDriver

ComfyUI + Mochi models

Practical-RIFE
```
Run the pipeline:
```
python scripts\main.py
```
# **Notes, Risks, TODOs**

- ElevenLabs key is hard-coded (fix required)
 
- Many Windows paths are hard-coded (refactor to config file)

- GPU process failures currently stop whole workflow — add retry logic

- Quality of generation is entirely correlated to generation models. New versions should experiment with embedding Sora, Veo 3 and other higher quality image and video generation models.

- Some optimizations allow for usage on low-vram devices, but higher quality generations should be ran on 24+ GB cards.






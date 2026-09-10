import os
import sys
import warnings

# disable all unnecessary environment variables and warnings. 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import io
import time
import json
import signal
import queue
import threading
import requests
import numpy as np 
import pyaudio
import webrtcvad
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from transformers import MarianTokenizer, MarianMTModel

# --- 1. 信号拦截：实现按下 Ctrl+C 瞬时无延迟退出 ---
def signal_handler(sig, frame):
    print("\n\n🛑 收到中断指令，立即终止运行...")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --- 2. 自动修复 ALSA 路径 ---
if "ALSA_PLUGIN_DIR" not in os.environ:
    for alsa_path in ["/usr/lib/x86_64-linux-gnu/alsa-lib", "/usr/lib/alsa-lib", "/usr/lib64/alsa-lib"]:
        if os.path.exists(alsa_path):
            os.environ["ALSA_PLUGIN_DIR"] = alsa_path
            break

# --- 3. 初始化模型（Whisper + MarianMT 专用翻译模型） ---
print("⏳ Loading STT model (faster-whisper)...")
stt_model = WhisperModel("base.en", device="cpu", compute_type="int8")
print("✅ STT model loaded!")

print("⏳ Loading Translation model (Helsinki-NLP/opus-mt-en-zh)...")
MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"
translator_tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
translator_model = MarianMTModel.from_pretrained(MODEL_NAME).to("cpu")
print("✅ Translation model loaded!")

# 翻译函数 (消除了警告日志，强制指定 CPU)
def translate_text(text):
    if not text.strip():
        return ""
    inputs = translator_tokenizer(text, return_tensors="pt", padding=True)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    translated = translator_model.generate(**inputs, max_new_tokens=128, max_length=None)
    res = translator_tokenizer.batch_decode(translated, skip_special_tokens=True)
    return res[0]

# --- 3.1 动态句子与短语纠错服务 (Dynamic Sentence & Phrase Error Correction Model/Service) ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
CORRECTION_MODEL = os.environ.get("CORRECTION_MODEL", "qwen2.5:0.5b")
last_classroom_context = ""

def correct_sentence_dynamic(english_text, context_text=""):
    """
    通过 LLM / 纠错服务动态修正语音转写中的上下文错误、语法误读及课堂/学术短语。
    当服务不可用或超时时，平滑降级至规则规范化，保证实时流水线不断流。
    """
    if not english_text or not english_text.strip():
        return ""

    prompt = f"""You are a dynamic classroom speech correction system.
Input English speech transcription may contain speech recognition (STT) mishearings, acoustic errors, or ungrammatical phrasing.

Previous Classroom Context: "{context_text}"
Input Transcription: "{english_text}"

Tasks:
1. Dynamically fix speech mishearings, ungrammatical phrasing, or broken sentence structure based on context.
2. Preserve original meaning accurately while improving English grammar and standard classroom terminology.

Output ONLY a JSON object matching this schema:
{{
  "corrected": "<dynamically corrected English sentence or phrase>"
}}
"""

    payload = {
        "model": CORRECTION_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 96
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=2.0)
        if response.status_code == 200:
            res_json = json.loads(response.json().get("response", "{}"))
            corrected = res_json.get("corrected", "").strip()
            if corrected:
                return corrected
    except Exception:
        pass

    # 服务不可用或超时时的平滑降级处理
    corrected = english_text.strip()
    replacements = {
        "i triple e": "IEEE",
        "python three": "Python 3",
        "macro economics": "macroeconomics",
        "gpu": "GPU"
    }
    for raw, fix in replacements.items():
        corrected = corrected.replace(raw, fix)
    
    if corrected and not corrected[0].isupper():
        corrected = corrected[0].upper() + corrected[1:]
    return corrected

# --- 4. 初始化音频设备 ---
FORMAT = pyaudio.paInt16
TARGET_RATE = 16000
FRAME_DURATION_MS = 30
TARGET_CHUNK = int(TARGET_RATE * FRAME_DURATION_MS / 1000)

audio = pyaudio.PyAudio()
default_device_index = None
try:
    default_info = audio.get_default_input_device_info()
    default_device_index = default_info.get("index")
except Exception:
    pass

stream = None
CHANNELS, HW_RATE = 2, 44100
for try_channels in [2, 1]:
    for try_rate in [44100, 48000, 16000]:
        try:
            chunk_size = int(try_rate * FRAME_DURATION_MS / 1000)
            stream = audio.open(
                format=FORMAT,
                channels=try_channels,
                rate=try_rate,
                input=True,
                frames_per_buffer=chunk_size,
                input_device_index=default_device_index
            )
            CHANNELS, HW_RATE = try_channels, try_rate
            break
        except Exception:
            continue
    if stream is not None:
        break

if stream is None:
    print("❌ 无法打开任何麦克风设备，请检查设备连接！")
    sys.exit(1)

vad = webrtcvad.Vad(2)  # 敏感度设置为 2，更好防误触
MAX_SILENCE_FRAMES = 30  # ~900ms 静音判定断句
MAX_SPEECH_FRAMES = 300  # 最长 9 秒强制切片

# 异步处理队列
task_queue = queue.Queue()

# --- 5. 后台工作线程：专门处理模型推理 ---
def worker():
    global last_classroom_context
    while True:
        audio_data = task_queue.get()
        if audio_data is None:
            break

        buf_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = buf_int16.astype(np.float32) / 32768.0

        # ⚡ 抗幻觉 1：音频均方根能量检测（过滤环境底噪、呼吸声）
        audio_energy = np.sqrt(np.mean(audio_float32**2))
        if audio_energy < 0.01:
            task_queue.task_done()
            continue

        # 1. 语音转文字 (STT) - 利用 initial_prompt 传入上一句课堂上下文，保持语意连贯
        t_stt_start = time.time()
        segments, _ = stt_model.transcribe(
            audio_float32, 
            beam_size=1,                         # 提速核心
            language="en",
            initial_prompt=last_classroom_context, # 动态上下文提示
            no_speech_threshold=0.6,             # ⚡ 抗幻觉 2：静音概率>60%直接跳过
            log_prob_threshold=-1.0,             # ⚡ 抗幻觉 3：修正拼写为 log_prob_threshold
            compression_ratio_threshold=2.4       # ⚡ 抗幻觉 4：防止重复打印特定单词
        )
        transcription = " ".join([segment.text.strip() for segment in segments]).strip()
        stt_delay = time.time() - t_stt_start

        # 2. 动态模型/服务句子与短语纠错 + 专用模型快速翻译
        if transcription:
            t_trans_start = time.time()
            corrected_en = correct_sentence_dynamic(transcription, context_text=last_classroom_context)
            translated_zh = translate_text(corrected_en)            
            trans_delay = time.time() - t_trans_start

            last_classroom_context = corrected_en # 动态更新上下文记录

            print(f"\n🎙️ [STT English ]: {transcription}")
            print(f"✏️ [Dynamic Correct]: {corrected_en}")
            print(f"🌐 [Translation ]: {translated_zh}")
            print(f"⏱️ STT: {stt_delay:.2f}s | Trans: {trans_delay:.2f}s | Total: {stt_delay + trans_delay:.2f}s")
            print("-" * 65 + "\n")

        task_queue.task_done()

# 启动推理线程
threading.Thread(target=worker, daemon=True).start()

print("🎤 All systems ready! (Press Ctrl+C to stop instantly)")
print("=" * 65)

# --- 6. 主音频采集循环 (永不阻塞) ---
speech_buffer = []
silence_frames = 0
HW_CHUNK = int(HW_RATE * FRAME_DURATION_MS / 1000)

try:
    while True:
        frame_raw = stream.read(HW_CHUNK, exception_on_overflow=False)
        frame_np = np.frombuffer(frame_raw, dtype=np.int16)

        if CHANNELS == 2:
            frame_mono = frame_np.reshape(-1, 2).mean(axis=1).astype(np.int16)
        else:
            frame_mono = frame_np

        if HW_RATE != TARGET_RATE:
            frame_16k = resample_poly(frame_mono, TARGET_RATE, HW_RATE).astype(np.int16)
        else:
            frame_16k = frame_mono

        if len(frame_16k) < TARGET_CHUNK:
            frame_16k = np.pad(frame_16k, (0, TARGET_CHUNK - len(frame_16k)))
        elif len(frame_16k) > TARGET_CHUNK:
            frame_16k = frame_16k[:TARGET_CHUNK]

        frame_bytes = frame_16k.tobytes()
        is_speech = vad.is_speech(frame_bytes, TARGET_RATE)

        if is_speech:
            speech_buffer.append(frame_bytes)
            silence_frames = 0
            print("\r🔊 [speaking...]", end="", flush=True)
        else:
            silence_frames += 1
            if len(speech_buffer) > 0:
                print("\r.  [waiting for pause...]", end="", flush=True)

        # 触发切片：静音超时 或 连续说话时间过长
        if (silence_frames > MAX_SILENCE_FRAMES or len(speech_buffer) > MAX_SPEECH_FRAMES) and len(speech_buffer) > 0:
            if len(speech_buffer) > 15:  # 过滤低于 450ms 的杂音
                task_queue.put(b''.join(speech_buffer))
            speech_buffer = []
            silence_frames = 0

except Exception as e:
    print(f"\nError: {e}")
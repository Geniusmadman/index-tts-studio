import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--root_path", type=str, default=None, help="Root path when the web UI is mounted behind a reverse proxy")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="zh_CN")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )
# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]  # skip experimental features
EMOTION_LABELS = ["喜", "怒", "哀", "惧", "厌恶", "低落", "惊喜", "平静"]

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)
SAVED_VOICES_DIR = os.path.join("prompts", "saved_voices")
os.makedirs(SAVED_VOICES_DIR, exist_ok=True)
SMART_LLM_CONFIG_PATH = os.path.join("prompts", "smart_llm_config.json")
NETWORK_SOURCE_CONFIG_PATH = os.path.join("prompts", "network_source_config.json")
MAX_UPLOAD_SIZE_MB = int(os.getenv("INDEXTTS_MAX_UPLOAD_MB", "1024"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
FFMPEG_CANDIDATES = [
    "ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/usr/bin/ffmpeg",
]
EXTRA_EXEC_PATHS = [
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
]

def ensure_exec_paths():
    paths = [path for path in os.environ.get("PATH", "").split(os.pathsep) if path]
    changed = False
    for path in EXTRA_EXEC_PATHS:
        if os.path.isdir(path) and path not in paths:
            paths.insert(0, path)
            changed = True
    if changed:
        os.environ["PATH"] = os.pathsep.join(paths)

ensure_exec_paths()

def resolve_ffmpeg_path():
    for candidate in FFMPEG_CANDIDATES:
        resolved = shutil.which(candidate) if os.path.basename(candidate) == candidate else candidate
        if resolved and os.path.exists(resolved) and os.access(resolved, os.X_OK):
            return resolved
    return None

def default_smart_llm_config():
    return {
        "analysis_mode": "OpenAI兼容大模型" if os.getenv("OPENAI_API_KEY") else "本地规则",
        "api_base": os.getenv("SMART_EMOTION_API_BASE", "https://api.openai.com/v1/chat/completions"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("SMART_EMOTION_MODEL", "gpt-4o-mini"),
        "use_proxy": False,
        "proxy": os.getenv("SMART_EMOTION_PROXY", os.getenv("HTTPS_PROXY", os.getenv("HTTP_PROXY", ""))),
        "pause_ms": 180,
    }

def load_smart_llm_config():
    config = default_smart_llm_config()
    if not os.path.exists(SMART_LLM_CONFIG_PATH):
        return config
    try:
        with open(SMART_LLM_CONFIG_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            config.update({key: value for key, value in saved.items() if key in config})
    except Exception as exc:
        print(f"Failed to load smart LLM config: {exc}")
    return config

def save_smart_llm_config(analysis_mode, api_base, api_key, model, use_proxy, proxy, pause_ms, keep_existing_key=False):
    saved_config = load_smart_llm_config() if keep_existing_key else {}
    next_api_key = api_key if api_key else saved_config.get("api_key", "")
    config = {
        "analysis_mode": analysis_mode or "本地规则",
        "api_base": (api_base or "").strip() or "https://api.openai.com/v1/chat/completions",
        "api_key": next_api_key or "",
        "model": (model or "").strip() or "gpt-4o-mini",
        "use_proxy": bool(use_proxy),
        "proxy": (proxy or "").strip(),
        "pause_ms": int(pause_ms or 180),
    }
    try:
        os.makedirs(os.path.dirname(SMART_LLM_CONFIG_PATH), exist_ok=True)
        with open(SMART_LLM_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return "大模型配置已保存。"
    except Exception as exc:
        print(f"Failed to save smart LLM config: {exc}")
        return "大模型配置保存失败，请检查 prompts 目录权限。"

def load_smart_llm_config_for_ui():
    config = load_smart_llm_config()
    return (
        gr.update(value=config["analysis_mode"]),
        gr.update(value=config["api_base"]),
        gr.update(value=""),
        gr.update(value=config["model"]),
        gr.update(value=config["use_proxy"]),
        gr.update(value=config["proxy"], interactive=bool(config["use_proxy"])),
        gr.update(value=config["pause_ms"]),
        "已载入上次保存的大模型配置。API Key 已保存在本机，页面不会回填显示。",
    )

def on_smart_proxy_toggle(use_proxy, analysis_mode, api_base, api_key, model, proxy, pause_ms):
    status = save_smart_llm_config(analysis_mode, api_base, api_key, model, use_proxy, proxy, pause_ms, keep_existing_key=True)
    return gr.update(interactive=bool(use_proxy)), status

def save_smart_llm_config_keep_key(analysis_mode, api_base, api_key, model, use_proxy, proxy, pause_ms):
    return save_smart_llm_config(
        analysis_mode,
        api_base,
        api_key,
        model,
        use_proxy,
        proxy,
        pause_ms,
        keep_existing_key=True,
    )

def default_network_source_config():
    return {
        "use_proxy": False,
        "proxy": os.getenv("NETWORK_SOURCE_PROXY", os.getenv("HTTPS_PROXY", os.getenv("HTTP_PROXY", ""))),
        "cookies_file": os.getenv("NETWORK_SOURCE_COOKIES", ""),
    }

def load_network_source_config():
    config = default_network_source_config()
    if not os.path.exists(NETWORK_SOURCE_CONFIG_PATH):
        return config
    try:
        with open(NETWORK_SOURCE_CONFIG_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            config.update({key: value for key, value in saved.items() if key in config})
    except Exception as exc:
        print(f"Failed to load network source config: {exc}")
    return config

def save_network_source_config(use_proxy, proxy, cookies_file):
    config = {
        "use_proxy": bool(use_proxy),
        "proxy": (proxy or "").strip(),
        "cookies_file": (cookies_file or "").strip(),
    }
    try:
        os.makedirs(os.path.dirname(NETWORK_SOURCE_CONFIG_PATH), exist_ok=True)
        with open(NETWORK_SOURCE_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return "网络素材配置已保存。"
    except Exception as exc:
        print(f"Failed to save network source config: {exc}")
        return "网络素材配置保存失败，请检查 prompts 目录权限。"

def on_network_proxy_toggle(use_proxy, proxy, cookies_file):
    status = save_network_source_config(use_proxy, proxy, cookies_file)
    return gr.update(interactive=bool(use_proxy)), status

MAX_LENGTH_TO_USE_SPEED = 70
example_cases = []
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None

        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES_ALL[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0),
                             ])

def get_example_cases(include_experimental = False):
    if include_experimental:
        return example_cases  # show every example

    # exclude emotion control mode 3 (emotion from text description)
    return [x for x in example_cases if x[1] != EMO_CHOICES_ALL[3]]

def format_glossary_markdown():
    """将词汇表转换为Markdown表格格式"""
    if not tts.normalizer.term_glossary:
        return i18n("暂无术语")

    lines = [f"| {i18n('术语')} | {i18n('中文读法')} | {i18n('英文读法')} |"]
    lines.append("|---|---|---|")

    for term, reading in tts.normalizer.term_glossary.items():
        zh = reading.get("zh", "") if isinstance(reading, dict) else reading
        en = reading.get("en", "") if isinstance(reading, dict) else reading
        lines.append(f"| {term} | {zh} | {en} |")

    return "\n".join(lines)

def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                       **kwargs)
    return gr.update(value=output,visible=True)

def build_generation_kwargs(args):
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    return {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }

def split_text_segments(text, max_text_tokens_per_segment):
    if not text or not text.strip():
        return []
    text_tokens_list = tts.tokenizer.tokenize(text)
    segments = tts.tokenizer.split_segments(
        text_tokens_list,
        max_text_tokens_per_segment=int(max_text_tokens_per_segment),
    )
    result = []
    for segment in segments:
        segment_text = detokenize_segment(segment)
        if segment_text:
            result.append(segment_text)
    return result

def detokenize_segment(tokens):
    text = "".join(tokens).replace("▁", " ").strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？、；：,.!?;:])", r"\1", text)
    text = re.sub(r"([（“‘])\s+", r"\1", text)
    text = re.sub(r"\s+([）”’])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def clamp_float(value, default=0.0, min_value=0.0, max_value=1.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(min_value, min(max_value, number))

def normalize_smart_emotion(item):
    vector = item.get("emo_vector") or item.get("vector") or []
    if not isinstance(vector, list):
        vector = []
    vector = [clamp_float(value) for value in vector[:8]]
    while len(vector) < 8:
        vector.append(0.0)

    if sum(vector) <= 0.01:
        vector[7] = 0.75

    return {
        "emotion": str(item.get("emotion") or item.get("label") or "平静")[:20],
        "emo_vector": vector,
        "emo_alpha": clamp_float(item.get("emo_alpha") or item.get("intensity"), default=0.65, min_value=0.15, max_value=1.0),
        "reason": str(item.get("reason") or "")[:80],
    }

def heuristic_emotion_for_segment(segment):
    text = segment or ""
    vector = [0.0] * 8
    emotion = "平静"
    reason = "本地规则判断"

    keyword_rules = [
        (0, "喜", ["开心", "高兴", "快乐", "幸福", "温暖", "喜欢", "成功", "胜利", "笑", "美好"]),
        (1, "怒", ["愤怒", "生气", "怒", "斥责", "抗议", "不满", "冲突", "争吵"]),
        (2, "哀", ["悲伤", "难过", "哭", "失去", "离别", "遗憾", "痛苦", "孤独", "牺牲"]),
        (3, "惧", ["害怕", "恐惧", "危险", "紧张", "惊慌", "威胁", "担心", "悬念"]),
        (4, "厌恶", ["厌恶", "讨厌", "恶心", "肮脏", "腐败", "鄙夷"]),
        (5, "低落", ["低落", "疲惫", "无奈", "沉重", "失望", "压抑", "迷茫"]),
        (6, "惊喜", ["突然", "惊喜", "震惊", "没想到", "竟然", "意外", "惊讶"]),
    ]
    for index, label, keywords in keyword_rules:
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits:
            vector[index] += min(0.85, 0.35 + hits * 0.18)
            emotion = label
            reason = f"命中“{keywords[0]}”等情绪线索"

    if "！" in text or "!" in text:
        strongest = max(range(7), key=lambda i: vector[i])
        vector[strongest] = min(1.0, max(vector[strongest], 0.55) + 0.12)
    if "？" in text or "?" in text:
        vector[3] = max(vector[3], 0.28)
        if emotion == "平静":
            emotion = "惧"
            reason = "疑问句增加不确定感"

    if sum(vector) <= 0.01:
        vector[7] = 0.75
        reason = "未发现明显情绪，保持平静叙述"

    intensity = min(0.95, max(0.45, sum(vector)))
    return normalize_smart_emotion({
        "emotion": emotion,
        "emo_vector": vector,
        "emo_alpha": intensity,
        "reason": reason,
    })

def extract_json_array(text):
    if not text:
        raise ValueError("empty LLM response")
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if match:
        text = match.group(1)
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)

def analyze_segments_with_llm(segments, api_base, api_key, model, proxy):
    import requests

    api_base = (api_base or "").strip() or "https://api.openai.com/v1/chat/completions"
    model = (model or "").strip() or "gpt-4o-mini"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是中文有声书导演。请分析每段文本的朗读情绪，并只返回 JSON 数组。"
                    "每项包含 index, emotion, emo_vector, emo_alpha, reason。"
                    "emo_vector 必须是 8 个 0~1 数字，顺序固定为：喜、怒、哀、惧、厌恶、低落、惊喜、平静。"
                    "emo_alpha 是 0.15~1.0 的情绪强度。不要返回 Markdown。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    [{"index": i + 1, "text": segment} for i, segment in enumerate(segments)],
                    ensure_ascii=False,
                ),
            },
        ],
    }
    proxies = None
    if proxy and proxy.strip():
        proxies = {"http": proxy.strip(), "https": proxy.strip()}

    response = requests.post(api_base, headers=headers, json=payload, proxies=proxies, timeout=90)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    parsed = extract_json_array(content)

    results = [None] * len(segments)
    for item in parsed:
        try:
            index = int(item.get("index", 0)) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(results):
            results[index] = normalize_smart_emotion(item)

    for i, result in enumerate(results):
        if result is None:
            results[i] = heuristic_emotion_for_segment(segments[i])
            results[i]["reason"] = "大模型未返回该段，使用本地规则补齐"
    return results

def analyze_segments(segments, mode, api_base, api_key, model, use_proxy, proxy):
    if mode == "OpenAI兼容大模型" and api_key and api_key.strip():
        return analyze_segments_with_llm(segments, api_base, api_key, model, proxy if use_proxy else "")
    return [heuristic_emotion_for_segment(segment) for segment in segments]

def create_silence_wav(path, pause_ms):
    pause_ms = max(0, int(pause_ms or 0))
    if pause_ms <= 0:
        return None
    duration = pause_ms / 1000
    ffmpeg_path = resolve_ffmpeg_path()
    if not ffmpeg_path:
        raise FileNotFoundError("ffmpeg")
    command = [
        ffmpeg_path, "-y",
        "-f", "lavfi",
        "-i", "anullsrc=r=22050:cl=mono",
        "-t", f"{duration:.3f}",
        "-acodec", "pcm_s16le",
        path,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return path

def merge_audio_files(audio_paths, output_path, pause_ms=180):
    if not audio_paths:
        raise ValueError("no audio files to merge")

    with tempfile.TemporaryDirectory() as tmp_dir:
        concat_paths = []
        silence_path = create_silence_wav(os.path.join(tmp_dir, "silence.wav"), pause_ms)
        for index, audio_path in enumerate(audio_paths):
            concat_paths.append(audio_path)
            if silence_path and index < len(audio_paths) - 1:
                concat_paths.append(silence_path)

        list_path = os.path.join(tmp_dir, "concat.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for path in concat_paths:
                safe_path = os.path.abspath(path).replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        command = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path,
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
    return output_path

def gen_smart_segments(prompt, text, analysis_mode, api_base, api_key, model, use_proxy, proxy,
                       pause_ms, max_text_tokens_per_segment=120,
                       *args, progress=gr.Progress()):
    save_smart_llm_config(analysis_mode, api_base, api_key, model, use_proxy, proxy, pause_ms, keep_existing_key=True)
    if not api_key:
        api_key = load_smart_llm_config().get("api_key", "")
    if not prompt:
        gr.Warning(i18n("请先上传或选择参考音色"))
        return gr.update(), []
    if not text or not text.strip():
        gr.Warning(i18n("请输入朗读文本"))
        return gr.update(), []

    segments = split_text_segments(text, max_text_tokens_per_segment)
    if not segments:
        gr.Warning(i18n("没有可生成的文本段落"))
        return gr.update(), []

    progress(0.02, desc="正在分析文本情绪...")
    try:
        emotions = analyze_segments(segments, analysis_mode, api_base, api_key, model, use_proxy, proxy)
    except Exception as exc:
        print(f"LLM emotion analysis failed, fallback to heuristic: {exc}")
        gr.Warning(i18n("大模型分析失败，已改用本地规则继续生成"))
        emotions = [heuristic_emotion_for_segment(segment) for segment in segments]

    task_dir = os.path.join("outputs", "tasks", f"smart_{int(time.time())}")
    os.makedirs(task_dir, exist_ok=True)
    kwargs = build_generation_kwargs(args)
    segment_paths = []
    table = []

    for index, (segment, emotion) in enumerate(zip(segments, emotions), start=1):
        progress(0.08 + 0.84 * (index - 1) / len(segments), desc=f"正在生成第 {index}/{len(segments)} 段...")
        segment_path = os.path.join(task_dir, f"segment_{index:03d}.wav")
        emo_vector = tts.normalize_emo_vec(emotion["emo_vector"], apply_bias=True)
        tts.infer(
            spk_audio_prompt=prompt,
            text=segment,
            output_path=segment_path,
            emo_audio_prompt=None,
            emo_alpha=emotion["emo_alpha"],
            emo_vector=emo_vector,
            use_random=False,
            verbose=cmd_args.verbose,
            max_text_tokens_per_segment=int(max_text_tokens_per_segment),
            **kwargs,
        )
        segment_paths.append(segment_path)
        table.append([
            index,
            segment,
            emotion["emotion"],
            round(emotion["emo_alpha"], 2),
            ", ".join(f"{label}:{value:.2f}" for label, value in zip(EMOTION_LABELS, emotion["emo_vector"])),
            emotion["reason"],
        ])

    progress(0.95, desc="正在合并音频...")
    output_path = os.path.join("outputs", f"smart_spk_{int(time.time())}.wav")
    try:
        merge_audio_files(segment_paths, output_path, pause_ms=pause_ms)
    except FileNotFoundError:
        gr.Error(i18n("未找到 ffmpeg，无法合并分段音频"))
        return gr.update(), table
    except subprocess.CalledProcessError as exc:
        print(f"Failed to merge audio: {exc.stderr or exc.stdout}")
        gr.Error(i18n("音频合并失败"))
        return gr.update(), table

    progress(1.0, desc="完成")
    return gr.update(value=output_path, visible=True), table

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def sanitize_voice_name(name):
    if not name:
        return ""
    safe_name = re.sub(r'[\\/:*?"<>|]+', "_", name.strip())
    safe_name = re.sub(r"\s+", "_", safe_name)
    return safe_name.strip("._ ")[:80]

def list_saved_voices():
    if not os.path.isdir(SAVED_VOICES_DIR):
        return []

    voices = []
    for filename in sorted(os.listdir(SAVED_VOICES_DIR)):
        path = os.path.join(SAVED_VOICES_DIR, filename)
        if os.path.isfile(path) and filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
            voices.append(os.path.splitext(filename)[0])
    return voices

def get_saved_voice_path(voice_name):
    safe_name = sanitize_voice_name(voice_name)
    if not safe_name:
        return None

    for filename in os.listdir(SAVED_VOICES_DIR):
        stem, ext = os.path.splitext(filename)
        if stem == safe_name and ext.lower() in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
            return os.path.join(SAVED_VOICES_DIR, filename)
    return None

def save_current_voice(voice_name, audio_path):
    if not audio_path:
        gr.Warning(i18n("请先上传或提取一个参考音色"))
        return gr.update(choices=list_saved_voices()), "请先上传或提取一个参考音色。"

    if isinstance(audio_path, dict):
        audio_path = audio_path.get("name") or audio_path.get("path")

    if not audio_path or not os.path.exists(audio_path):
        gr.Error(i18n("找不到当前参考音频文件"))
        return gr.update(choices=list_saved_voices()), "找不到当前参考音频文件。"

    safe_name = sanitize_voice_name(voice_name)
    if not safe_name:
        gr.Warning(i18n("请给这个音色起一个名字"))
        return gr.update(choices=list_saved_voices()), "请给这个音色起一个名字。"

    ext = os.path.splitext(audio_path)[1].lower() or ".wav"
    if ext not in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
        ext = ".wav"

    os.makedirs(SAVED_VOICES_DIR, exist_ok=True)
    target_path = os.path.join(SAVED_VOICES_DIR, f"{safe_name}{ext}")
    shutil.copy2(audio_path, target_path)
    choices = list_saved_voices()
    gr.Info(i18n("已保存到常用音色库"), duration=2)
    return gr.update(choices=choices, value=safe_name), f"已保存：`{target_path}`"

def load_saved_voice(voice_name):
    if not voice_name:
        return gr.update(), "请选择一个已保存音色。"

    voice_path = get_saved_voice_path(voice_name)
    if not voice_path:
        gr.Error(i18n("找不到已保存的音色文件"))
        return gr.update(), "找不到已保存的音色文件。"

    gr.Info(i18n("已设为参考音色"), duration=2)
    return gr.update(value=voice_path), f"当前参考音色：`{voice_path}`"

def refresh_saved_voices():
    choices = list_saved_voices()
    status = "已刷新常用音色库。" if choices else "还没有保存常用音色。"
    return gr.update(choices=choices, value=None), status

def get_upload_path(file_value):
    if isinstance(file_value, dict):
        return file_value.get("name") or file_value.get("path")
    return file_value

def format_file_size(path):
    try:
        size = os.path.getsize(path)
    except OSError:
        return "未知大小"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"

def is_file_too_large(path):
    try:
        return os.path.getsize(path) > MAX_UPLOAD_SIZE_BYTES
    except OSError:
        return False

def voice_status_html(kind, title, detail=""):
    detail_html = f"<p>{html.escape(detail)}</p>" if detail else ""
    return (
        f'<div class="voice-status-card is-{kind}">'
        f'<span class="voice-status-dot"></span>'
        f'<div><strong>{html.escape(title)}</strong>{detail_html}</div>'
        f'</div>'
    )

def local_voice_upload_feedback(media_file):
    media_path = get_upload_path(media_file)
    if not media_path:
        return gr.update(), voice_status_html("idle", "等待上传", f"请选择一段本地音频或视频。单个文件最大 {MAX_UPLOAD_SIZE_MB} MB。")
    filename = os.path.basename(media_path)
    size_label = format_file_size(media_path) if os.path.exists(media_path) else "等待服务端接收"
    if os.path.exists(media_path) and is_file_too_large(media_path):
        return gr.update(value=None), voice_status_html("error", "文件超过大小限制", f"{filename} · {size_label}。单个文件最大 {MAX_UPLOAD_SIZE_MB} MB。")
    return gr.update(), voice_status_html("ready", "上传已完成", f"{filename} · {size_label}。点击“处理并用作音色”开始抽取音轨。")

def mark_local_processing(media_file):
    if not media_file:
        return gr.update(interactive=True), voice_status_html("warn", "还没有上传素材", "请先选择本地音频或视频文件。")
    media_path = get_upload_path(media_file)
    if media_path and os.path.exists(media_path) and is_file_too_large(media_path):
        return gr.update(interactive=True), voice_status_html("error", "文件超过大小限制", f"单个文件最大 {MAX_UPLOAD_SIZE_MB} MB，请先压缩或截取素材。")
    return gr.update(interactive=False), voice_status_html("busy", "正在处理本地素材", "正在抽取音轨并转换为 24kHz 单声道 WAV，请保持页面打开。")

def mark_network_processing(url, start_time, end_time):
    url = (url or "").strip()
    if not url:
        return gr.update(interactive=True), voice_status_html("warn", "还没有填写链接", "请先粘贴 B站、抖音、小红书等素材链接。")
    try:
        start_seconds, end_seconds, requested_end = normalize_network_clip_range(start_time, end_time)
        detail = f"准备解析并截取 {format_clip_time(start_seconds)} - {format_clip_time(end_seconds)}。"
        if requested_end != end_seconds:
            detail += " 你设置的跨度超过 60 秒，系统会自动限制为 60 秒。"
    except ValueError:
        detail = "正在校验时间格式。若格式错误，下一步会给出提示。"
    return gr.update(interactive=False), voice_status_html("busy", "正在解析网络素材", detail)

def restore_processing_button():
    return gr.update(interactive=True)

def cancel_local_voice_task():
    return (
        gr.update(value=None),
        gr.update(interactive=True),
        voice_status_html("warn", "已取消上传或处理", f"已清空本地素材选择。单个文件最大 {MAX_UPLOAD_SIZE_MB} MB。"),
    )

def convert_media_to_prompt_audio(media_path, prefix):
    if not media_path or not os.path.exists(media_path):
        raise FileNotFoundError(media_path or "")
    ffmpeg_path = resolve_ffmpeg_path()
    if not ffmpeg_path:
        raise FileNotFoundError("ffmpeg")
    os.makedirs("prompts", exist_ok=True)
    output_path = os.path.join("prompts", f"{prefix}_{int(time.time())}.wav")
    command = [
        ffmpeg_path,
        "-y",
        "-i", media_path,
        "-vn",
        "-ac", "1",
        "-ar", "24000",
        "-acodec", "pcm_s16le",
        output_path,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output_path

def process_local_voice_media(media_file, progress=gr.Progress()):
    if not media_file:
        gr.Warning(i18n("请先上传本地音频或视频"))
        return gr.update(), voice_status_html("warn", "请先上传本地音频或视频")

    media_path = get_upload_path(media_file)
    if not media_path or not os.path.exists(media_path):
        gr.Error(i18n("找不到上传的媒体文件"))
        return gr.update(), voice_status_html("error", "找不到上传的媒体文件", "请重新选择文件后再处理。")
    if is_file_too_large(media_path):
        return gr.update(), voice_status_html("error", "文件超过大小限制", f"单个文件最大 {MAX_UPLOAD_SIZE_MB} MB。")

    try:
        progress(0.15, desc="正在读取上传文件...")
        progress(0.45, desc="正在抽取音轨并转换格式...")
        output_path = convert_media_to_prompt_audio(media_path, "local_ref")
        progress(0.9, desc="正在设置参考音色...")
    except FileNotFoundError:
        gr.Error(i18n("未找到 ffmpeg，请先安装 ffmpeg"))
        return gr.update(), voice_status_html("error", "未找到 ffmpeg", "请先安装 ffmpeg 后再处理音视频。")
    except subprocess.CalledProcessError as exc:
        error_message = (exc.stderr or exc.stdout or "").strip()
        print(f"Failed to process local media: {error_message}")
        gr.Error(i18n("音视频处理失败，请确认文件包含可读取的音轨"))
        return gr.update(), voice_status_html("error", "音视频处理失败", "请确认文件包含可读取的音轨，或先剪成较短的音频再上传。")

    gr.Info(i18n("已设为参考音色"), duration=2)
    progress(1.0, desc="完成")
    return gr.update(value=output_path), voice_status_html("success", "已设为参考音色", f"当前参考音色：{output_path}")

def find_downloaded_media_file(download_dir):
    candidates = []
    for root, _, files in os.walk(download_dir):
        for filename in files:
            path = os.path.join(root, filename)
            if os.path.isfile(path):
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: os.path.getsize(path))

def summarize_yt_dlp_error(exc):
    message = str(exc).strip()
    cause = getattr(exc, "__cause__", None)
    if cause:
        message = f"{message} {cause}".strip()
    message = re.sub(r"\s+", " ", message)
    if not message:
        return "未返回具体错误。"
    if len(message) > 260:
        message = message[:260].rstrip() + "..."
    lowered = message.lower()
    if any(keyword in lowered for keyword in ["login", "cookie", "cookies", "sign in", "登录"]):
        return f"{message}。可能需要登录 Cookie。"
    if any(keyword in lowered for keyword in ["forbidden", "403", "captcha", "verify", "验证", "风控"]):
        return f"{message}。可能被平台风控或需要验证码。"
    if any(keyword in lowered for keyword in ["unsupported url", "no suitable extractor"]):
        return f"{message}。当前 yt-dlp 可能不支持该链接格式。"
    return message

def parse_clip_time(value, default=None):
    if value is None:
        return default
    value = str(value).strip()
    if not value:
        return default
    try:
        if ":" not in value:
            seconds = float(value)
        else:
            parts = [float(part) for part in value.split(":")]
            if len(parts) == 2:
                seconds = parts[0] * 60 + parts[1]
            elif len(parts) == 3:
                seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
            else:
                raise ValueError
    except (TypeError, ValueError):
        raise ValueError("invalid clip time")
    return max(0.0, seconds)

def format_clip_time(seconds):
    seconds = int(round(max(0, seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"

def normalize_network_clip_range(start_time, end_time):
    start_seconds = parse_clip_time(start_time, default=0.0)
    end_seconds = parse_clip_time(end_time, default=start_seconds + 60.0)
    if end_seconds <= start_seconds:
        raise ValueError("end time must be greater than start time")
    requested_end = end_seconds
    if end_seconds - start_seconds > 60:
        end_seconds = start_seconds + 60.0
    return start_seconds, end_seconds, requested_end

def process_network_voice_media(url, start_time, end_time, progress=gr.Progress()):
    url = (url or "").strip()
    if not url:
        gr.Warning(i18n("请先输入网络素材链接"))
        return gr.update(), voice_status_html("warn", "请先输入网络素材链接")

    try:
        start_seconds, end_seconds, requested_end = normalize_network_clip_range(start_time, end_time)
    except ValueError:
        gr.Warning(i18n("请输入正确的起止时间"))
        return gr.update(), voice_status_html("warn", "时间格式不正确", "支持秒数、MM:SS、HH:MM:SS，且结束时间必须大于起始时间。")

    try:
        progress(0.08, desc="正在加载 yt-dlp...")
        import yt_dlp
        from yt_dlp.postprocessor.ffmpeg import FFmpegPostProcessor
        from yt_dlp.utils import download_range_func
    except ImportError:
        gr.Error(i18n("未安装 yt-dlp，请重新同步项目依赖"))
        return gr.update(), voice_status_html("error", "未安装 yt-dlp", "请重新同步项目依赖。")

    network_config = load_network_source_config()
    use_proxy = network_config["use_proxy"]
    proxy = network_config["proxy"]
    cookies_file = network_config["cookies_file"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        ffmpeg_path = resolve_ffmpeg_path()
        if not ffmpeg_path:
            gr.Error(i18n("未找到 ffmpeg，请先安装 ffmpeg"))
            return gr.update(), voice_status_html("error", "未找到 ffmpeg", "请先安装 ffmpeg 后再处理网络素材。")
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        if ffmpeg_dir and ffmpeg_dir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmp_dir, "%(title).80s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": False,
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
            "download_ranges": download_range_func(None, [(start_seconds, end_seconds)]),
            "force_keyframes_at_cuts": True,
            "ffmpeg_location": ffmpeg_path,
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        }
        if use_proxy and proxy and proxy.strip():
            ydl_opts["proxy"] = proxy.strip()
        if cookies_file and cookies_file.strip():
            cookie_path = os.path.expanduser(cookies_file.strip())
            if not os.path.exists(cookie_path):
                return gr.update(), voice_status_html("error", "Cookie 文件不存在", f"找不到：{cookie_path}")
            ydl_opts["cookiefile"] = cookie_path
        try:
            progress(0.2, desc="正在解析素材链接...")
            ffmpeg_location_token = FFmpegPostProcessor._ffmpeg_location.set(ffmpeg_path)
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.extract_info(url, download=True)
            finally:
                FFmpegPostProcessor._ffmpeg_location.reset(ffmpeg_location_token)
            progress(0.72, desc="正在定位下载片段...")
            media_path = find_downloaded_media_file(tmp_dir)
            if not media_path:
                raise RuntimeError("no downloaded media file")
            progress(0.82, desc="正在抽取音频...")
            output_path = convert_media_to_prompt_audio(media_path, "network_ref")
            progress(0.95, desc="正在设置参考音色...")
        except FileNotFoundError:
            gr.Error(i18n("未找到 ffmpeg，请先安装 ffmpeg"))
            return gr.update(), voice_status_html("error", "未找到 ffmpeg", "请先安装 ffmpeg 后再处理网络素材。")
        except Exception as exc:
            error_detail = summarize_yt_dlp_error(exc)
            print(f"Failed to process network media: {error_detail}")
            gr.Error(i18n("网络素材处理失败，请查看页面错误详情"))
            return gr.update(), voice_status_html("error", "网络素材处理失败", error_detail)

    gr.Info(i18n("已处理网络素材，并设为参考音色"), duration=2)
    clip_note = f"已截取 {format_clip_time(start_seconds)} - {format_clip_time(end_seconds)}。"
    if requested_end != end_seconds:
        clip_note += " 你设置的跨度超过 60 秒，已自动限制为 60 秒。"
    progress(1.0, desc="完成")
    return gr.update(value=output_path), voice_status_html("success", "已设为参考音色", f"{clip_note} 当前参考音色：{output_path}")

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Righteous&display=swap');

:root {
    --tts-bg: #090914;
    --tts-surface: rgba(18, 18, 34, 0.88);
    --tts-surface-soft: rgba(255, 255, 255, 0.055);
    --tts-ink: #f8fafc;
    --tts-muted: #d7def7;
    --tts-faint: #9eabd1;
    --tts-line: rgba(129, 140, 248, 0.22);
    --tts-line-strong: rgba(45, 212, 191, 0.42);
    --tts-primary: #2dd4bf;
    --tts-primary-dark: #14b8a6;
    --tts-primary-soft: rgba(45, 212, 191, 0.14);
    --tts-accent: #ff5c35;
    --tts-accent-dark: #f43f5e;
    --tts-violet: #8b5cf6;
    --tts-lime: #a3ff12;
    --tts-shadow: 0 24px 70px rgba(0, 0, 0, 0.34);
    --tts-radius: 14px;
}

.gradio-container {
    background:
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(0deg, rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(145deg, #090914 0%, #12112a 42%, #061a1c 100%) !important;
    background-size: 48px 48px, 48px 48px, auto !important;
    color: var(--tts-ink) !important;
    font-family: Poppins, -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif !important;
    line-height: 1.5 !important;
}

.main {
    max-width: 1500px !important;
    margin: 0 auto !important;
    padding: 18px 22px 34px !important;
}

.tts-topbar {
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: minmax(280px, 1fr) auto;
    gap: 18px;
    align-items: center;
    margin: 0 0 16px;
    padding: 22px 24px;
    border: 1px solid rgba(45, 212, 191, 0.28);
    border-radius: 18px;
    background:
        linear-gradient(120deg, rgba(45, 212, 191, 0.13), transparent 34%),
        linear-gradient(280deg, rgba(139, 92, 246, 0.22), transparent 46%),
        rgba(14, 14, 28, 0.9);
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32), inset 0 1px 0 rgba(255,255,255,0.08);
}

.tts-topbar::after {
    content: "";
    position: absolute;
    inset: auto 24px 0 24px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--tts-primary), var(--tts-violet), transparent);
    animation: scanline 3.8s ease-in-out infinite;
}

.tts-kicker {
    color: var(--tts-lime);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.tts-title {
    color: var(--tts-ink);
    font-family: Righteous, Poppins, sans-serif !important;
    font-size: 34px;
    line-height: 1.18;
    font-weight: 400;
    letter-spacing: 0;
    margin: 0;
    text-shadow: 0 0 24px rgba(45, 212, 191, 0.22);
}

.tts-subtitle {
    color: var(--tts-muted);
    font-size: 14px;
    line-height: 1.65;
    max-width: 820px;
    margin: 8px 0 0;
}

.tts-status {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px;
}

.tts-chip {
    min-height: 32px;
    display: inline-flex;
    align-items: center;
    border: 1px solid rgba(45, 212, 191, 0.34);
    border-radius: 999px;
    color: #e8fffb;
    background: rgba(45, 212, 191, 0.12);
    padding: 6px 11px;
    font-size: 12px;
    font-weight: 700;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
}

.sonic-meter {
    display: flex;
    align-items: end;
    gap: 4px;
    height: 42px;
    margin-top: 14px;
}

.sonic-meter i {
    width: 5px;
    height: 12px;
    border-radius: 999px;
    background: linear-gradient(180deg, var(--tts-lime), var(--tts-primary), var(--tts-violet));
    box-shadow: 0 0 18px rgba(45, 212, 191, 0.42);
    animation: equalize 1.25s ease-in-out infinite;
}

.sonic-meter i:nth-child(2n) { animation-delay: -0.28s; height: 24px; }
.sonic-meter i:nth-child(3n) { animation-delay: -0.58s; height: 34px; }
.sonic-meter i:nth-child(4n) { animation-delay: -0.78s; height: 18px; }
.sonic-meter i:nth-child(5n) { animation-delay: -1.05s; height: 39px; }

.studio-shell {
    gap: 14px !important;
    align-items: stretch !important;
}

.studio-panel {
    min-width: 0 !important;
    padding: 14px !important;
    border: 1px solid var(--tts-line) !important;
    border-radius: 18px !important;
    background:
        linear-gradient(180deg, rgba(255,255,255,0.065), rgba(255,255,255,0.025)),
        var(--tts-surface) !important;
    box-shadow: 0 18px 54px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255,255,255,0.08) !important;
    transition: border-color 220ms ease, transform 220ms ease, box-shadow 220ms ease !important;
}

.studio-panel:hover {
    border-color: rgba(45, 212, 191, 0.48) !important;
    transform: translateY(-2px);
    box-shadow: 0 26px 80px rgba(0, 0, 0, 0.34), 0 0 36px rgba(45, 212, 191, 0.08) !important;
}

.studio-panel .block {
    box-shadow: none !important;
}

.voice-panel {
    min-width: 330px !important;
}

.script-panel {
    min-width: 360px !important;
}

.result-panel {
    min-width: 250px !important;
}

.panel-heading {
    display: flex;
    align-items: center;
    gap: 9px;
    min-height: 28px;
    margin: 0 0 10px;
    color: var(--tts-ink);
}

.panel-heading span {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(163,255,18,0.16), rgba(45,212,191,0.24));
    color: var(--tts-lime);
    font-size: 12px;
    font-weight: 800;
}

.panel-heading strong {
    font-size: 15px;
    font-weight: 850;
    color: #ffffff;
}

.script-panel textarea {
    min-height: 360px !important;
}

.result-panel audio {
    min-height: 86px;
}

.action-row {
    gap: 10px !important;
}

.action-row button {
    width: 100% !important;
}

.compact-accordion {
    margin-top: 10px !important;
    box-shadow: none !important;
}

.lower-workspace {
    margin-top: 14px !important;
}

.lower-workspace > .tab-nav {
    background: rgba(255,255,255,0.045) !important;
    border: 1px solid var(--tts-line) !important;
    border-radius: 16px !important;
    padding: 4px !important;
    margin-bottom: 12px !important;
}

.lower-workspace > .tab-nav button {
    border-radius: 9px !important;
    min-height: 40px !important;
}

.settings-row, .toggle-row {
    gap: 12px !important;
}

.block, .form, .panel, .tabs, .tabitem, .accordion, .dataset, .wrap {
    border-radius: var(--tts-radius) !important;
}

.block {
    border-color: var(--tts-line) !important;
    background: var(--tts-surface) !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22) !important;
}

.form {
    border-color: var(--tts-line) !important;
    background: transparent !important;
}

.tabs {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

.tabs > .tab-nav {
    border-bottom: 1px solid var(--tts-line-strong) !important;
    margin-bottom: 16px !important;
}

.tabs > .tab-nav button {
    min-height: 42px !important;
    color: var(--tts-muted) !important;
    font-weight: 700 !important;
}

.tabs > .tab-nav button.selected {
    color: var(--tts-primary) !important;
    border-bottom: 3px solid var(--tts-primary) !important;
}

label, .label-wrap span {
    color: var(--tts-ink) !important;
    font-weight: 750 !important;
    font-size: 13px !important;
}

.info {
    color: var(--tts-muted) !important;
    font-size: 12px !important;
}

textarea, input {
    color: var(--tts-ink) !important;
    background: var(--tts-surface-soft) !important;
    border-color: var(--tts-line) !important;
    font-size: 15px !important;
}

textarea::placeholder, input::placeholder {
    color: rgba(203, 213, 225, 0.62) !important;
}

.wrap, .wrap .wrap-inner, .input-container, .secondary-wrap {
    background: transparent !important;
}

[data-testid="block-label"], .block-label {
    background: linear-gradient(135deg, rgba(12,19,30,0.96), rgba(16,35,39,0.94)) !important;
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    border: 1px solid rgba(45,212,191,0.44) !important;
    border-radius: 8px !important;
    box-shadow: 0 0 0 1px rgba(0,0,0,0.18), 0 8px 20px rgba(0,0,0,0.24) !important;
    text-shadow: 0 1px 8px rgba(0,0,0,0.55) !important;
}

.gradio-container label.container > span,
.gradio-container div.container > span,
.gradio-container fieldset > span,
.gradio-container .head label > span,
.gradio-container .head > label > span,
.gradio-container .wrap .head span.has-info,
.gradio-container label > span.has-info,
.gradio-container .block > label > span,
.gradio-container [id^="component-"] > label > span,
.gradio-container [id^="component-"] > div > span {
    display: inline-flex !important;
    align-items: center !important;
    width: fit-content !important;
    max-width: 100% !important;
    min-height: 26px !important;
    margin: 0 0 6px !important;
    padding: 4px 8px !important;
    border: 1px solid rgba(45, 212, 191, 0.42) !important;
    border-radius: 7px !important;
    background: linear-gradient(135deg, rgba(9, 9, 20, 0.96), rgba(18, 26, 43, 0.96)) !important;
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    font-size: 12px !important;
    font-weight: 800 !important;
    line-height: 1.25 !important;
    text-shadow: 0 1px 8px rgba(0, 0, 0, 0.62) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 8px 18px rgba(0,0,0,0.22) !important;
}

.gradio-container label.container > span + .info,
.gradio-container div.container > span + .info,
.gradio-container .info {
    display: block !important;
    width: fit-content !important;
    max-width: 100% !important;
    margin: 0 0 8px !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    color: #d7def7 !important;
    -webkit-text-fill-color: #d7def7 !important;
    text-shadow: none !important;
}

.dropdown, .dropdown input, .dropdown button {
    background: rgba(255,255,255,0.055) !important;
    color: var(--tts-ink) !important;
    border-color: var(--tts-line) !important;
}

select, option {
    background: #141428 !important;
    color: var(--tts-ink) !important;
}

input[type="range"] {
    accent-color: var(--tts-primary) !important;
}

input[type="checkbox"], input[type="radio"] {
    accent-color: var(--tts-primary) !important;
}

textarea {
    min-height: 42px !important;
    line-height: 1.65 !important;
}

.script-panel textarea {
    min-height: 360px !important;
}

textarea:focus, input:focus {
    border-color: var(--tts-primary) !important;
    box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.16), 0 0 26px rgba(45, 212, 191, 0.12) !important;
}

#gen_button, #smart_gen_button {
    background: var(--tts-accent) !important;
    border: 1px solid var(--tts-accent) !important;
    color: white !important;
    min-height: 52px !important;
    font-size: 16px;
    font-weight: 800;
    border-radius: 14px !important;
    box-shadow: 0 14px 34px rgba(255, 92, 53, 0.24), inset 0 1px 0 rgba(255,255,255,0.22) !important;
    transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease, filter 180ms ease !important;
}

#smart_gen_button {
    background: linear-gradient(135deg, #2dd4bf, #8b5cf6) !important;
    border-color: rgba(45, 212, 191, 0.72) !important;
    box-shadow: 0 14px 34px rgba(45, 212, 191, 0.2), 0 0 30px rgba(139, 92, 246, 0.16) !important;
}

#gen_button:hover, #smart_gen_button:hover {
    transform: translateY(-2px);
    filter: saturate(1.12);
}

#gen_button:hover {
    background: var(--tts-accent-dark) !important;
    border-color: var(--tts-accent-dark) !important;
    box-shadow: 0 18px 42px rgba(244, 63, 94, 0.32) !important;
}

#smart_gen_button:hover {
    background: linear-gradient(135deg, #a3ff12, #2dd4bf 42%, #8b5cf6) !important;
    box-shadow: 0 18px 46px rgba(45, 212, 191, 0.28), 0 0 38px rgba(139, 92, 246, 0.22) !important;
}

.voice-source-tabs {
    margin-top: 2px !important;
}

.voice-source-tabs > .tab-nav {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    gap: 5px !important;
    padding: 4px !important;
    border: 1px solid rgba(45, 212, 191, 0.18) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.04) !important;
}

.voice-source-tabs > .tab-nav button {
    width: 100% !important;
    min-height: 38px !important;
    border-radius: 8px !important;
}

.voice-source-tabs .overflow-menu,
.voice-source-tabs .overflow-menu button,
.voice-source-tabs .overflow-dropdown,
.voice-source-tabs .overflow-dropdown button {
    background: rgba(9, 9, 20, 0.98) !important;
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    border-color: rgba(45, 212, 191, 0.38) !important;
}

.voice-source-tabs .overflow-dropdown {
    padding: 8px !important;
    border: 1px solid rgba(45, 212, 191, 0.38) !important;
    border-radius: 10px !important;
    box-shadow: 0 18px 40px rgba(0,0,0,0.48), 0 0 24px rgba(45,212,191,0.14) !important;
}

.voice-source-tabs .overflow-dropdown button:hover {
    background: rgba(45, 212, 191, 0.18) !important;
}

.voice-status-card {
    display: grid;
    grid-template-columns: 14px 1fr;
    gap: 10px;
    align-items: start;
    margin: 10px 0 4px;
    padding: 11px 12px;
    border: 1px solid rgba(45, 212, 191, 0.22);
    border-radius: 10px;
    background: rgba(255,255,255,0.055);
    color: #f8fafc;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
}

.voice-status-card strong {
    display: block;
    color: #ffffff;
    font-size: 13px;
    line-height: 1.35;
}

.voice-status-card p {
    margin: 3px 0 0;
    color: #d7def7;
    font-size: 12px;
    line-height: 1.55;
}

.voice-status-dot {
    width: 10px;
    height: 10px;
    margin-top: 4px;
    border-radius: 999px;
    background: var(--tts-primary);
    box-shadow: 0 0 16px rgba(45, 212, 191, 0.58);
}

.voice-status-card.is-idle .voice-status-dot { background: var(--tts-faint); box-shadow: none; }
.voice-status-card.is-ready .voice-status-dot { background: var(--tts-lime); box-shadow: 0 0 16px rgba(163, 255, 18, 0.5); }
.voice-status-card.is-success .voice-status-dot { background: var(--tts-primary); }
.voice-status-card.is-warn .voice-status-dot { background: #fbbf24; box-shadow: 0 0 16px rgba(251, 191, 36, 0.5); }
.voice-status-card.is-error .voice-status-dot { background: #fb7185; box-shadow: 0 0 16px rgba(251, 113, 133, 0.55); }

.voice-status-card.is-busy {
    border-color: rgba(45, 212, 191, 0.48);
    background:
        linear-gradient(90deg, rgba(45,212,191,0.12), rgba(139,92,246,0.10), rgba(45,212,191,0.12)),
        rgba(255,255,255,0.055);
    background-size: 220% 100%;
    animation: statusFlow 1.4s linear infinite;
}

.voice-status-card.is-busy .voice-status-dot {
    animation: statusPulse 0.9s ease-in-out infinite;
}

#process_local_media_button,
#process_network_media_button,
#cancel_local_media_button {
    background: var(--tts-primary) !important;
    border: 1px solid var(--tts-primary) !important;
    color: white !important;
    min-height: 42px !important;
    font-weight: 800 !important;
    box-shadow: 0 10px 20px rgba(0, 127, 115, 0.18) !important;
}

#process_local_media_button:hover,
#process_network_media_button:hover,
#cancel_local_media_button:hover {
    background: var(--tts-primary-dark) !important;
    border-color: var(--tts-primary-dark) !important;
}

#process_local_media_button:disabled,
#process_network_media_button:disabled {
    opacity: 0.68 !important;
    cursor: wait !important;
    filter: saturate(0.8) !important;
}

#cancel_local_media_button {
    background: rgba(251, 113, 133, 0.12) !important;
    border-color: rgba(251, 113, 133, 0.48) !important;
    color: #ffe4e8 !important;
    min-height: 38px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 10px 20px rgba(0,0,0,0.14) !important;
}

#cancel_local_media_button:hover {
    background: rgba(251, 113, 133, 0.2) !important;
    border-color: rgba(251, 113, 133, 0.72) !important;
}

button {
    border-radius: 9px !important;
    cursor: pointer !important;
}

button.primary, button.secondary, .button {
    font-weight: 700 !important;
}

button:not(#gen_button):not(#smart_gen_button):not(#process_local_media_button):not(#process_network_media_button):not(#cancel_local_media_button) {
    background: rgba(255,255,255,0.055) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(45, 212, 191, 0.22) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 10px 24px rgba(0,0,0,0.14) !important;
    transition: border-color 180ms ease, background 180ms ease, transform 180ms ease !important;
}

button:not(#gen_button):not(#smart_gen_button):not(#process_local_media_button):not(#process_network_media_button):not(#cancel_local_media_button):hover {
    background: rgba(45, 212, 191, 0.12) !important;
    border-color: rgba(45, 212, 191, 0.46) !important;
    transform: translateY(-1px);
}

label:has(input[type="radio"]), label:has(input[type="checkbox"]) {
    background: rgba(255,255,255,0.045) !important;
    color: #f8fafc !important;
    border-color: rgba(45, 212, 191, 0.18) !important;
}

label:has(input[type="radio"]:checked), label:has(input[type="checkbox"]:checked) {
    background: rgba(45, 212, 191, 0.2) !important;
    border-color: rgba(45, 212, 191, 0.52) !important;
}

.accordion {
    border-color: var(--tts-line) !important;
    background: rgba(255,255,255,0.035) !important;
    box-shadow: none !important;
}

.accordion > .label-wrap {
    min-height: 46px !important;
    color: var(--tts-ink) !important;
    font-weight: 800 !important;
}

table, .wrap table, .table-wrap table {
    background: var(--tts-surface) !important;
    color: var(--tts-ink) !important;
    border-color: var(--tts-line) !important;
}

thead, tbody, tr {
    background: transparent !important;
}

th, .wrap table th, .table-wrap table th {
    background: rgba(45, 212, 191, 0.12) !important;
    color: #f8fafc !important;
    font-weight: 800 !important;
    border-color: rgba(45, 212, 191, 0.18) !important;
}

td, .wrap table td, .table-wrap table td {
    background: rgba(255,255,255,0.04) !important;
    color: #e5e7eb !important;
    border-color: rgba(45, 212, 191, 0.12) !important;
}

tr:nth-child(even) td, .wrap table tr:nth-child(even) td {
    background: rgba(255,255,255,0.065) !important;
}

@keyframes equalize {
    0%, 100% { transform: scaleY(0.45); opacity: 0.68; }
    45% { transform: scaleY(1.18); opacity: 1; }
    70% { transform: scaleY(0.72); opacity: 0.86; }
}

@keyframes scanline {
    0%, 100% { opacity: 0.38; transform: scaleX(0.72); }
    50% { opacity: 1; transform: scaleX(1); }
}

@keyframes statusFlow {
    from { background-position: 0% 50%; }
    to { background-position: 220% 50%; }
}

@keyframes statusPulse {
    0%, 100% { transform: scale(0.82); opacity: 0.62; }
    50% { transform: scale(1.2); opacity: 1; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

audio {
    width: 100% !important;
}

@media (max-width: 900px) {
    .main {
        padding: 12px !important;
    }

    .tts-topbar {
        grid-template-columns: 1fr;
    }

    .tts-status {
        justify-content: flex-start;
    }

    .tts-title {
        font-size: 24px;
    }

    .studio-shell {
        display: block !important;
    }

    .studio-shell > div {
        width: 100% !important;
        max-width: 100% !important;
        margin-bottom: 12px !important;
    }

    .studio-panel {
        padding: 12px !important;
    }

    .script-panel textarea {
        min-height: 280px !important;
    }

    .action-row {
        display: block !important;
    }

    .action-row button {
        width: 100% !important;
        margin-bottom: 10px !important;
    }
}

footer {
    display: none !important;
}

.gradio-container,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container p,
.gradio-container label,
.gradio-container span,
.gradio-container div,
.gradio-container button {
    color: var(--tts-ink) !important;
}

.tts-topbar .tts-kicker {
    color: var(--tts-lime) !important;
}

.tts-topbar .tts-title {
    color: #ffffff !important;
    opacity: 1 !important;
    text-shadow: 0 0 22px rgba(45, 212, 191, 0.32), 0 2px 18px rgba(0, 0, 0, 0.55) !important;
}

.tts-topbar .tts-subtitle {
    color: #dbeafe !important;
    opacity: 1 !important;
}

.tts-chip {
    color: #f0fffb !important;
}

.panel-heading,
.panel-heading strong,
.accordion > .label-wrap,
.accordion > .label-wrap span,
.tabs > .tab-nav button,
.lower-workspace > .tab-nav button,
label,
.label-wrap span,
.info,
.markdown,
.prose,
.wrap {
    color: #f8fafc !important;
    opacity: 1 !important;
}

.tabs > .tab-nav button:not(.selected),
.lower-workspace > .tab-nav button:not(.selected),
.info,
.tts-subtitle {
    color: #d7def7 !important;
}

.tabs > .tab-nav button.selected,
.lower-workspace > .tab-nav button.selected {
    color: var(--tts-lime) !important;
}

textarea,
input,
select,
.dropdown,
.dropdown *,
td,
th {
    color: #f8fafc !important;
}

textarea::placeholder,
input::placeholder {
    color: #b8c4e6 !important;
    opacity: 1 !important;
}

.toast-wrap,
.toast,
[data-testid="toast"],
[class*="toast"],
[class*="notification"],
[class*="Toast"] {
    color-scheme: dark !important;
}

.toast,
[data-testid="toast"],
[class*="toast"] > div,
[class*="notification"] > div,
[class*="Toast"] > div {
    background: linear-gradient(135deg, rgba(9, 9, 20, 0.98), rgba(18, 26, 43, 0.98)) !important;
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    border: 1px solid rgba(45, 212, 191, 0.42) !important;
    box-shadow: 0 18px 44px rgba(0, 0, 0, 0.48), 0 0 24px rgba(45, 212, 191, 0.14) !important;
}

.toast *,
[data-testid="toast"] *,
[class*="toast"] *,
[class*="notification"] *,
[class*="Toast"] * {
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
}
"""

APP_HEAD = """
<style id="indextts-contrast-lock">
html body .gradio-container,
html body .gradio-container * {
    color-scheme: dark !important;
}
html body .gradio-container .tts-title {
    color: #ffffff !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #ffffff !important;
}
html body .gradio-container .tts-subtitle,
html body .gradio-container .panel-heading,
html body .gradio-container .panel-heading strong,
html body .gradio-container .accordion > .label-wrap,
html body .gradio-container .accordion > .label-wrap span,
html body .gradio-container label,
html body .gradio-container .label-wrap span,
html body .gradio-container .info,
html body .gradio-container .tabs button,
html body .gradio-container .markdown,
html body .gradio-container .prose,
html body .gradio-container p,
html body .gradio-container h1,
html body .gradio-container h2,
html body .gradio-container h3,
html body .gradio-container h4 {
    color: #f8fafc !important;
    opacity: 1 !important;
    -webkit-text-fill-color: currentColor !important;
}
html body .gradio-container .tts-kicker,
html body .gradio-container .tabs button.selected,
html body .gradio-container .lower-workspace button.selected {
    color: #a3ff12 !important;
    -webkit-text-fill-color: #a3ff12 !important;
}
html body .gradio-container .tts-chip,
html body .gradio-container button,
html body .gradio-container th,
html body .gradio-container td,
html body .gradio-container input,
html body .gradio-container textarea,
html body .gradio-container select {
    color: #f8fafc !important;
    -webkit-text-fill-color: currentColor !important;
}
html body .gradio-container [data-testid="block-label"],
html body .gradio-container .block-label,
html body .gradio-container label.container > span,
html body .gradio-container div.container > span,
html body .gradio-container fieldset > span,
html body .gradio-container .head label > span,
html body .gradio-container .head > label > span,
html body .gradio-container .wrap .head span.has-info,
html body .gradio-container label > span.has-info,
html body .gradio-container .block > label > span,
html body .gradio-container [id^="component-"] > label > span,
html body .gradio-container [id^="component-"] > div > span {
    display: inline-flex !important;
    align-items: center !important;
    width: fit-content !important;
    max-width: 100% !important;
    min-height: 26px !important;
    margin: 0 0 6px !important;
    padding: 4px 8px !important;
    border: 1px solid rgba(45, 212, 191, 0.42) !important;
    border-radius: 7px !important;
    background: linear-gradient(135deg, rgba(9, 9, 20, 0.96), rgba(18, 26, 43, 0.96)) !important;
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    font-size: 12px !important;
    font-weight: 800 !important;
    line-height: 1.25 !important;
    text-shadow: 0 1px 8px rgba(0, 0, 0, 0.62) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 8px 18px rgba(0,0,0,0.22) !important;
}
html body .gradio-container label.container > span + .info,
html body .gradio-container div.container > span + .info,
html body .gradio-container .info {
    display: block !important;
    width: fit-content !important;
    max-width: 100% !important;
    margin: 0 0 8px !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    color: #d7def7 !important;
    -webkit-text-fill-color: #d7def7 !important;
    text-shadow: none !important;
}
</style>
<script>
(() => {
  const styleId = "indextts-runtime-theme-lock";
  const css = `
    html, body { background:#090914 !important; color-scheme:dark !important; }
    .gradio-container, .gradio-container * { color-scheme:dark !important; }
    .gradio-container .tts-title {
      color:#fff !important; opacity:1 !important; -webkit-text-fill-color:#fff !important;
      text-shadow:0 0 22px rgba(45,212,191,.32),0 2px 18px rgba(0,0,0,.55) !important;
    }
    .gradio-container .tts-subtitle,
    .gradio-container .panel-heading,
    .gradio-container .panel-heading strong,
    .gradio-container .accordion > .label-wrap,
    .gradio-container .accordion > .label-wrap span,
    .gradio-container label,
    .gradio-container .label-wrap span,
    .gradio-container .info,
    .gradio-container .tabs button,
    .gradio-container .markdown,
    .gradio-container .prose,
    .gradio-container p,
    .gradio-container h1,
    .gradio-container h2,
    .gradio-container h3,
    .gradio-container h4 {
      color:#f8fafc !important; opacity:1 !important; -webkit-text-fill-color:currentColor !important;
    }
    .gradio-container .tts-kicker,
    .gradio-container .tabs button.selected,
    .gradio-container .lower-workspace button.selected {
      color:#a3ff12 !important; -webkit-text-fill-color:#a3ff12 !important;
    }
    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select,
    .gradio-container th,
    .gradio-container td {
      color:#f8fafc !important; -webkit-text-fill-color:currentColor !important;
    }
    .gradio-container input::placeholder,
    .gradio-container textarea::placeholder {
      color:#b8c4e6 !important; opacity:1 !important; -webkit-text-fill-color:#b8c4e6 !important;
    }
    .gradio-container .voice-source-tabs .overflow-menu,
    .gradio-container .voice-source-tabs .overflow-menu button,
    .gradio-container .voice-source-tabs .overflow-dropdown,
    .gradio-container .voice-source-tabs .overflow-dropdown button {
      background:rgba(9,9,20,.98) !important;
      color:#f8fafc !important;
      -webkit-text-fill-color:#f8fafc !important;
      border-color:rgba(45,212,191,.38) !important;
    }
    .gradio-container .voice-source-tabs .overflow-dropdown {
      padding:8px !important;
      border:1px solid rgba(45,212,191,.38) !important;
      border-radius:10px !important;
      box-shadow:0 18px 40px rgba(0,0,0,.48), 0 0 24px rgba(45,212,191,.14) !important;
    }
    .gradio-container .voice-source-tabs .overflow-dropdown button:hover {
      background:rgba(45,212,191,.18) !important;
    }
    .gradio-container [data-testid="block-label"],
    .gradio-container .block-label,
    .gradio-container label.container > span,
    .gradio-container div.container > span,
    .gradio-container fieldset > span,
    .gradio-container .head label > span,
    .gradio-container .head > label > span,
    .gradio-container .wrap .head span.has-info,
    .gradio-container label > span.has-info,
    .gradio-container .block > label > span,
    .gradio-container [id^="component-"] > label > span,
    .gradio-container [id^="component-"] > div > span {
      display:inline-flex !important; align-items:center !important; width:fit-content !important; max-width:100% !important;
      min-height:26px !important; margin:0 0 6px !important; padding:4px 8px !important;
      border:1px solid rgba(45,212,191,.42) !important; border-radius:7px !important;
      background:linear-gradient(135deg, rgba(9,9,20,.96), rgba(18,26,43,.96)) !important;
      color:#f8fafc !important; -webkit-text-fill-color:#f8fafc !important;
      font-size:12px !important; font-weight:800 !important; line-height:1.25 !important;
      text-shadow:0 1px 8px rgba(0,0,0,.62) !important;
      box-shadow:inset 0 1px 0 rgba(255,255,255,.08), 0 8px 18px rgba(0,0,0,.22) !important;
    }
    .gradio-container label.container > span + .info,
    .gradio-container div.container > span + .info,
    .gradio-container .info {
      display:block !important; width:fit-content !important; max-width:100% !important;
      margin:0 0 8px !important; padding:0 !important; border:0 !important; background:transparent !important;
      color:#d7def7 !important; -webkit-text-fill-color:#d7def7 !important; text-shadow:none !important;
    }
    .toast-wrap,
    .toast,
    [data-testid="toast"],
    [class*="toast"],
    [class*="notification"],
    [class*="Toast"] {
      color-scheme:dark !important;
    }
    .toast,
    [data-testid="toast"],
    [class*="toast"] > div,
    [class*="notification"] > div,
    [class*="Toast"] > div {
      background:linear-gradient(135deg, rgba(9,9,20,.98), rgba(18,26,43,.98)) !important;
      color:#f8fafc !important;
      -webkit-text-fill-color:#f8fafc !important;
      border:1px solid rgba(45,212,191,.42) !important;
      box-shadow:0 18px 44px rgba(0,0,0,.48), 0 0 24px rgba(45,212,191,.14) !important;
    }
    .toast *,
    [data-testid="toast"] *,
    [class*="toast"] *,
    [class*="notification"] *,
    [class*="Toast"] * {
      color:#f8fafc !important;
      -webkit-text-fill-color:#f8fafc !important;
    }
  `;
  const maxUploadSizeMb = 1024;
  const maxUploadSizeBytes = maxUploadSizeMb * 1024 * 1024;
  const activeUploadControllers = new Set();
  const activeUploadXhrs = new Set();
  function showUploadLimitToast(file) {
    const existing = document.querySelector(".indextts-upload-limit-toast");
    if (existing) existing.remove();
    const toast = document.createElement("div");
    toast.className = "indextts-upload-limit-toast";
    toast.setAttribute("role", "alert");
    toast.innerHTML = `<strong>文件超过大小限制</strong><p>${file.name} 超过 ${maxUploadSizeMb} MB，请压缩或截取后再上传。</p>`;
    Object.assign(toast.style, {
      position: "fixed",
      top: "18px",
      right: "18px",
      zIndex: "999999",
      maxWidth: "360px",
      padding: "12px 14px",
      borderRadius: "12px",
      border: "1px solid rgba(45,212,191,.52)",
      background: "linear-gradient(135deg, rgba(9,9,20,.98), rgba(18,26,43,.98))",
      color: "#f8fafc",
      boxShadow: "0 18px 44px rgba(0,0,0,.5), 0 0 24px rgba(45,212,191,.16)",
      fontFamily: "Poppins, -apple-system, BlinkMacSystemFont, sans-serif",
      lineHeight: "1.45"
    });
    toast.querySelector("strong").style.cssText = "display:block;color:#fff;font-size:13px;margin-bottom:4px;";
    toast.querySelector("p").style.cssText = "margin:0;color:#d7def7;font-size:12px;";
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 5200);
  }
  function isLocalVoiceInput(input) {
    const accept = (input.getAttribute("accept") || "").toLowerCase();
    return accept.includes(".mp4") || accept.includes(".mov") || accept.includes(".mkv") || accept.includes(".webm");
  }
  function showUploadCancelToast() {
    const file = { name: "当前上传任务" };
    const existing = document.querySelector(".indextts-upload-limit-toast");
    if (existing) existing.remove();
    const toast = document.createElement("div");
    toast.className = "indextts-upload-limit-toast";
    toast.setAttribute("role", "alert");
    toast.innerHTML = `<strong>已取消上传/处理</strong><p>已尝试中断正在上传的文件，并清空本地素材选择。</p>`;
    Object.assign(toast.style, {
      position: "fixed",
      top: "18px",
      right: "18px",
      zIndex: "999999",
      maxWidth: "360px",
      padding: "12px 14px",
      borderRadius: "12px",
      border: "1px solid rgba(251,113,133,.52)",
      background: "linear-gradient(135deg, rgba(9,9,20,.98), rgba(43,18,28,.98))",
      color: "#f8fafc",
      boxShadow: "0 18px 44px rgba(0,0,0,.5), 0 0 24px rgba(251,113,133,.16)",
      fontFamily: "Poppins, -apple-system, BlinkMacSystemFont, sans-serif",
      lineHeight: "1.45"
    });
    toast.querySelector("strong").style.cssText = "display:block;color:#fff;font-size:13px;margin-bottom:4px;";
    toast.querySelector("p").style.cssText = "margin:0;color:#ffd9df;font-size:12px;";
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4200);
  }
  function installUploadGuard() {
    document.querySelectorAll('input[type="file"]').forEach((input) => {
      if (input.dataset.indexttsUploadGuard === "1" || !isLocalVoiceInput(input)) return;
      input.dataset.indexttsUploadGuard = "1";
      input.addEventListener("change", (event) => {
        const files = Array.from(input.files || []);
        const oversized = files.find((file) => file.size > maxUploadSizeBytes);
        if (!oversized) return;
        event.preventDefault();
        event.stopPropagation();
        input.value = "";
        showUploadLimitToast(oversized);
      }, true);
    });
  }
  function clearLocalVoiceInputs() {
    document.querySelectorAll('input[type="file"]').forEach((input) => {
      if (!isLocalVoiceInput(input)) return;
      input.value = "";
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }
  function abortActiveUploads() {
    activeUploadControllers.forEach((controller) => controller.abort());
    activeUploadControllers.clear();
    activeUploadXhrs.forEach((xhr) => {
      try { xhr.abort(); } catch (_) {}
    });
    activeUploadXhrs.clear();
    clearLocalVoiceInputs();
    showUploadCancelToast();
  }
  function installUploadAbortPatch() {
    if (!window.__indexttsUploadAbortPatched) {
      window.__indexttsUploadAbortPatched = true;
      const originalFetch = window.fetch.bind(window);
      window.fetch = (resource, init = {}) => {
        const url = typeof resource === "string" ? resource : resource?.url || "";
        if (String(url).includes("/gradio_api/upload")) {
          const controller = new AbortController();
          activeUploadControllers.add(controller);
          const nextInit = { ...init, signal: init.signal || controller.signal };
          return originalFetch(resource, nextInit).finally(() => activeUploadControllers.delete(controller));
        }
        return originalFetch(resource, init);
      };
      const originalOpen = XMLHttpRequest.prototype.open;
      const originalSend = XMLHttpRequest.prototype.send;
      XMLHttpRequest.prototype.open = function(method, url, ...rest) {
        this.__indexttsUploadUrl = String(url || "");
        return originalOpen.call(this, method, url, ...rest);
      };
      XMLHttpRequest.prototype.send = function(...args) {
        if (this.__indexttsUploadUrl && this.__indexttsUploadUrl.includes("/gradio_api/upload")) {
          activeUploadXhrs.add(this);
          this.addEventListener("loadend", () => activeUploadXhrs.delete(this), { once: true });
        }
        return originalSend.apply(this, args);
      };
    }
    document.querySelectorAll("#cancel_local_media_button").forEach((button) => {
      if (button.dataset.indexttsCancelGuard === "1") return;
      button.dataset.indexttsCancelGuard = "1";
      button.addEventListener("click", abortActiveUploads, true);
    });
  }
  function installLock() {
    let style = document.getElementById(styleId);
    if (!style) {
      style = document.createElement("style");
      style.id = styleId;
      document.head.appendChild(style);
    } else if (style.parentElement !== document.head || style !== document.head.lastElementChild) {
      document.head.appendChild(style);
    }
    if (style.textContent !== css) style.textContent = css;
    installUploadGuard();
    installUploadAbortPatch();
  }
  installLock();
  window.addEventListener("load", installLock);
  setTimeout(installLock, 300);
  setTimeout(installLock, 1200);
  setTimeout(installLock, 3000);
  new MutationObserver(installLock).observe(document.head, {childList:true});
})();
</script>
"""

with gr.Blocks(
    title="智能配音工作台",
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="orange",
        neutral_hue="slate",
    ),
    css=APP_CSS,
    head=APP_HEAD,
) as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <section class="tts-topbar">
      <div>
        <div class="tts-kicker">本地语音生成</div>
        <h1 class="tts-title">智能配音工作台</h1>
        <p class="tts-subtitle">音色素材、文本生成、智能情绪与专业调音集中在一个本机工作台。</p>
        <div class="sonic-meter" aria-hidden="true">
          <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
          <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
        </div>
      </div>
      <div class="tts-status">
        <span class="tts-chip">工作台</span>
        <span class="tts-chip">智能导演</span>
        <span class="tts-chip">专业调音</span>
        <span class="tts-chip">本机运行</span>
      </div>
    </section>
    ''')

    with gr.Tab(i18n("工作台")):
        with gr.Row(elem_classes=["studio-shell"]):
            os.makedirs("prompts",exist_ok=True)
            network_source_config = load_network_source_config()
            with gr.Column(scale=4, elem_classes=["studio-panel", "voice-panel"]):
                gr.HTML('<div class="panel-heading"><span>01</span><strong>音色来源</strong></div>')
                with gr.Tabs(elem_classes=["voice-source-tabs"]):
                    with gr.Tab(i18n("本地上传")):
                        local_voice_media = gr.File(
                            label=i18n("上传本地音视频"),
                            file_types=[
                                ".wav", ".mp3", ".flac", ".m4a", ".ogg",
                                ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm",
                            ],
                            type="filepath",
                        )
                        process_local_media_button = gr.Button(
                            i18n("处理并用作音色"),
                            elem_id="process_local_media_button",
                        )
                        cancel_local_media_button = gr.Button(
                            i18n("取消上传/处理"),
                            elem_id="cancel_local_media_button",
                        )
                        local_voice_status = gr.Markdown(voice_status_html("idle", "等待上传", f"请选择一段本地音频或视频。单个文件最大 {MAX_UPLOAD_SIZE_MB} MB。"))
                    with gr.Tab(i18n("网络素材")):
                        network_voice_url = gr.Textbox(
                            label=i18n("素材链接"),
                            placeholder="https://www.bilibili.com/video/... / 抖音 / 小红书",
                        )
                        with gr.Row():
                            network_clip_start = gr.Textbox(
                                label=i18n("截取起点"),
                                placeholder="默认 00:00",
                            )
                            network_clip_end = gr.Textbox(
                                label=i18n("截取终点"),
                                placeholder="默认起点后 60 秒",
                            )
                        with gr.Accordion(i18n("下载设置"), open=False):
                            with gr.Row():
                                network_use_proxy = gr.Checkbox(
                                    label=i18n("启用下载代理"),
                                    value=network_source_config["use_proxy"],
                                )
                                network_proxy = gr.Textbox(
                                    label=i18n("代理地址"),
                                    value=network_source_config["proxy"],
                                    placeholder="http://127.0.0.1:7897",
                                    interactive=network_source_config["use_proxy"],
                                )
                            network_cookies_file = gr.Textbox(
                                label=i18n("Cookie 文件路径"),
                                value=network_source_config["cookies_file"],
                                placeholder="可选，例如 ~/Downloads/cookies.txt",
                            )
                            network_config_status = gr.Markdown("网络素材配置会在解析时自动保存。")
                        process_network_media_button = gr.Button(
                            i18n("解析并用作音色"),
                            elem_id="process_network_media_button",
                        )
                        network_voice_status = gr.Markdown(voice_status_html("idle", "等待链接", "粘贴素材链接后，系统会默认截取前 60 秒。"))
                    with gr.Tab(i18n("常用音色库")):
                        saved_voice_dropdown = gr.Dropdown(
                            label=i18n("选择已保存音色"),
                            choices=list_saved_voices(),
                            value=None,
                            allow_custom_value=False,
                        )
                        with gr.Row():
                            load_voice_button = gr.Button(i18n("使用该音色"))
                            refresh_voices_button = gr.Button(i18n("刷新列表"))
                        library_voice_status = gr.Markdown("")
                prompt_audio = gr.Audio(
                    label=i18n("当前参考音色"),
                    key="prompt_audio",
                    type="filepath",
                    interactive=False,
                )
                voice_name = gr.Textbox(
                    label=i18n("保存名称"),
                    placeholder=i18n("例如：我的旁白音色"),
                )
                save_voice_button = gr.Button(i18n("保存到常用音色库"))
                saved_voice_status = gr.Markdown("")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column(scale=7, elem_classes=["studio-panel", "script-panel"]):
                gr.HTML('<div class="panel-heading"><span>02</span><strong>朗读内容</strong></div>')
                input_text_single = gr.TextArea(label=i18n("朗读文本"),key="input_text_single", placeholder=i18n("在这里输入要生成配音的正文"), info=f"{i18n('当前模型版本：')}{tts.model_version or '1.0'}")
                with gr.Row(elem_classes=["action-row"]):
                    gen_button = gr.Button(i18n("直接生成"), key="gen_button", elem_id="gen_button", interactive=True)
                    smart_gen_button = gr.Button(i18n("智能分段生成"), elem_id="smart_gen_button")
            with gr.Column(scale=4, elem_classes=["studio-panel", "result-panel"]):
                gr.HTML('<div class="panel-heading"><span>03</span><strong>生成结果</strong></div>')
                output_audio = gr.Audio(label=i18n("试听生成结果"), visible=True,key="output_audio")

        smart_llm_config = load_smart_llm_config()
        with gr.Tabs(elem_classes=["lower-workspace"]):
            with gr.Tab(i18n("智能导演")):
                with gr.Row(elem_classes=["settings-row"]):
                    with gr.Column(scale=1):
                        smart_analysis_mode = gr.Radio(
                            choices=["本地规则", "OpenAI兼容大模型"],
                            value=smart_llm_config["analysis_mode"],
                            label=i18n("情绪分析方式"),
                        )
                        smart_pause_ms = gr.Slider(
                            label=i18n("段间停顿毫秒"),
                            minimum=0,
                            maximum=1200,
                            value=smart_llm_config["pause_ms"],
                            step=20,
                        )
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间"),
                        )
                    with gr.Column(scale=2):
                        with gr.Row():
                            smart_api_base = gr.Textbox(
                                label=i18n("API 地址"),
                                value=smart_llm_config["api_base"],
                                placeholder="https://api.openai.com/v1/chat/completions",
                            )
                            smart_model = gr.Textbox(
                                label=i18n("模型名称"),
                                value=smart_llm_config["model"],
                            )
                        with gr.Row():
                            smart_api_key = gr.Textbox(
                                label=i18n("API Key"),
                                value="",
                                type="password",
                                placeholder=i18n("已保存则可留空；输入新 Key 会更新本机配置"),
                            )
                            smart_use_proxy = gr.Checkbox(
                                label=i18n("启用代理"),
                                value=smart_llm_config["use_proxy"],
                            )
                            smart_proxy = gr.Textbox(
                                label=i18n("代理地址"),
                                value=smart_llm_config["proxy"],
                                placeholder="http://127.0.0.1:7897",
                                interactive=smart_llm_config["use_proxy"],
                            )
                        with gr.Row():
                            smart_save_config_button = gr.Button(i18n("保存大模型配置"))
                            smart_config_status = gr.Markdown("已载入大模型配置。API Key 已保存在本机，页面不会回填显示。")
                with gr.Row():
                    with gr.Column(scale=1):
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )
                    with gr.Column(scale=1):
                        smart_segments_table = gr.Dataframe(
                            headers=[
                                i18n("序号"),
                                i18n("分段文本"),
                                i18n("情绪"),
                                i18n("强度"),
                                i18n("情感向量"),
                                i18n("依据"),
                            ],
                            wrap=True,
                        )

            with gr.Tab(i18n("专业调音")):
                with gr.Row(elem_classes=["toggle-row"]):
                    experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)
                    glossary_checkbox = gr.Checkbox(label=i18n("开启术语词汇读音"), value=tts.normalizer.enable_glossary)
                with gr.Accordion(i18n("情感控制"), open=False):
                    with gr.Row():
                        emo_control_method = gr.Radio(
                            choices=EMO_CHOICES_OFFICIAL,
                            type="index",
                            value=EMO_CHOICES_OFFICIAL[0],label=i18n("情感控制方式"))
                        # we MUST have an extra, INVISIBLE list of *all* emotion control
                        # methods so that gr.Dataset() can fetch ALL control mode labels!
                        # otherwise, the gr.Dataset()'s experimental labels would be empty!
                        emo_control_method_all = gr.Radio(
                            choices=EMO_CHOICES_ALL,
                            type="index",
                            value=EMO_CHOICES_ALL[0], label=i18n("情感控制方式"),
                            visible=False)  # do not render

                    with gr.Group(visible=False) as emotion_reference_group:
                        with gr.Row():
                            emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

                    with gr.Row(visible=False) as emotion_randomize_group:
                        emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

                    with gr.Group(visible=False) as emotion_vector_group:
                        with gr.Row():
                            with gr.Column():
                                vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                            with gr.Column():
                                vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                                vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)

                    with gr.Group(visible=False) as emo_text_group:
                        create_experimental_warning_message()
                        with gr.Row():
                            emo_text = gr.Textbox(label=i18n("情感描述文本"),
                                                  placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                                                  value="",
                                                  info=i18n("例如：委屈巴巴、危险在悄悄逼近"))

                    with gr.Row(visible=False) as emo_weight_group:
                        emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)

                with gr.Accordion(i18n("自定义术语读音"), open=False, visible=tts.normalizer.enable_glossary) as glossary_accordion:
                    with gr.Row():
                        with gr.Column(scale=1):
                            glossary_term = gr.Textbox(
                                label=i18n("术语"),
                                placeholder="IndexTTS2",
                            )
                            glossary_reading_zh = gr.Textbox(
                                label=i18n("中文读法"),
                                placeholder="Index T-T-S 二",
                            )
                            glossary_reading_en = gr.Textbox(
                                label=i18n("英文读法"),
                                placeholder="Index T-T-S two",
                            )
                            btn_add_term = gr.Button(i18n("添加术语"), scale=1)
                        with gr.Column(scale=2):
                            glossary_table = gr.Markdown(
                                value=format_glossary_markdown()
                            )

                with gr.Accordion(i18n("高级生成参数"), open=False, visible=True) as advanced_settings_group:
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                do_sample = gr.Checkbox(label=i18n("启用采样"), value=True, info=i18n("关闭后结果更稳定，但表现力可能下降"))
                                temperature = gr.Slider(label=i18n("随机温度"), minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                            with gr.Row():
                                top_p = gr.Slider(label=i18n("核心采样概率"), minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                                top_k = gr.Slider(label=i18n("候选数量"), minimum=0, maximum=100, value=30, step=1)
                                num_beams = gr.Slider(label=i18n("搜索束数量"), value=3, minimum=1, maximum=10, step=1)
                            with gr.Row():
                                repetition_penalty = gr.Number(label=i18n("重复惩罚"), precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                                length_penalty = gr.Number(label=i18n("长度惩罚"), precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                            max_mel_tokens = gr.Slider(label=i18n("音频最大生成长度"), value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("数值过小可能导致音频被截断"), key="max_mel_tokens")
                    advanced_params = [
                        do_sample, top_p, top_k, temperature,
                        length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                        # typical_sampling, typical_mass,
                    ]

            with gr.Tab(i18n("示例素材")):
                # we must use `gr.Dataset` to support dynamic UI rewrites, since `gr.Examples`
                # binds tightly to UI and always restores the initial state of all components,
                # such as the list of available choices in emo_control_method.
                example_table = gr.Dataset(label=i18n("示例素材"),
            samples_per_page=20,
            samples=get_example_cases(include_experimental=False),
            type="values",
            # these components are NOT "connected". it just reads the column labels/available
            # states from them, so we MUST link to the "all options" versions of all components,
            # such as `emo_control_method_all` (to be able to see EXPERIMENTAL text labels)!
            components=[prompt_audio,
                        emo_control_method_all,  # important: support all mode labels!
                        input_text_single,
                        emo_upload,
                        emo_weight,
                        emo_text,
                        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        )

    def on_example_click(example):
        print(f"Example clicked: ({len(example)} values) = {example!r}")
        return (
            gr.update(value=example[0]),
            gr.update(value=example[1]),
            gr.update(value=example[2]),
            gr.update(value=example[3]),
            gr.update(value=example[4]),
            gr.update(value=example[5]),
            gr.update(value=example[6]),
            gr.update(value=example[7]),
            gr.update(value=example[8]),
            gr.update(value=example[9]),
            gr.update(value=example[10]),
            gr.update(value=example[11]),
            gr.update(value=example[12]),
            gr.update(value=example[13]),
        )

    # click() event works on both desktop and mobile UI
    example_table.click(on_example_click,
                        inputs=[example_table],
                        outputs=[prompt_audio,
                                 emo_control_method,
                                 input_text_single,
                                 emo_upload,
                                 emo_weight,
                                 emo_text,
                                 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
    )

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = detokenize_segment(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {
                segments_preview: gr.update(value=df),
            }

    # 术语词汇表事件处理函数
    def on_add_glossary_term(term, reading_zh, reading_en):
        """添加术语到词汇表并自动保存"""
        term = term.rstrip()
        reading_zh = reading_zh.rstrip()
        reading_en = reading_en.rstrip()

        if not term:
            gr.Warning(i18n("请输入术语"))
            return gr.update()
            
        if not reading_zh and not reading_en:
            gr.Warning(i18n("请至少输入一种读法"))
            return gr.update()
        

        # 构建读法数据
        if reading_zh and reading_en:
            reading = {"zh": reading_zh, "en": reading_en}
        elif reading_zh:
            reading = {"zh": reading_zh}
        elif reading_en:
            reading = {"en": reading_en}
        else:
            reading = reading_zh or reading_en

        # 添加到词汇表
        tts.normalizer.term_glossary[term] = reading

        # 自动保存到文件
        try:
            tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
            gr.Info(i18n("词汇表已更新"), duration=1)
        except Exception as e:
            gr.Error(i18n("保存词汇表时出错"))
            print(f"Error details: {e}")
            return gr.update()

        # 更新Markdown表格
        return gr.update(value=format_glossary_markdown())
        

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )

    def on_experimental_change(is_experimental, current_mode_index):
        # 切换情感控制选项
        new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
        # if their current mode selection doesn't exist in new choices, reset to 0.
        # we don't verify that OLD index means the same in NEW list, since we KNOW it does.
        new_index = current_mode_index if current_mode_index < len(new_choices) else 0

        return (
            gr.update(choices=new_choices, value=new_choices[new_index]),
            gr.update(samples=get_example_cases(include_experimental=is_experimental)),
        )

    experimental_checkbox.change(
        on_experimental_change,
        inputs=[experimental_checkbox, emo_control_method],
        outputs=[emo_control_method, example_table]
    )

    def on_glossary_checkbox_change(is_enabled):
        """控制术语词汇表的可见性"""
        tts.normalizer.enable_glossary = is_enabled
        return gr.update(visible=is_enabled)

    glossary_checkbox.change(
        on_glossary_checkbox_change,
        inputs=[glossary_checkbox],
        outputs=[glossary_accordion]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    local_voice_media.change(
        local_voice_upload_feedback,
        inputs=[local_voice_media],
        outputs=[local_voice_media, local_voice_status],
    )

    local_voice_process_event = process_local_media_button.click(
        mark_local_processing,
        inputs=[local_voice_media],
        outputs=[process_local_media_button, local_voice_status],
        show_progress="hidden",
    ).then(
        process_local_voice_media,
        inputs=[local_voice_media],
        outputs=[prompt_audio, local_voice_status],
        show_progress="full",
    ).then(
        restore_processing_button,
        outputs=[process_local_media_button],
        show_progress="hidden",
    )

    cancel_local_media_button.click(
        cancel_local_voice_task,
        outputs=[local_voice_media, process_local_media_button, local_voice_status],
        cancels=[local_voice_process_event],
        show_progress="hidden",
    )

    process_network_media_button.click(
        mark_network_processing,
        inputs=[network_voice_url, network_clip_start, network_clip_end],
        outputs=[process_network_media_button, network_voice_status],
        show_progress="hidden",
    ).then(
        process_network_voice_media,
        inputs=[network_voice_url, network_clip_start, network_clip_end],
        outputs=[prompt_audio, network_voice_status],
        show_progress="full",
    ).then(
        restore_processing_button,
        outputs=[process_network_media_button],
        show_progress="hidden",
    )

    network_use_proxy.change(
        on_network_proxy_toggle,
        inputs=[network_use_proxy, network_proxy, network_cookies_file],
        outputs=[network_proxy, network_config_status],
        show_progress="hidden",
    )
    network_proxy.change(
        save_network_source_config,
        inputs=[network_use_proxy, network_proxy, network_cookies_file],
        outputs=[network_config_status],
        show_progress="hidden",
    )
    network_cookies_file.change(
        save_network_source_config,
        inputs=[network_use_proxy, network_proxy, network_cookies_file],
        outputs=[network_config_status],
        show_progress="hidden",
    )

    save_voice_button.click(
        save_current_voice,
        inputs=[voice_name, prompt_audio],
        outputs=[saved_voice_dropdown, saved_voice_status],
    )

    saved_voice_dropdown.change(
        load_saved_voice,
        inputs=[saved_voice_dropdown],
        outputs=[prompt_audio, library_voice_status],
    )

    load_voice_button.click(
        load_saved_voice,
        inputs=[saved_voice_dropdown],
        outputs=[prompt_audio, library_voice_status],
    )

    refresh_voices_button.click(
        refresh_saved_voices,
        inputs=[],
        outputs=[saved_voice_dropdown, library_voice_status],
    )

    def on_demo_load():
        """页面加载时重新加载glossary数据"""
        try:
            tts.normalizer.load_glossary_from_yaml(tts.glossary_path)
        except Exception as e:
            gr.Error(i18n("加载词汇表时出错"))
            print(f"Failed to reload glossary on page load: {e}")
        return gr.update(value=format_glossary_markdown())

    # 术语词汇表事件绑定
    btn_add_term.click(
        on_add_glossary_term,
        inputs=[glossary_term, glossary_reading_zh, glossary_reading_en],
        outputs=[glossary_table]
    )

    # 页面加载时重新加载glossary
    demo.load(
        on_demo_load,
        inputs=[],
        outputs=[glossary_table]
    )

    demo.load(
        refresh_saved_voices,
        inputs=[],
        outputs=[saved_voice_dropdown, library_voice_status]
    )

    demo.load(
        load_smart_llm_config_for_ui,
        inputs=[],
        outputs=[
            smart_analysis_mode,
            smart_api_base,
            smart_api_key,
            smart_model,
            smart_use_proxy,
            smart_proxy,
            smart_pause_ms,
            smart_config_status,
        ],
    )

    smart_analysis_mode.change(
        save_smart_llm_config_keep_key,
        inputs=[smart_analysis_mode, smart_api_base, smart_api_key, smart_model, smart_use_proxy, smart_proxy, smart_pause_ms],
        outputs=[smart_config_status],
    )
    smart_api_base.change(
        save_smart_llm_config_keep_key,
        inputs=[smart_analysis_mode, smart_api_base, smart_api_key, smart_model, smart_use_proxy, smart_proxy, smart_pause_ms],
        outputs=[smart_config_status],
    )
    smart_api_key.change(
        save_smart_llm_config,
        inputs=[smart_analysis_mode, smart_api_base, smart_api_key, smart_model, smart_use_proxy, smart_proxy, smart_pause_ms],
        outputs=[smart_config_status],
    )
    smart_model.change(
        save_smart_llm_config_keep_key,
        inputs=[smart_analysis_mode, smart_api_base, smart_api_key, smart_model, smart_use_proxy, smart_proxy, smart_pause_ms],
        outputs=[smart_config_status],
    )
    smart_pause_ms.change(
        save_smart_llm_config_keep_key,
        inputs=[smart_analysis_mode, smart_api_base, smart_api_key, smart_model, smart_use_proxy, smart_proxy, smart_pause_ms],
        outputs=[smart_config_status],
    )
    smart_proxy.change(
        save_smart_llm_config_keep_key,
        inputs=[smart_analysis_mode, smart_api_base, smart_api_key, smart_model, smart_use_proxy, smart_proxy, smart_pause_ms],
        outputs=[smart_config_status],
    )
    smart_use_proxy.change(
        on_smart_proxy_toggle,
        inputs=[smart_use_proxy, smart_analysis_mode, smart_api_base, smart_api_key, smart_model, smart_proxy, smart_pause_ms],
        outputs=[smart_proxy, smart_config_status],
    )
    smart_save_config_button.click(
        save_smart_llm_config_keep_key,
        inputs=[smart_analysis_mode, smart_api_base, smart_api_key, smart_model, smart_use_proxy, smart_proxy, smart_pause_ms],
        outputs=[smart_config_status],
    )

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text,emo_random,
                             max_text_tokens_per_segment,
                             *advanced_params,
                     ],
                     outputs=[output_audio])

    smart_gen_button.click(
        gen_smart_segments,
        inputs=[
            prompt_audio,
            input_text_single,
            smart_analysis_mode,
            smart_api_base,
            smart_api_key,
            smart_model,
            smart_use_proxy,
            smart_proxy,
            smart_pause_ms,
            max_text_tokens_per_segment,
            *advanced_params,
        ],
        outputs=[output_audio, smart_segments_table],
    )



if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, root_path=cmd_args.root_path)

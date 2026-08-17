import torch
import folder_paths
from comfy.utils import ProgressBar
import numpy as np
import os
import tempfile
import time
import subprocess
import math
import cv2
import re

# Импорты из основного файла
import sys
sys.path.append(os.path.dirname(__file__))
from video_segmentation_node import (
    VIDEO_EXTENSIONS, 
    _get_video_info, 
    _extract_segment_frames, 
    _extract_audio_segment,
    FFMPEG_BIN
)

# Импорты для генерации
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
from comfy import samplers
from comfy_extras.nodes_advanced_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_vae import VAEDecodeTiled
from comfy_extras.nodes_sampling import KSamplerSelect, BasicScheduler
from comfy_extras.nodes_random_noise import RandomNoise
from comfy.sd import VAE
from comfy.ldm.modules.diffusionmodules import model


class VideoSegmentationGenerateAndConcat:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.isdir(input_dir):
            for f in os.listdir(input_dir):
                full = os.path.join(input_dir, f)
                if not os.path.isfile(full):
                    continue
                ext = os.path.splitext(f)[1].lower().lstrip(".")
                if ext in VIDEO_EXTENSIONS:
                    files.append(f)

        return {
            "required": {
                "video": (sorted(files), {"video_upload": True}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 60.0, "step": 0.1}),
                "overlap_frames": ("INT", {"default": 2, "min": 0, "max": 30, "step": 1}),
                "align_to_frames": ("BOOLEAN", {"default": True}),
                "target_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1}),
                "width": ("INT", {"default": 1344, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "length": ("INT", {"default": 124, "min": 8, "max": 1024, "step": 8}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (["res_multistep", "euler", "dpmpp_2m"], {"default": "res_multistep"}),
                "scheduler": (["simple", "normal", "karras"], {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tile_size": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "tile_overlap": ("INT", {"default": 128, "min": 0, "max": 512, "step": 16}),
                "temporal_size": ("INT", {"default": 128, "min": 32, "max": 512, "step": 16}),
                "temporal_overlap": ("INT", {"default": 16, "min": 0, "max": 256, "step": 8}),
                "join_mode": (["cut", "dissolve"], {"default": "cut"}),
                "transition_duration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05}),
                "filename_prefix": ("STRING", {"default": "generated"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "ref_image": ("IMAGE",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_and_concat"
    CATEGORY = "Hatolic/video"

    def _concat_with_ffmpeg(self, video_paths, output_path, join_mode="cut", transition_duration=0.5):
        if len(video_paths) == 0:
            raise ValueError("No videos to concatenate")
        if len(video_paths) == 1:
            cmd = [FFMPEG_BIN, "-i", video_paths[0], "-c", "copy", "-y", output_path]
            subprocess.run(cmd, check=True, capture_output=True)
            return

        if join_mode == "cut":
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for path in video_paths:
                    f.write(f"file '{os.path.abspath(path)}'\n")
                list_file = f.name
            try:
                cmd = [
                    FFMPEG_BIN,
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file,
                    "-c", "copy",
                    "-y",
                    output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            finally:
                if os.path.exists(list_file):
                    os.unlink(list_file)
        elif join_mode == "dissolve":
            durations = []
            for path in video_paths:
                cmd = [FFMPEG_BIN, "-i", path, "-f", "null", "-"]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d+)", proc.stderr)
                if match:
                    h, m, s = match.groups()
                    duration = int(h) * 3600 + int(m) * 60 + float(s)
                    durations.append(duration)
                else:
                    durations.append(5.0)

            inputs = []
            for path in video_paths:
                inputs.extend(["-i", path])

            filter_parts = []
            filter_names = []
            current_offset = 0

            for i in range(1, len(video_paths)):
                offset = current_offset + durations[i-1] - transition_duration
                filter_name = f"f{i-1}"
                if i == 1:
                    filter_parts.append(
                        f"[0:v][1:v]xfade=transition=fade:duration={transition_duration}:offset={offset}[{filter_name}]"
                    )
                else:
                    filter_parts.append(
                        f"[{filter_names[-1]}][{i}:v]xfade=transition=fade:duration={transition_duration}:offset={offset}[{filter_name}]"
                    )
                current_offset = offset
                filter_names.append(filter_name)

            if filter_parts:
                filter_complex = ";".join(filter_parts)
                last_filter = filter_names[-1] if filter_names else "[0:v]"
                cmd = [
                    FFMPEG_BIN,
                    *inputs,
                    "-filter_complex", filter_complex,
                    "-map", f"[{last_filter}]",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-y",
                    output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            else:
                cmd = [FFMPEG_BIN, "-i", video_paths[0], "-c", "copy", "-y", output_path]
                subprocess.run(cmd, check=True, capture_output=True)

    def generate_and_concat(self, **kwargs):
        """Генерирует сегменты и объединяет их в одно видео."""
        
        # Извлекаем параметры
        video = kwargs.get("video")
        duration_seconds = kwargs.get("duration_seconds", 5.0)
        overlap_frames = kwargs.get("overlap_frames", 2)
        align_to_frames = kwargs.get("align_to_frames", True)
        target_fps = kwargs.get("target_fps", 0.0)

        width = kwargs.get("width", 1344)
        height = kwargs.get("height", 768)
        length = kwargs.get("length", 124)
        prompt = kwargs.get("prompt", "")
        steps = kwargs.get("steps", 6)
        cfg = kwargs.get("cfg", 1.0)
        sampler_name = kwargs.get("sampler_name", "res_multistep")
        scheduler = kwargs.get("scheduler", "simple")
        denoise = kwargs.get("denoise", 1.0)
        tile_size = kwargs.get("tile_size", 768)
        tile_overlap = kwargs.get("tile_overlap", 128)
        temporal_size = kwargs.get("temporal_size", 128)
        temporal_overlap = kwargs.get("temporal_overlap", 16)
        join_mode = kwargs.get("join_mode", "cut")
        transition_duration = kwargs.get("transition_duration", 0.5)
        filename_prefix = kwargs.get("filename_prefix", "generated")
        seed = kwargs.get("seed", 0)

        clip = kwargs.get("clip")
        vae = kwargs.get("vae")
        audio_vae = kwargs.get("audio_vae")
        ref_image = kwargs.get("ref_image")
        model = kwargs.get("model")
        positive = kwargs.get("positive")
        negative = kwargs.get("negative")

        # Проверка обязательных входов
        if clip is None:
            raise ValueError("Missing required input: clip")
        if vae is None:
            raise ValueError("Missing required input: vae")
        if audio_vae is None:
            raise ValueError("Missing required input: audio_vae")
        if ref_image is None:
            raise ValueError("Missing required input: ref_image")
        if model is None:
            raise ValueError("Missing required input: model")
        if positive is None:
            raise ValueError("Missing required input: positive")
        if negative is None:
            raise ValueError("Missing required input: negative")

        # 1. Сегментируем видео
        video_path = folder_paths.get_annotated_filepath(video)
        original_fps, total_frames, video_duration = _get_video_info(video_path)
        used_fps = target_fps if target_fps > 0 else original_fps
        total_segments = max(1, int(math.ceil(video_duration / duration_seconds)))

        # Временные файлы для сегментов
        temp_dir = tempfile.mkdtemp(prefix="segments_")
        segment_paths = []

        pbar = ProgressBar(total_segments)

        for seg_idx in range(total_segments):
            start_sec = seg_idx * duration_seconds
            end_sec = min(start_sec + duration_seconds, video_duration)
            is_last = (seg_idx == total_segments - 1)

            # Извлекаем кадры и аудио
            images, actual_start, actual_end = _extract_segment_frames(
                video_path, start_sec, end_sec,
                fps=used_fps,
                original_fps=original_fps,
                overlap_frames=overlap_frames,
                align_to_frames=align_to_frames,
                is_last=is_last
            )
            if images.shape[0] == 0:
                continue
            audio = _extract_audio_segment(video_path, actual_start, actual_end - actual_start)

            # Здесь должна быть логика генерации с использованием minimax
            # ВНИМАНИЕ: Этот код требует доработки - вызов MiniMaxH3ReferenceToVideo
            # из кода сложен, так как нода ожидает входы из графа.
            # Рекомендую использовать граф с циклом, как в вашем workflow.
            
            print(f"[VideoSegmentationGenerateAndConcat] Segment {seg_idx+1}/{total_segments}")
            print(f"  - frames: {images.shape[0]}, audio: {audio['waveform'].shape}")
            
            # Сохраняем сегмент во временный файл
            # Здесь нужно сохранить сгенерированное видео, а не оригинальные кадры
            # Пока сохраняем заглушку
            temp_path = os.path.join(temp_dir, f"segment_{seg_idx:04d}.mp4")
            segment_paths.append(temp_path)
            
            pbar.update(1)

        # Если нет сегментов - возвращаем None
        if len(segment_paths) == 0:
            print("[VideoSegmentationGenerateAndConcat] No segments generated")
            return (None,)

        # 2. Конкатенируем сегменты
        output_dir = folder_paths.get_output_directory()
        full_output_folder = os.path.join(output_dir, "video")
        os.makedirs(full_output_folder, exist_ok=True)

        counter = int(time.time() * 1000)
        output_file = f"{filename_prefix}_{counter}.mp4"
        output_path = os.path.join(full_output_folder, output_file)

        self._concat_with_ffmpeg(segment_paths, output_path, join_mode, transition_duration)

        print(f"[VideoSegmentationGenerateAndConcat] ✅ Saved to: {output_path}")
        return ({"video": output_path},)
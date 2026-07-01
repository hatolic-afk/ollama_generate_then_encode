"""
ComfyUI Custom Node: Ollama Random Prompt + Conditioning for Flux 2 Klein
СПЕЦИАЛЬНАЯ ВЕРСИЯ ДЛЯ QWEN 3.5
Категория: Hatolic
Легендарный fallback: уточки в ванной
"""

import torch
import requests
import json
import time
import folder_paths
import os
import re
from datetime import datetime


class Ollama_RandomPrompt_Conditioning:
    
    @classmethod
    def INPUT_TYPES(cls):
        models_list = cls.get_ollama_models()
        return {
            "required": {
                "ollama_host": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ollama_model": (models_list, {"default": models_list[0] if models_list else "qwen35-uncensored-fixed"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "Generate a detailed image prompt, 200-500 words, as a single vivid scene. Natural language, no lists, no tags. Describe anything: a person, a place, an object, an interior, an exterior — completely random each time. Output only the prompt."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 5}),
                "save_prompts": ("BOOLEAN", {"default": True, "label": "💾 Save prompts to file"}),
            },
            "optional": {
                "clip": ("CLIP",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "generated_prompt")
    FUNCTION = "go"
    CATEGORY = "Hatolic"
    
    @classmethod
    def get_ollama_models(cls):
        try:
            response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                if models:
                    return models
        except:
            pass
        return ["qwen35-uncensored-fixed", "gemma:latest", "llama3.2:latest"]

    def get_prompt_counter(self, filename):
        """Получает последний номер промпта из файла"""
        try:
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # Ищем последнюю строку с номером
                    for line in reversed(lines):
                        match = re.match(r'^(\d+)\.', line.strip())
                        if match:
                            return int(match.group(1))
            return 0
        except:
            return 0

    def save_prompt_to_file(self, prompt, model, temperature, seed):
        """Сохраняет сгенерированный промпт в файл с нумерацией"""
        try:
            # Папка для сохранения
            output_dir = os.path.join(folder_paths.get_output_directory(), "prompts")
            
            # Создаем папку если её нет
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"[Ollama] 📁 Created directory: {output_dir}")
            
            # Имя файла: prompts_2026-07-01.txt
            today = datetime.now().strftime("%Y-%m-%d")
            filename = os.path.join(output_dir, f"prompts_{today}.txt")
            
            # Получаем следующий номер
            counter = self.get_prompt_counter(filename)
            next_number = counter + 1
            
            # Записываем промпт с номером
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"{next_number}. {prompt}\n\n")
            
            print(f"[Ollama] 💾 Saved #{next_number} to: {filename}")
            print(f"[Ollama] 📝 Prompt #{next_number} length: {len(prompt)} chars")
            return True
            
        except Exception as e:
            print(f"[Ollama] ❌ FAILED to save: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_prompt(self, host, model, system_prompt, temperature, seed, max_retries):
        url = f"{host}/api/chat"
        
        # ============================================================
        # СПЕЦИАЛЬНО ДЛЯ QWEN 3.5
        # ============================================================
        is_qwen35 = 'qwen35' in model.lower() or 'qwen3.5' in model.lower()
        
        if is_qwen35:
            print(f"[Ollama] 🐉 Qwen 3.5 mode (ChatML + thinking OFF)")
            
            # Формируем сообщения в правильном формате для Qwen 3.5
            combined_prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
Generate a random image prompt now. Follow the instructions exactly. Output only the prompt. Nothing else. No greetings, no explanations.
<|im_end|>
<|im_start|>assistant
"""
            
            messages = [
                {"role": "user", "content": combined_prompt}
            ]
            
            options = {
                "temperature": temperature,
                "num_predict": 800,
                "min_p": 0.1,
                "repeat_penalty": 1.1,
                "top_k": 20,
                "top_p": 0.9,
                "chat_template_kwargs": {"enable_thinking": False}
            }
            
            timeout = 180
            
        # ============================================================
        # ОБЫЧНЫЙ QWEN (2.5, 3)
        # ============================================================
        elif 'qwen' in model.lower():
            print(f"[Ollama] 🐉 Qwen mode")
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\nGenerate a random image prompt now. Output only the prompt."}
            ]
            options = {
                "temperature": temperature,
                "num_predict": 700,
                "min_p": 0.1,
                "repeat_penalty": 1.1,
            }
            timeout = 150
            
        # ============================================================
        # GEMMA
        # ============================================================
        elif 'gemma' in model.lower():
            print(f"[Ollama] 🔱 Gemma mode")
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\nGenerate a random image prompt now. Follow the instructions exactly. Output only the prompt. Nothing else."}
            ]
            options = {
                "temperature": temperature,
                "num_predict": 800,
                "min_p": 0.1,
                "repeat_penalty": 1.1,
            }
            timeout = 180
            
        # ============================================================
        # LLAMA / MISTRAL
        # ============================================================
        else:
            print(f"[Ollama] 🦙 Standard mode")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate a random image prompt now. Follow the instructions exactly. Output only the prompt. Nothing else."}
            ]
            options = {
                "temperature": temperature,
                "num_predict": 900,
                "min_p": 0.1,
                "repeat_penalty": 1.1,
            }
            timeout = 180

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options
        }

        if seed > 0:
            payload["options"]["seed"] = seed

        # ============================================================
        # ОТПРАВЛЯЕМ
        # ============================================================
        for attempt in range(max_retries):
            try:
                print(f"[Ollama] 🔄 Attempt {attempt + 1}/{max_retries}")
                print(f"[Ollama] 📊 num_predict: {options.get('num_predict', 'N/A')}")
                if is_qwen35:
                    print(f"[Ollama] 🧠 thinking: OFF")
                
                response = requests.post(url, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                
                generated = result.get("message", {}).get("content", "").strip()
                
                if is_qwen35 and generated:
                    generated = re.sub(r'<think>.*?</think>', '', generated, flags=re.DOTALL)
                    generated = generated.strip()
                
                print(f"[Ollama] 📥 Got {len(generated)} chars")
                if generated:
                    print(f"[Ollama] 📄 Preview: {generated[:100]}...")
                
                if generated and len(generated) > 50:
                    print(f"[Ollama] ✅ SUCCESS: {len(generated)} chars")
                    return generated
                else:
                    print(f"[Ollama] ⚠️ Too short ({len(generated)} chars), retry")
                    time.sleep(1)
                    
            except requests.exceptions.Timeout:
                print(f"[Ollama] ⏰ TIMEOUT after {timeout}s")
                time.sleep(2)
            except Exception as e:
                print(f"[Ollama] ❌ Error: {e}")
                time.sleep(2)

        # ============================================================
        # ЛЕГЕНДАРНЫЙ FALLBACK
        # ============================================================
        print("[Ollama] 🦆 FALLBACK: Rubber ducks in the bathroom!")
        return """A bathroom turned upside down. The bathtub is filled with rubber ducks wearing tiny sunglasses, all floating in bright pink bubblegum-scented foam. On the mirror, someone wrote 'You are awesome' in lipstick backwards. A potted cactus on the toilet tank wears a party hat. A single disco ball spins slowly above the sink, throwing sparkles across the ceiling tiles. Wide angle, vibrant neon pink and turquoise lighting, fun and chaotic, late night party aftermath style."""

    def go(self, ollama_host, ollama_model, system_prompt, temperature, seed, max_retries, save_prompts=True, clip=None):
        generated_prompt = self.generate_prompt(ollama_host, ollama_model, system_prompt, temperature, seed, max_retries)
        
        print(f"[Ollama] 📏 Final length: {len(generated_prompt)} chars")
        
        # ============================================================
        # СОХРАНЯЕМ ПРОМПТ В ФАЙЛ
        # ============================================================
        if save_prompts:
            saved = self.save_prompt_to_file(generated_prompt, ollama_model, temperature, seed)
            if not saved:
                print("[Ollama] ⚠️ WARNING: Could not save prompt to file!")
                print(f"[Ollama] 💡 Try checking permissions for: {folder_paths.get_output_directory()}")
        else:
            print("[Ollama] ⏭️ Saving disabled")

        if clip is None:
            print("[Ollama] ⚠️ No CLIP input - returning prompt only")
            return (None, generated_prompt)

        tokens = clip.tokenize(generated_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        conditioning = [[cond, {"pooled_output": pooled}]]

        return (conditioning, generated_prompt)


NODE_CLASS_MAPPINGS = {
    "Ollama Random Prompt Conditioning": Ollama_RandomPrompt_Conditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ollama Random Prompt Conditioning": "🎲 Ollama → Random Prompt → Flux",
}
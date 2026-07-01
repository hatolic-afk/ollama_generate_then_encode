# 🎲 Ollama Random Prompt Generator for ComfyUI

**Автоматическая генерация случайных промптов для Flux с использованием локального Ollama**

---

## 📋 Описание

Кастомная нода для **ComfyUI**, которая автоматически генерирует случайные промпты для генерации изображений с помощью локальных LLM моделей через **Ollama**. Нода подготавливает готовые **conditioning** для моделей Flux, экономя время на ручном создании промптов и расширяя творческие возможности. Специальная оптимизация для **Qwen 3.5** с автоматическим отключением режима мышления.

---

## ✨ Возможности

- 🤖 Поддержка Qwen 3.5, Qwen 2.5, Gemma, Llama, Mistral
- 🎯 Специальная оптимизация для Qwen 3.5 (отключение мышления)
- 💾 Автосохранение промптов в файл с нумерацией
- 🎲 Случайная генерация - каждый раз новый промпт
- 🔧 Настройка температуры, seed, системного промпта
- 🦆 Надежный fallback при ошибках
- 📝 Готовые conditioning для Flux

---

## 🚀 Установка

```bash
# 1. Установите Ollama
curl -fsSL https://ollama.com/install.sh | sh  # macOS/Linux
# Windows: скачайте с https://ollama.com/download/windows

# 2. Установите модель
ollama pull qwen35-uncensored-fixed  # Рекомендуется
# или: ollama pull gemma:latest, llama3.2:latest

# 3. Установите ноду в ComfyUI
cd ComfyUI/custom_nodes/
git clone https://github.com/yourusername/ComfyUI-Ollama-Random-Prompt.git

# 4. Запустите Ollama
ollama serve

# 5. Перезапустите ComfyUI

 Сохранение промптов
Промпты сохраняются в: ComfyUI/output/prompts/prompts_YYYY-MM-DD.txt

Формат:
1. [текст промпта]
2. [текст промпта]
3. [текст промпта]


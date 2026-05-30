# MiniGrid NanoVLM fine-tuning

Проект реализует пайплайн дообучения (fine-tuning), который адаптирует vision-and-language модель (NanoVLM) для управления агентом в среде MiniGrid EmptyEnv.

## Методы
- SFT на парах (image, action) от экспертной политики
- GRPO с прямым выводом действия
- GRPO с выводом текст + действие

## Модель
NanoVLM-222M: SigLIP vision encoder + SmolLM2-135M language decoder + modality projector.

## Среда
MiniGrid-Empty-6x6-v0. Действия: `left` (0), `right` (1), `forward` (2). Наблюдение: RGB-рендер.

## Установка зависимостей
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
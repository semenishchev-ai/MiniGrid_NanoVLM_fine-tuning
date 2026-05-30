# MiniGrid NanoVLM fine-tuning

Проект реализует пайплайн дообучения (fine-tuning), который адаптирует vision-and-language модель (NanoVLM) для управления агентом в среде MiniGrid EmptyEnv.

## Методы
- SFT на парах (image, action) от экспертной политики
- GRPO с прямым выводом действия
- GRPO с выводом текст + действие

## Установка зависимостей
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone --branch v0.1 --depth 1 https://github.com/huggingface/nanoVLM.git external/nanoVLM
```

## Модель

Базовая модель: [lusxvr/nanoVLM-222M](https://huggingface.co/lusxvr/nanoVLM-222M) (SigLIP-base + SmolLM2-135M, ветка nanoVLM v0.1). Загрузка: `src/model.py`, функция `load_vlm` — `VisionLanguageModel.from_pretrained` из `external/nanoVLM`.

Предсказание действия — генерация текста через LM head: ответ одно слово (`left` / `right` / `forward`). Промпт: `model.prompt` в `configs/base.yaml`.

Датасет для SFT: `src/dataset.py`, класс `MiniGridActionDataset`. Читает `data/episodes/`, каждый шаг — пара (PNG, имя действия). Формат текста как в nanoVLM VQA: `Question: {prompt} Answer:` + метка `answer`. Батчи: `make_collator` → `ActionCollator` (маска loss только на токенах ответа, логика как `VQACollator` в nanoVLM).

## Среда и эксперт

Среда: `MiniGrid-Empty-Random-6x6-v0` (`configs/base.yaml`, `env.name`) — случайные старт агента и цель каждый эпизод. Базовый класс — `minigrid.minigrid_env.MiniGridEnv`. Обёртка `src/env.py`, класс `MiniGridWrapper`: поверх EmptyEnv цепочка `RGBImgObsWrapper` (полный RGB в `obs["image"]`) и `ImgObsWrapper` (наблюдение — только массив изображения). Размер кадра: `height * tile_size` × `width * tile_size`, `tile_size` по умолчанию берётся из среды (32 → 192×192 для 6×6).

Действия: `left` (0), `right` (1), `forward` (2). Константы и маппинг имён — в `src/env.py`.

Эксперт: `src/expert.py`, класс `ExpertPolicy`. Политика знает полную карту (`env.unwrapped.grid`), ищет клетку `goal`, строит кратчайший путь BFS по проходимым клеткам (`None`, `floor`, `goal`), выдаёт поворот или `forward` к следующей клетке пути.


## Данные

Сбор: `python scripts/collect_data.py` (параметры из `configs/base.yaml`: `data.dir`, `data.num_episodes=1000`, `env.*`). `data/episodes` перезаписывается полностью.

Эксперт проходит 1000 эпизодов (сиды `seed` … `seed + 999`). Для эпизода `i` создаётся каталог `data/episodes/{i:06d}/`:
- `{step:04d}.png` — RGB-наблюдение перед шагом `step`;
- `actions.json` — список действий `actions` (int 0/1/2), дублирование в `action_names`, `num_steps`, `seed`.

Логика сбора: `src/data_collection.py` (`collect_episode`, `collect_dataset`).

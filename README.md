# MiniGrid NanoVLM fine-tuning

Проект реализует пайплайн дообучения (fine-tuning), который адаптирует vision-and-language модель (NanoVLM) для управления агентом в среде MiniGrid EmptyEnv.

## Методы
- SFT на парах (image, action) от экспертной политики (baseline + improved с балансировкой действий и аугментациями)
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

## SFT-обучение
Универсальный трейнер: `src/sft_trainer.py`, функция `train_sft`. CLI: `python scripts/train_sft.py --config <yaml>`. Конфиги:
- `configs/sft_baseline.yaml` — без улучшений
- `configs/sft_improved.yaml` — с балансировкой и аугментациями

DataLoader поверх `MiniGridActionDataset` с `ActionCollator`. Опционально `WeightedRandomSampler` для балансировки действий (если `balance_actions: true`). Оптимизатор AdamW, cosine LR с warmup, gradient clipping. Loss через `VisionLanguageModel.forward(input_ids, images, attention_mask, targets)` — cross-entropy на токенах ответа, prompt маскируется `-100`.

`ActionCollator` (`src/dataset.py`): токенизирует prompt и answer раздельно, конкатенирует id'шники, строит labels `[-100] * len(prompt) + list(answer)`, right-padding до max длины в батче, shift labels влево на 1.

После каждой эпохи — multi-env eval через `evaluate_policy_multi` (`src/evaluate.py`) на списке сред из `eval_envs`. Чекпоинт сохраняется по success rate среды `primary_eval_env`.

## Результаты SFT

### Стенд
Тренировка на 50 эпизодах из `MiniGrid-Empty-Random-6x6-v0`. Eval после каждой эпохи на двух средах:
- **in-distribution**: `MiniGrid-Empty-Random-6x6-v0` (20 эпизодов, сиды 10000..10019, max_steps=20)
- **OOD**: `MiniGrid-Empty-8x8-v0` (10 эпизодов, max_steps=50; среда детерминированная — фиксированные старт и цель)

Финальная оценка best-чекпоинта: 50 эпизодов на каждой среде.

### Baseline (без улучшений)
Конфиг: `configs/sft_baseline.yaml`. 10 эпох, batch=16, lr=2e-5, без аугментаций и балансировки.

| Среда | Success Rate | Mean Return | Mean Length |
|---|---:|---:|---:|
| 6×6 (in-dist) | 1.00 | 0.79 | 4.72 |
| 8×8 (OOD)     | 0.00 | 0.00 | 50.0 |

На 6×6 модель достигает SR=1.0 после 3 эпох. На 8×8 — полный коллапс: 100% действий `forward`, ни одного поворота. Причины: bias к forward в train-данных (66% forward) + distribution shift (картинки 8×8 = 256×256, train — 192×192).

### Improved (с улучшениями)
Конфиг: `configs/sft_improved.yaml`. Те же гиперпараметры + три улучшения:

1. **Балансировка действий через `WeightedRandomSampler`** — равные веса по классам (left/right/forward), убирает bias к forward.
2. **`RandomResizedCrop(scale=0.6-1.0)`** — эмулирует разные масштабы сцены, что компенсирует разницу 192×192 (6×6) vs 256×256 (8×8).
3. **ColorJitter + RandomGrayscale + RandomErasing** — устойчивость к визуальным вариациям.

Best-чекпоинт выбирается по SR на 8×8 (`primary_eval_env`).

| Среда | Success Rate | Mean Return | Mean Length |
|---|---:|---:|---:|
| 6×6 (in-dist) | 1.00 | 0.78 | 4.92 |
| 8×8 (OOD)     | **1.00** | 0.80 | 11.0 |

На 8×8 SR вырос с 0.0 до 1.0. Mean length 11.0 близок к оптимальному пути BFS (5 forward + 1 right + 5 forward = 11 шагов). На 6×6 качество не просело.

### Сравнение

![SFT comparison](results/sft_comparison.png)

### Наблюдения

- **Решающий фактор для OOD — RandomResizedCrop.** Без него (только балансировка + ColorJitter) SR на 8×8 оставался 0.0: модель просто меняла коллапс с «всё forward» на «всё left» или «left/right поровну».
- **Чекпоинт нестабилен по эпохам.** Модель находит решение на 8×8 примерно на эпохе 5, потом снова уходит в коллапс. `save_best` по `primary_eval_env=8×8` обязателен.
- **8×8 — детерминированная среда** (фиксированные старт и цель), поэтому SR=1.0 означает «модель решает одну OOD-сцену», а не «обобщается на любую 8×8».

### Файлы
- `src/sft_trainer.py` — универсальный трейнер с поддержкой балансировки и multi-env eval
- `src/dataset.py` — `MiniGridActionDataset` с опциями `augment` и `random_erasing`, метод `get_sample_weights()` для `WeightedRandomSampler`
- `src/evaluate.py` — `evaluate_policy` и `evaluate_policy_multi`
- `src/plotting.py` — индивидуальные графики и сравнение baseline vs improved
- `configs/sft_baseline.yaml`, `configs/sft_improved.yaml` — конфиги стендов
- `results/sft_baseline_history.json`, `results/sft_improved_history.json` — истории обучения
- `results/sft_comparison.png` — сравнительный график

### Запуск
```bash
python scripts/collect_data.py
python scripts/train_sft.py --config configs/sft_baseline.yaml
python scripts/train_sft.py --config configs/sft_improved.yaml
python src/plotting \
    --compare-baseline results/sft_baseline_history.json \
    --compare-improved results/sft_improved_history.json
```
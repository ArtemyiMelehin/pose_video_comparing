# Pose Video Compare (MediaPipe + DTW + Tips)

Демо для сравнения двух видео с человеком (йога/танцы):
- извлекаем 33 ключевые точки (MediaPipe Pose),
- нормализуем позы (центр — таз, масштаб — плечи, выравнивание по плечам),
- сравниваем метриками **cosine** и **weighted L1**,
- выравниваем по времени **DTW** (Sakoe–Chiba),
- строим **heatmap по суставам**, сохраняем **видео-оверлей** с подсветками и **топ-3 подсказками** по кадрам.

## Установка

```bash
python -m venv pose_comparation_env
# Windows: pose_comparation_env\Scripts\activate
# Unix:    source pose_comparation_env/bin/activate

pip install -r requirements.txt
pip install -e .   # установить модуль pose_eval (из src/)


# Быстрый старт

## Сложите видео в data/ (не обязательно, пути можно указывать любые) и запустите:

```bash
python scripts/compare_videos.py \
  --ref data/ref/reference_video.mp4 \
  --usr data/usr/user_video.mp4 \
  --alpha 0.7 \
  --band 15 \
  --tips_side left
```

Все артефакты прогона попадут в `outputs/runs/<timestamp>/`:

* `metrics.csv` — покадровые метрики по DTW-пути (`ref_idx, usr_idx, cosine, wL1, mix_cost`)
* `metrics.png` — график `mix_cost`, `1 - cosine`, `wL1`
* `joints_stats.csv` — средняя L1-ошибка по каждому суставу
* `joints_heatmap.png` — горизонтальный бар-чарт по суставам (по убыванию ошибки)
* `overlay_heat.mp4` — видео «реф | пользователь» со скелетом, heat-кружками, подсветкой сегментов и топ-3 подсказками на русском
* `summary.json` — краткая сводка (средние метрики и параметры)

## Параметры

* `--alpha` — вес смеси стоимости: `cost = α*(1 - cosine) + (1-α)*wL1` (0.6–0.8 обычно хорошо)
* `--band` — ширина ленты DTW (кадры); 10–20 — сбалансировано
* `--tips_side` — где рисовать подсказки на пользовательском кадре: `left` или `right`
* `--outdir` — если указать, результаты сложатся именно туда (вместо `outputs/runs/<timestamp>`)

## Где что в коде

```
src/pose_eval/
  backends/mediapipe_backend.py   # извлечение поз из видео (33 точки + visibility)
  core/metrics.py                 # cosine_similarity, weighted_l1, compute_cosine_wl1
  core/dtw.py                     # DTW с лентой (Sakoe–Chiba)
  core/normalize.py               # индексы и нормализация поз
  core/tips.py                    # углы, пороги, генерация подсказок
  viz/draw.py                     # базовая отрисовка скелета и цветов
  viz/overlay.py                  # сборка видео-оверлея (heat + подсветки + PIL-текст)
  io/exports.py                   # создание папки прогона, сохранение CSV/PNG/summary
scripts/
  compare_videos.py               # основной CLI скрипт
  extract_reference.py            # (опц.) извлечь позы референса в .npz
```

## Тонкая настройка

Пороги подсветки и подсказок находятся в `pose_eval/core/tips.py`:

```python
TH_MINOR = 10   # жёлтая подсветка
TH_MAJOR = 20   # красная подсветка
```

Поднимайте/опускайте их под планку строгости (йога — 7–10°, танцы — 10–20°).

## Известные ограничения

* MediaPipe Pose — 2D-оценка; при сильных поворотах/окклюзиях возможны ошибки.
* Убедитесь, что исходные видео имеют стабильную ориентацию; модуль сам ресайзит стороны
  под общий размер, но не вращает кадры по EXIF.
* Видимость из MediaPipe используется в метриках; точки с `visibility < 0.3` практически игнорируются.

```

## Демонстрация
![Сравнение поз](docs/demo_pose_estimation.gif)

# Подробный план дальнейшей работы

Дата: 2026-06-20

## Цель

Проверить, можно ли восстанавливать перемещение БПЛА без GNSS по данным IMU, optical flow, барометра и признакам моторов. GPS/GNSS используется только как эталонная траектория для обучения, валидации и визуального сравнения.

Итоговый результат должен отвечать на три вопроса:

1. Какие признаки реально помогают оценивать `dx, dy, dz`.
2. Какая модель лучше восстанавливает перемещение на новых полетах.
3. Насколько быстро накапливается drift при восстановлении полной траектории без GPS.

## Разделение исходных и сгенерированных данных

Исходные данные:

- `artifacts/data.csv` - исходный CSV модуля позиционирования, внутри найдено 7 сегментов по сбросам `TimeStamp`.
- `artifacts/linear_15_01_2025.csv` - отдельный исходный полет.
- `artifacts/triangle_15_01_2025.csv` - отдельный исходный полет.
- `artifacts/2025-01-15 16-46-48.log` - исходный ArduPilot DataFlash text log.
- `artifacts/2025-01-15 16-46-48.bin` - исходный DataFlash binary log.
- `derived/dataflash/*.csv` - экспорт из DataFlash, производные табличные данные, но не синтетическая генерация модели.

Сгенерированные артефакты:

- `artifacts/generated/` - карты, replay, GeoJSON, manifest-файлы.
- `reports/*.md` - отчеты и планы.
- Будущие датасеты для обучения должны лежать в `derived/datasets/`.
- Будущие предсказания моделей должны лежать в `derived/predictions/`.
- Будущие модели должны лежать в `models/`.

Правило: не записывать новые результаты в корень `artifacts/`, чтобы не спутать их с исходными файлами.

## Текущее состояние

Сделано:

- Экспортированы DataFlash-сообщения в `derived/dataflash/`.
- Построены GPS-карты и replay полетов.
- Добавлена инвентаризация GPS-файлов: `reports/gps_flight_inventory.md`.
- Найдено, что `artifacts/data.csv` содержит 7 отдельных сегментов/полетов.
- Сделаны первые ridge-baseline эксперименты по исходным CSV и DataFlash.
- Для DataFlash показано, что оконные признаки с моторами лучше нулевого baseline на горизонте 5 секунд.

Ключевой вывод: случайные или хронологические split внутри одного полета недостаточны. Следующая оценка должна делаться на полетах, которых не было в обучении.

## Журнал выполнения

### 2026-06-20 - Фаза 1 выполнена

Создан скрипт `src/build_flight_index.py`.

Сгенерированы:

- `derived/datasets/flight_index.csv`;
- `derived/datasets/flight_index.json`;
- `reports/flight_index.md`.

Индекс содержит 10 GPS-полетов:

- 7 сегментов из `artifacts/data.csv`: `module_data_s01` ... `module_data_s07`;
- `linear_15_01_2025`;
- `triangle_15_01_2025`;
- `dataflash_2025_01_15`.

No-GPS временные блоки из `artifacts/data.csv` не включены в индекс полетов. Они не являются пригодными GPS-target сегментами для текущей задачи.

### 2026-06-20 - Фаза 2 выполнена

Создан скрипт `src/prepare_flight_tracks.py`.

Сгенерированы:

- `derived/datasets/tracks/{flight_id}_track.csv` для всех 10 полетов;
- `artifacts/generated/gps_flights/{flight_id}/map.html`;
- `artifacts/generated/gps_flights/{flight_id}/simulation.html`;
- `artifacts/generated/gps_flights/{flight_id}/path.svg`;
- `artifacts/generated/gps_flights/{flight_id}/track.geojson`;
- `artifacts/generated/gps_flights/{flight_id}/manifest.json`;
- `reports/flight_tracks.md`.

Проверено:

- количество точек в каждом `track.csv` совпадает с `flight_index.csv`;
- `time_s` начинается с нуля внутри каждого `flight_id`;
- `east_m`, `north_m`, `up_m` считаются от первой GPS-точки конкретного полета;
- накопленная `distance_m` монотонна;
- длина трека совпадает с индексом полетов.

### 2026-06-20 - Добавлен единый просмотрщик треков

Создан скрипт `src/build_track_viewer.py`.

Сгенерирован файл:

- `artifacts/generated/gps_flights/index.html`.

Viewer позволяет из одного места:

- выбирать `flight_id`;
- фильтровать треки по `module` / `dataflash`;
- открывать карту;
- смотреть replay;
- смотреть SVG-график;
- открывать `track.csv`, GeoJSON и manifest.

`src/prepare_flight_tracks.py` теперь автоматически пересобирает этот viewer после обновления треков.

### 2026-06-20 - Фаза 3 начата: module-only оконные датасеты

Создан скрипт `src/build_window_dataset.py`.

Сгенерированы module-only датасеты:

- `derived/datasets/windows_module_h1000_l1000.npz`;
- `derived/datasets/windows_module_h3000_l3000.npz`;
- `derived/datasets/windows_module_h5000_l5000.npz`;
- `derived/datasets/windows_module_h1000_l1000_features.json`;
- `derived/datasets/windows_module_h3000_l3000_features.json`;
- `derived/datasets/windows_module_h5000_l5000_features.json`;
- `derived/datasets/windows_module_h1000_l1000_meta.csv`;
- `derived/datasets/windows_module_h3000_l3000_meta.csv`;
- `derived/datasets/windows_module_h5000_l5000_meta.csv`;
- `reports/window_datasets.md`.

Параметры текущих датасетов:

- источник признаков: только `module`;
- полеты: 9 module-полетов (`module_data_s01` ... `module_data_s07`, `linear_15_01_2025`, `triangle_15_01_2025`);
- сенсорный поток дискретизирован до `20 ms`;
- входные признаки: агрегаты `last`, `mean`, `std`, `min`, `max`, `delta`, `integral_s`;
- размер `X`: 168 признаков;
- target: `dx_east_m`, `dy_north_m`, `dz_up_m`.

Проверено:

- GPS/GNSS-колонки не входят в `feature_names`;
- `.npz` массивы не содержат NaN/inf;
- `flight_id` сохранен для каждого окна;
- DataFlash не смешан с module-only датасетами.

### 2026-06-20 - Фазы 4/5 начаты: split и baseline по оконным датасетам

Создан скрипт `src/run_window_baselines.py`.

Сгенерированы:

- `reports/experiments/module_window_baselines.md`;
- `derived/predictions/module_window_baselines/.../*_pred.csv`.

Проверенные split-сценарии:

- `module_data_holdout`: train `module_data_s01..s05`, validation `module_data_s06`, test `module_data_s07`;
- `route_holdout_triangle`: test `triangle_15_01_2025`;
- `route_holdout_linear`: test `linear_15_01_2025`.

Проверенные baseline-модели:

- `zero`;
- `train_mean`;
- `ridge` с выбором `alpha` по validation-полету.

Текущий вывод:

- на честных holdout-полетах `ridge` пока не превосходит `zero/train_mean`;
- это указывает на слабую переносимость текущих агрегатных признаков между полетами;
- перед нейросетевыми моделями нужно проверить target, начальные скачки высоты, нормализацию маршрутов и устойчивость признаков.

### 2026-06-20 - Диагностика target и отладочные split

Созданы скрипты:

- `src/analyze_window_targets.py`;
- `src/filter_window_datasets.py`;
- `src/run_chronological_debug.py`.

Сгенерированы:

- `reports/experiments/module_window_target_diagnostics.md`;
- `reports/window_dataset_filters.md`;
- `derived/datasets/windows_module_h1000_l1000_trim5.npz`;
- `derived/datasets/windows_module_h3000_l3000_trim5.npz`;
- `derived/datasets/windows_module_h5000_l5000_trim5.npz`;
- `reports/experiments/module_window_baselines_trim5.md`;
- `reports/experiments/module_window_chronological_debug.md`.

Найдено:

- у `module_data_s04`, `module_data_s06`, `triangle_15_01_2025` есть подозрительные стартовые скачки высоты;
- у `triangle_15_01_2025` высота в начале падает примерно на 35 м за первые секунды;
- удаление первых 5 секунд почти не меняет итог baseline;
- chronological-debug внутри тех же полетов тоже оставляет `ridge` хуже `zero`.

Вывод:

- проблема не только в стартовых скачках;
- текущая линейная модель на глобальных `east/north/up` target не извлекает устойчивый сигнал из агрегатов сенсоров;
- следующий технический шаг: проверить альтернативную постановку target, прежде всего horizontal-only и body/path-relative displacement, затем уже переходить к nonlinear/sequence-моделям.

### 2026-06-20 - Проверены horizontal-only и path-relative target

Обновлен `src/run_window_baselines.py`:

- добавлен `--target-mode xy`;
- baseline может оценивать только горизонтальный target без `dz`;
- отчеты используют имена target из `.npz`.

Создан скрипт:

- `src/build_path_relative_datasets.py`.

Сгенерированы:

- `reports/experiments/module_window_baselines_xy.md`;
- `derived/predictions/module_window_baselines_xy_base/.../*_pred.csv`;
- `derived/datasets/windows_module_h1000_l1000_pathrel.npz`;
- `derived/datasets/windows_module_h3000_l3000_pathrel.npz`;
- `derived/datasets/windows_module_h5000_l5000_pathrel.npz`;
- `reports/path_relative_datasets.md`;
- `reports/experiments/module_window_baselines_pathrel.md`;
- `derived/predictions/module_window_baselines_pathrel_clean/.../*_pred.csv`.

Вывод:

- `horizontal-only` уменьшает абсолютные ошибки, но не меняет ранжирование моделей: `zero` остается лучшим почти везде;
- `path-relative` target (`along_m`, `cross_m`) тоже не делает `ridge` лучше простых baseline;
- в path-relative датасетах валидное направление движения есть примерно у трети окон, значит большая часть окон относится к зависанию или очень малому смещению;
- следующий шаг: явно разделить `hover` и `moving` окна, запустить baseline только на moving-окнах и отдельно решить, нужно ли моделировать hover как отдельный режим.

### 2026-06-20 - Проверены moving-only окна

Обновлен `src/filter_window_datasets.py`:

- добавлен фильтр `--min-horizontal-target-m`;
- default-фильтр больше не захватывает `trim`, `pathrel` и `move` производные датасеты.

Сгенерированы:

- `derived/datasets/windows_module_h1000_l1000_move1.npz`;
- `derived/datasets/windows_module_h3000_l3000_move1.npz`;
- `derived/datasets/windows_module_h5000_l5000_move1.npz`;
- `reports/window_dataset_filters_move1.md`;
- `reports/experiments/module_window_baselines_move1.md`;
- `derived/predictions/module_window_baselines_move1/.../*_pred.csv`.

Фильтр:

- `time_s >= 5`;
- горизонтальный target `sqrt(dx^2 + dy^2) >= 1 m`.

Результат:

- moving-only сильно уменьшает число окон: например `h1000_l1000` с 42613 до 10273;
- на `route_holdout_linear` ridge стал лучшим для горизонтов 1/3/5 секунд;
- на `route_holdout_triangle` и `module_data_holdout` ridge все еще не лучший;
- для `module_data_holdout` после фильтра мало train-окон, особенно на 1 секунде.

Вывод:

- сенсоры действительно содержат полезный сигнал на движущихся окнах;
- текущая модель неустойчива между маршрутами;
- следующий шаг: строить двухрежимную постановку `hover vs moving`, а regression-модель обучать/оценивать отдельно на moving; затем попробовать нелинейный baseline на moving-only.

### 2026-06-20 - Проверены two-stage и nonlinear moving-only baseline

Добавлен `src/run_two_stage_baseline.py`.

Постановка:

- binary gate определяет режим `hover/moving`;
- `moving` означает горизонтальный target `sqrt(dx^2 + dy^2) >= 1 m`;
- если gate предсказывает `hover`, displacement = 0;
- если gate предсказывает `moving`, используется ridge-регрессия, обученная только на moving-окнах;
- дополнительно считается diagnostic-only `oracle_gate`, где режим берется из true target;
- для устойчивости добавлены `two_stage_clipped` и `oracle_gate_clipped`: норма предсказания ограничена 95-м перцентилем train moving target.

Сгенерированы:

- `reports/experiments/module_window_two_stage.md`;
- `derived/predictions/module_window_two_stage/.../*_pred.csv`.

Главный результат:

- на `route_holdout_linear` идеальный gate улучшает baseline:
  - `h1000_l1000`: `oracle_gate_clipped` MAE 3D 3.021 против `zero` 3.262;
  - `h3000_l3000`: `oracle_gate_clipped` 8.556 против `zero` 9.585;
  - `h5000_l5000`: `oracle_gate_clipped` 14.020 против `zero` 15.906;
- реальный `two_stage` хуже, потому что classifier переносится плохо: например для `h1000_l1000/route_holdout_linear` true moving rate = 0.296, predicted moving rate = 0.906;
- clipping уменьшает отдельные большие ошибки, но не решает ошибку gate.

Добавлен `src/run_rff_baseline.py`.

Постановка:

- pure NumPy, без `sklearn`/`torch`;
- nonlinear random Fourier features + ridge;
- запуск на `*_move1.npz`;
- сравнение с `zero`, `train_mean`, обычным `ridge`.

Сгенерированы:

- `reports/experiments/module_window_rff_move1.md`;
- `derived/predictions/module_window_rff_move1/.../*_pred.csv`.

Результат:

- `rff_ridge` не стал лучшим ни на одном test split;
- на `route_holdout_linear` лучшим остается обычный `ridge`;
- на `route_holdout_triangle` и `module_data_holdout` лучшими остаются `zero` или `train_mean`.

Вывод:

- простое добавление нелинейности не решает текущую проблему;
- полезный сигнал есть, но он проявляется только в части маршрутов и ломается при переносе;
- дальше стоит не усложнять один и тот же window baseline, а перейти к одному из двух направлений:
  - улучшить gate/режимы движения и отдельно анализировать ошибки classifier;
  - продолжить DataFlash/sequence-модель, где уже есть motor/attitude/battery признаки и первый baseline лучше `zero`.

### 2026-06-20 - Добавлен DataFlash sweep и viewer предсказаний

Добавлен `src/run_dataflash_sweep.py`.

Постановка:

- используется только `derived/dataflash/*.csv`, без смешивания с module-data;
- target строится из `POS.csv` как future local displacement;
- проверяются feature sets:
  - `imu`: `IMU`, `BARO`;
  - `imu_att`: `IMU`, `ATT`, `BARO`;
  - `all`: `IMU`, `ATT`, `BARO`, `BAT`, `MOTB`, `RCOU_motor_features`;
- проверяются окна `1000:1000`, `3000:3000`, `5000:5000` ms;
- split внутри одного DataFlash лога chronological `60/20/20`;
- `ridge_alpha` выбирается только по validation;
- для диагностики пишется alpha sensitivity: validation MAE 3D и test MAE 3D для всех alpha.

Сгенерированы:

- `reports/experiments/dataflash_sweep.md`;
- `derived/predictions/dataflash_sweep/.../*_pred.csv`.

Лучшие test baseline по честному validation-selected правилу:

- `imu h1000_l1000`: `ridge`, MAE 3D 2.793;
- `imu h3000_l3000`: `ridge`, MAE 3D 8.213;
- `imu h5000_l5000`: `ridge`, MAE 3D 13.697;
- `imu_att h1000_l1000`: `ridge`, MAE 3D 2.834;
- `imu_att h3000_l3000`: `ridge`, MAE 3D 8.203;
- `imu_att h5000_l5000`: `ridge`, MAE 3D 13.445;
- `all h1000_l1000`: `zero`, MAE 3D 2.839;
- `all h3000_l3000`: `zero`, MAE 3D 8.354;
- `all h5000_l5000`: `zero`, MAE 3D 13.970.

Важное наблюдение:

- старый DataFlash baseline `80/20` показывал, что `all` с моторами лучше `zero` на 5 секундах;
- новый `60/20/20` sweep показывает, что `all` чувствителен к выбору регуляризации;
- для `all h5000_l5000` validation выбирает `alpha=1`, но test MAE 3D становится 18.500;
- при более сильной регуляризации test может быть лучше, например `alpha=100` дает test MAE 3D 13.453, но validation для него хуже;
- значит один chronological validation segment не всегда надежно выбирает гиперпараметры.

Добавлен `src/build_dataflash_prediction_viewer.py`.

Сгенерирован:

- `artifacts/generated/dataflash_predictions/index.html`.

Viewer показывает для каждого prediction CSV:

- серый путь: current POS в момент предсказания;
- синий путь: true future POS;
- красный путь: predicted future POS;
- MAE 3D и P95 3D для выбранного варианта.

Важно: это viewer future-position predictions, а не автономная накопленная траектория. Overlapping window predictions пока не надо интерпретировать как полноценный inertial navigation replay.

Следующий шаг:

- сделать rolling/time-block validation для DataFlash, чтобы подбор `alpha` не зависел от одного validation-участка;
- после этого уже строить sequence-модель или drift replay на выбранной устойчивой конфигурации.

### 2026-06-20 - Добавлены rolling validation и sparse rollout для DataFlash

Добавлен `src/run_dataflash_rolling_validation.py`.

Постановка:

- DataFlash log делится на 6 contiguous chronological blocks;
- каждый fold:
  - train = все блоки до validation;
  - validation = следующий блок;
  - test = блок сразу после validation;
- при `min_train_blocks=2` получается 3 folds;
- `ridge_alpha` выбирается отдельно на validation каждого fold;
- результаты агрегируются по всем test-блокам.

Сгенерированы:

- `reports/experiments/dataflash_rolling_validation.md`;
- `derived/predictions/dataflash_rolling_validation/.../*_pred.csv`;
- `artifacts/generated/dataflash_rolling_predictions/index.html`.

Главный rolling-validation результат:

- на 1 и 3 секундах `zero` часто остается лучшим по overall MAE 3D;
- на 5 секундах `imu_att h5000_l5000 ridge` стал лучшим по overall MAE 3D:
  - `zero`: MAE 3D 16.335, RMSE 3D 24.584, P95 49.655;
  - `ridge`: MAE 3D 16.293, RMSE 3D 21.241, P95 43.857;
- `all h5000_l5000 ridge` хуже по MAE 3D: 19.227, хотя RMSE/P95 лучше zero;
- выбранные alpha нестабильны, особенно для `all`: разные folds выбирают `1`, `1000`, `100000`.

Добавлен `src/build_dataflash_rollout.py`.

Постановка sparse rollout:

- используется prediction CSV из rolling validation;
- внутри каждого fold берутся только непересекающиеся 5-секундные prediction rows;
- стартуем из true current POS первого выбранного окна;
- дальше накапливаем predicted displacement;
- сравниваем accumulated predicted POS с true future POS;
- это не IMU-rate inertial integration, а sparse 5-second displacement rollout.

Сгенерированы:

- `reports/experiments/dataflash_rollout_imu_att_h5000.md`;
- `reports/experiments/dataflash_rollout_imu_att_h5000_zero.md`;
- `reports/experiments/dataflash_rollout_imu_att_h5000_train_mean.md`;
- `reports/experiments/dataflash_rollout_summary.md`;
- `derived/predictions/dataflash_rollout/*.csv`;
- `artifacts/generated/dataflash_rollout/index.html`;
- `artifacts/generated/dataflash_rollout_zero/index.html`;
- `artifacts/generated/dataflash_rollout_train_mean/index.html`.

Rollout summary:

| model | steps | final error m | mean error m | max error m |
| --- | ---: | ---: | ---: | ---: |
| zero | 87 | 210.141 | 103.898 | 284.380 |
| train_mean | 87 | 210.890 | 97.994 | 235.566 |
| ridge | 87 | 80.832 | 116.900 | 354.817 |

Вывод:

- ridge улучшает final error на folds 1 и 3;
- ridge резко проваливается на fold 2;
- по mean rollout error ridge хуже `zero` и `train_mean`;
- оконная ошибка `dx/dy/dz` сама по себе недостаточна для выбора модели;
- следующий этап должен оценивать модели сразу по двум уровням:
  - local displacement error;
  - accumulated rollout/drift.

Следующий технический шаг:

- сделать sequence baseline для DataFlash `imu_att h5000_l5000`;
- сравнивать его с `zero/train_mean/ridge` не только по MAE/RMSE displacement, но и по sparse rollout metrics;
- если sequence-модель не улучшит fold 2, анализировать признаки/участок fold 2 отдельно.

### 2026-06-21 - Проверен sequence baseline для DataFlash

Добавлен `src/run_dataflash_sequence_baseline.py`.

Постановка:

- используется только DataFlash `imu_att`: `IMU`, `ATT`, `BARO`;
- horizon = 5000 ms;
- lookback = 5000 ms;
- sequence length = 20;
- признаки: 20 упорядоченных сэмплов из lookback-интервала, flattened в один вектор;
- модель: ridge regression на sequence-векторе;
- GPS/POS используется только как target future displacement;
- split тот же rolling time-block validation, что и для агрегатного DataFlash baseline.

Сгенерированы:

- `reports/experiments/dataflash_sequence_imu_att_h5000.md`;
- `derived/predictions/dataflash_sequence_rolling/.../*_pred.csv`;
- `artifacts/generated/dataflash_sequence_predictions/index.html`;
- `reports/experiments/dataflash_rollout_sequence_imu_att_h5000.md`;
- `artifacts/generated/dataflash_rollout_sequence/index.html`.

Результат per-fold alpha selection:

- `zero`: MAE 3D 16.251, RMSE 3D 24.523, P95 49.658;
- `sequence_ridge`: MAE 3D 18.190, RMSE 3D 23.655, P95 39.244;
- rollout mean error хуже: 155.636 m.

Вывод: per-fold выбор `alpha` нестабилен. Таблица sensitivity показала, что `alpha=100` лучший по mean validation MAE 3D.

Запущен fixed-alpha вариант:

```bash
python3 src/run_dataflash_sequence_baseline.py \
  --fixed-alpha 100 \
  --report reports/experiments/dataflash_sequence_imu_att_h5000_fixed100.md \
  --pred-dir derived/predictions/dataflash_sequence_fixed100
```

Сгенерированы:

- `reports/experiments/dataflash_sequence_imu_att_h5000_fixed100.md`;
- `derived/predictions/dataflash_sequence_fixed100/.../*_pred.csv`;
- `artifacts/generated/dataflash_sequence_fixed100_predictions/index.html`;
- `reports/experiments/dataflash_rollout_sequence_imu_att_h5000_fixed100.md`;
- `derived/predictions/dataflash_rollout/imu_att_h5000_l5000_sequence_fixed100_rollout.csv`;
- `artifacts/generated/dataflash_rollout_sequence_fixed100/index.html`.

Fixed-alpha sequence result:

- `zero`: MAE 3D 16.251, RMSE 3D 24.523, P95 49.658;
- `aggregate ridge`: MAE 3D 16.293, RMSE 3D 21.241, P95 43.857;
- `sequence_ridge fixed alpha=100`: MAE 3D 15.316, RMSE 3D 17.307, P95 30.754.

Rollout comparison:

| model | steps | final error m | mean error m | max error m |
| --- | ---: | ---: | ---: | ---: |
| zero | 87 | 210.141 | 103.898 | 284.380 |
| train_mean | 87 | 210.890 | 97.994 | 235.566 |
| aggregate_ridge | 87 | 80.832 | 116.900 | 354.817 |
| sequence_ridge_per_fold_alpha | 87 | 284.451 | 155.636 | 432.895 |
| sequence_ridge_fixed_alpha_100 | 87 | 284.451 | 95.471 | 284.451 |

Вывод:

- fixed-alpha sequence впервые дает заметное улучшение локальных метрик относительно `zero` и агрегатного ridge;
- fixed-alpha sequence немного улучшает mean rollout error относительно `train_mean`;
- final rollout error остается плохим из-за fold 3;
- следующий шаг: анализ fold 3 и/или добавить bias correction / gate по участкам, потому что локальная модель уже лучше, но накопление все еще дрейфует.

### 2026-06-21 - Добавлена validation-residual bias correction

Обновлен `src/run_dataflash_sequence_baseline.py`.

Добавлена модель:

- `sequence_ridge_bias_corrected`;
- ridge обучается как раньше;
- на validation block считается средний residual: `mean(y_val - pred_val)`;
- этот residual добавляется к test-предсказаниям текущего fold;
- test target в коррекции не используется.

Запуск:

```bash
python3 src/run_dataflash_sequence_baseline.py \
  --fixed-alpha 100 \
  --report reports/experiments/dataflash_sequence_imu_att_h5000_fixed100_bias.md \
  --pred-dir derived/predictions/dataflash_sequence_fixed100_bias
```

Сгенерированы:

- `reports/experiments/dataflash_sequence_imu_att_h5000_fixed100_bias.md`;
- `derived/predictions/dataflash_sequence_fixed100_bias/.../*_pred.csv`;
- `artifacts/generated/dataflash_sequence_fixed100_bias_predictions/index.html`;
- `reports/experiments/dataflash_rollout_sequence_imu_att_h5000_fixed100_bias.md`;
- `derived/predictions/dataflash_rollout/imu_att_h5000_l5000_sequence_fixed100_bias_rollout.csv`;
- `artifacts/generated/dataflash_rollout_sequence_fixed100_bias/index.html`.

Дополнительно добавлен `src/analyze_dataflash_fold_residuals.py`.

Сгенерирован:

- `reports/experiments/dataflash_fold_residuals_sequence_fixed100.md`.

Локальные метрики:

| model | MAE 3D | RMSE 3D | P95 3D |
| --- | ---: | ---: | ---: |
| zero | 16.251 | 24.523 | 49.658 |
| sequence_ridge fixed alpha=100 | 15.316 | 17.307 | 30.754 |
| sequence_ridge_bias_corrected | 14.295 | 16.684 | 29.954 |

Fold-level:

- fold 1: bias correction улучшает MAE 3D с 14.458 до 13.680;
- fold 2: ухудшает MAE 3D с 15.833 до 17.245;
- fold 3: улучшает MAE 3D с 15.658 до 11.959.

Rollout:

| model | steps | final error m | mean error m | max error m |
| --- | ---: | ---: | ---: | ---: |
| train_mean | 87 | 210.890 | 97.994 | 235.566 |
| aggregate_ridge | 87 | 80.832 | 116.900 | 354.817 |
| sequence_ridge_fixed_alpha_100 | 87 | 284.451 | 95.471 | 284.451 |
| sequence_ridge_bias_corrected | 87 | 52.818 | 106.143 | 234.332 |

Вывод:

- bias correction существенно улучшает local displacement metrics;
- final rollout error стал лучшим из проверенных вариантов;
- mean rollout error все еще хуже `train_mean`;
- fold 2 остается проблемным: validation residual correction переносится на него плохо.
- residual diagnostic показывает, что fold 3 исправляется почти полностью по east-bias, а fold 2 после correction получает переисправление east residual: raw mean east residual `3.311`, corrected `8.354`.

Следующий технический шаг:

- сделать анализ fold 2: сравнить распределения target/prediction/residual с folds 1 и 3;
- проверить, не меняется ли направление/режим движения или GPS segment behavior;
- попробовать ограниченную correction: применять bias correction только если validation residual стабилен или если она улучшает validation rollout.

### 2026-06-21 - Добавлен tuned shrinkage для bias correction

Обновлен `src/run_dataflash_sequence_baseline.py`.

Добавлено:

- аргумент `--tune-bias-shrink`;
- candidate shrink factors: `0.25`, `0.5`, `0.75`, `1.0`;
- для каждого fold коэффициент выбирается только по validation MAE;
- новая модель в prediction CSV: `sequence_ridge_bias_tuned`.

Запуск:

```bash
python3 src/run_dataflash_sequence_baseline.py \
  --fixed-alpha 100 \
  --tune-bias-shrink \
  --report reports/experiments/dataflash_sequence_imu_att_h5000_fixed100_shrink.md \
  --pred-dir derived/predictions/dataflash_sequence_fixed100_shrink
```

Сгенерированы:

- `reports/experiments/dataflash_sequence_imu_att_h5000_fixed100_shrink.md`;
- `derived/predictions/dataflash_sequence_fixed100_shrink/.../*_pred.csv`;
- `artifacts/generated/dataflash_sequence_fixed100_shrink_predictions/index.html`;
- `reports/experiments/dataflash_rollout_sequence_imu_att_h5000_fixed100_shrink.md`;
- `derived/predictions/dataflash_rollout/imu_att_h5000_l5000_sequence_fixed100_shrink_rollout.csv`;
- `artifacts/generated/dataflash_rollout_sequence_fixed100_shrink/index.html`.

Локальные метрики:

| model | MAE 3D | RMSE 3D | P95 3D |
| --- | ---: | ---: | ---: |
| zero | 16.251 | 24.523 | 49.658 |
| sequence_ridge fixed alpha=100 | 15.316 | 17.307 | 30.754 |
| sequence_ridge_bias_corrected | 14.295 | 16.684 | 29.954 |
| sequence_ridge_bias_tuned | 13.990 | 16.353 | 30.245 |

Rollout:

| model | steps | final error m | mean error m | max error m |
| --- | ---: | ---: | ---: | ---: |
| train_mean | 87 | 210.890 | 97.994 | 235.566 |
| aggregate_ridge | 87 | 80.832 | 116.900 | 354.817 |
| sequence_ridge_bias_corrected | 87 | 52.818 | 106.143 | 234.332 |
| sequence_ridge_bias_tuned | 87 | 52.818 | 93.538 | 187.322 |

Fold rollout:

- fold 1: final 159.646, mean 78.312, max 159.646;
- fold 2: final 126.784, mean 121.245, max 187.322;
- fold 3: final 52.818, mean 81.058, max 140.646.

Вывод:

- tuned shrinkage стал текущим лучшим вариантом по rollout;
- он сохраняет лучший final error `52.818 m`;
- mean rollout error стал лучше `train_mean`;
- max rollout error тоже стал лучше `train_mean`;
- fold 2 больше не проваливается так сильно, потому что validation выбрал shrink `0.5`.

Следующий технический шаг:

- проверить, можно ли еще улучшить fold 2 через выбор shrink по validation rollout, а не по validation MAE;
- затем оформить текущий best model как кандидат для финального отчета: `sequence_ridge_bias_tuned`, `imu_att`, `h5000_l5000`, `sequence_len=20`, `alpha=100`.

### 2026-06-21 - Проверен выбор shrink по validation rollout

Обновлен `src/run_dataflash_sequence_baseline.py`.

Добавлено:

- аргумент `--bias-shrink-metric`;
- варианты: `mae`, `rollout_mean`, `rollout_final`, `rollout_max`;
- для rollout-метрик score считается на validation block через sparse non-overlapping rollout.

Запуск:

```bash
python3 src/run_dataflash_sequence_baseline.py \
  --fixed-alpha 100 \
  --tune-bias-shrink \
  --bias-shrink-metric rollout_mean \
  --report reports/experiments/dataflash_sequence_imu_att_h5000_fixed100_shrink_rollout.md \
  --pred-dir derived/predictions/dataflash_sequence_fixed100_shrink_rollout
```

Сгенерированы:

- `reports/experiments/dataflash_sequence_imu_att_h5000_fixed100_shrink_rollout.md`;
- `derived/predictions/dataflash_sequence_fixed100_shrink_rollout/.../*_pred.csv`;
- `reports/experiments/dataflash_rollout_sequence_imu_att_h5000_fixed100_shrink_rollout.md`;
- `derived/predictions/dataflash_rollout/imu_att_h5000_l5000_sequence_fixed100_shrink_rolloutmetric_rollout.csv`;
- `artifacts/generated/dataflash_sequence_fixed100_shrink_rollout_predictions/index.html`;
- `artifacts/generated/dataflash_rollout_sequence_fixed100_shrink_rolloutmetric/index.html`.

Результат:

| shrink selection | final error m | mean error m | max error m |
| --- | ---: | ---: | ---: |
| validation MAE | 52.818 | 93.538 | 187.322 |
| validation rollout mean | 52.818 | 97.233 | 234.332 |

Вывод:

- rollout-based shrink selection улучшает fold 1;
- но fold 2 снова получает полный bias correction и ухудшается;
- текущим best остается validation-MAE shrink selection.

Следующий технический шаг:

- оформить финальный сравнительный отчет по best model;
- собрать одну страницу/раздел с командами воспроизведения, таблицами и ссылками на viewer.

### 2026-06-21 - Оформлена финальная сводка DataFlash candidate

Созданы:

- `reports/final_dataflash_report.md`;
- `artifacts/generated/dataflash_final_report/index.html`.

Содержимое:

- описание исходных и производных DataFlash-данных;
- явно указано, что `POS.csv` используется только как target/evaluation;
- текущий best model:
  - `sequence_ridge_bias_tuned`;
  - `imu_att`;
  - horizon/lookback `5000/5000 ms`;
  - `sequence_len=20`;
  - `alpha=100`;
  - shrink selection by validation MAE;
- таблица local displacement metrics;
- таблица sparse rollout metrics;
- ссылки на prediction viewer, rollout viewer, CSV и подробные отчеты.

Текущий best result:

| metric | value |
| --- | ---: |
| local MAE 3D | 13.990 m |
| local RMSE 3D | 16.353 m |
| local P95 3D | 30.245 m |
| rollout final error | 52.818 m |
| rollout mean error | 93.538 m |
| rollout max error | 187.322 m |

Следующий технический шаг:

- сделать воспроизводимый one-command pipeline для DataFlash best model;
- либо перейти к `reports/final_report.md` и собрать общий отчет по всем этапам: GPS inventory, module-data baselines, DataFlash best model, ограничения.

## Фаза 1. Нормализация полетов

Цель: получить единый каталог полетов с понятными метаданными и без смешивания сегментов.

### Задачи

1. Создать скрипт `src/build_flight_index.py`.
2. На входе читать:
   - `artifacts/data.csv` с автоматическим разбиением на 7 сегментов;
   - `artifacts/linear_15_01_2025.csv`;
   - `artifacts/triangle_15_01_2025.csv`;
   - `derived/dataflash/GPS.csv` или `derived/dataflash/POS.csv`.
3. Для каждого полета записать:
   - `flight_id`;
   - исходный файл;
   - номер сегмента;
   - формат источника: `module` или `dataflash`;
   - длительность;
   - число GPS-точек;
   - 2D-длина GPS-трека;
   - диапазон высоты;
   - частоты основных сенсоров;
   - список доступных признаков.
4. Сохранить общий индекс:
   - `derived/datasets/flight_index.csv`;
   - `derived/datasets/flight_index.json`;
   - `reports/flight_index.md`.

### Критерий готовности

- Каждый полет имеет стабильный `flight_id`.
- `data.csv` представлен как 7 отдельных полетов.
- В отчете явно указано, какие файлы являются исходными, а какие производными.

## Фаза 2. Единый формат траектории

Цель: привести GPS/POS к локальной системе координат в метрах.

### Задачи

1. Создать скрипт `src/prepare_flight_tracks.py`.
2. Для каждого `flight_id` построить таблицу:
   - `time_s`;
   - `lat`;
   - `lon`;
   - `alt_m`;
   - `east_m`;
   - `north_m`;
   - `up_m`;
   - `speed_mps`;
   - `distance_m`.
3. Система координат:
   - origin = первая валидная GPS-точка полета;
   - `east_m`, `north_m`, `up_m` в локальной ENU-системе;
   - для сравнения моделей использовать именно локальные метры.
4. Сохранить:
   - `derived/datasets/tracks/{flight_id}_track.csv`;
   - `artifacts/generated/gps_flights/{flight_id}/map.html`;
   - `artifacts/generated/gps_flights/{flight_id}/simulation.html`.

### Критерий готовности

- Для каждого полета можно открыть карту и replay.
- В `track.csv` нет скачков между сегментами.
- Суммарная длина трека совпадает с `reports/gps_flight_inventory.md` в разумных пределах.

## Фаза 3. Сбор обучающих окон

Цель: собрать supervised dataset вида `X -> dx, dy, dz`, где GPS используется только как target.

### Задачи

1. Создать скрипт `src/build_window_dataset.py`.
2. Поддержать два семейства источников:
   - `module`: IMU, optical flow, магнитометры, барометр, лидар из CSV модуля;
   - `dataflash`: `IMU`, `ATT`, `BARO`, `BAT`, `MOTB`, `RCOU_motor_features`.
3. Для каждого окна считать признаки:
   - `last`;
   - `mean`;
   - `std`;
   - `min`;
   - `max`;
   - `delta`;
   - интеграл по времени для ускорений и optical flow;
   - нормы `acc_norm`, `gyro_norm`, `mag_norm`;
   - моторные признаки: средняя тяга, разброс, диапазон, дифференциалы пар.
4. Горизонты target:
   - `1000 ms`;
   - `3000 ms`;
   - `5000 ms`;
   - опционально `10000 ms`.
5. Размеры lookback:
   - `1000 ms`;
   - `3000 ms`;
   - `5000 ms`.
6. Сохранить датасеты:
   - `derived/datasets/windows_h{horizon_ms}_l{lookback_ms}.npz`;
   - `derived/datasets/windows_h{horizon_ms}_l{lookback_ms}_features.json`;
   - `derived/datasets/windows_h{horizon_ms}_l{lookback_ms}_meta.csv`.

### Критерий готовности

- Каждое окно содержит `flight_id`.
- Входные признаки не содержат GPS-координат.
- Target `dx, dy, dz` считается из GPS/POS в метрах.
- Можно исключить целый полет из train и использовать его как test.

## Фаза 4. Правильные split-сценарии

Цель: проверять переносимость модели между полетами.

### Основные split

1. `module_data_holdout`
   - train: 5 сегментов из `artifacts/data.csv`;
   - validation: 1 сегмент из `artifacts/data.csv`;
   - test: 1 сегмент из `artifacts/data.csv`.

2. `route_holdout`
   - train: `data.csv` segments + `linear`;
   - test: `triangle`.

3. `dataflash_holdout`
   - train: module CSV flights;
   - test: DataFlash GPS/POS, если признаки сопоставимы;
   - использовать осторожно, потому что набор сенсоров отличается.

4. `chronological_debug`
   - train/test внутри одного полета;
   - использовать только для отладки, не как главный результат.

### Критерий готовности

- В отчете по каждому эксперименту указан split.
- Главные выводы делаются только по holdout-полетам.

## Фаза 5. Baseline-модели

Цель: получить надежные нижние ориентиры качества.

### Модели

1. `zero displacement`
   - всегда предсказывает `dx=0, dy=0, dz=0`.

2. `train mean displacement`
   - предсказывает среднее смещение train-окон.

3. `ridge`
   - линейная модель на агрегатах окна.

4. `random forest` или `gradient boosting`
   - нелинейный baseline на агрегатах окна.

5. `MLP`
   - простая нейросеть по агрегатам окна.

### Выходы

- `reports/experiments/{experiment_id}.md`;
- `derived/predictions/{experiment_id}/{flight_id}_pred.csv`;
- `artifacts/generated/prediction_replays/{experiment_id}/{flight_id}.html`.

### Критерий готовности

- Для каждого baseline есть таблица метрик.
- Для test-полетов есть replay `GPS vs predicted`.
- Есть сравнение с `zero displacement`.

## Фаза 6. Временные нейросетевые модели

Цель: проверить, улучшают ли sequence-модели восстановление перемещения.

### Модели

1. `GRU`
   - первый вариант, обычно проще и быстрее LSTM.

2. `LSTM`
   - сравнить с примером из `POLI_NA.zip`.

3. `1D-CNN`
   - быстрый вариант для временных паттернов.

4. `TCN` или `CNN-GRU`
   - только если первые модели дают смысл.

### Входы

Последовательности длиной:

- `1 s`;
- `3 s`;
- `5 s`;

с downsample до стабильной частоты, например `20 Hz` или `50 Hz`.

### Выход

- `dx, dy, dz` на горизонте `1/3/5 s`.

### Критерий готовности

- Есть train/val/test loss.
- Есть метрики на holdout-полетах.
- Есть replay накопленной траектории.
- Модель лучше baseline хотя бы на одном честном split.

## Фаза 7. Восстановление полной траектории

Цель: перейти от разовых `dx, dy, dz` к траектории без GPS.

### Задачи

1. Создать скрипт `src/reconstruct_trajectory.py`.
2. На вход:
   - `track.csv` с GPS для эталона;
   - `pred.csv` с предсказанными смещениями.
3. Реализовать накопление:
   - stride = `sample_ms`;
   - predicted position = cumulative sum of predicted displacement;
   - начальная точка = `0,0,0`.
4. Считать drift:
   - ошибка через `10 s`;
   - ошибка через `30 s`;
   - ошибка через `60 s`;
   - final displacement error;
   - mean trajectory error.
5. Обновить `simulation.html`:
   - синяя линия = GPS;
   - красная линия = predicted;
   - текущие точки обеих траекторий;
   - график ошибки во времени.

### Критерий готовности

- По каждому test-полету видно, где модель уходит от GPS.
- Можно сравнить drift разных моделей на одной таблице.

## Фаза 8. Анализ признаков и гипотез

Цель: ответить, какие данные полезны.

### Наборы признаков

1. `imu_only`
   - акселерометр, гироскоп.

2. `imu_baro`
   - IMU + барометр/высота.

3. `imu_flow`
   - IMU + optical flow.

4. `imu_motors`
   - IMU + моторные признаки.

5. `all`
   - все доступные признаки.

### Гипотезы

1. Моторные признаки уменьшают ошибку на маневрах.
2. Optical flow слабый в текущем виде из-за малого диапазона или плохой синхронизации.
3. Ориентация (`Roll/Pitch/Yaw`) сильно улучшает переносимость, но надо решить, разрешено ли использовать оценку автопилота.
4. Body-frame target может быть стабильнее global ENU target для обучения.

### Критерий готовности

- Есть ablation-таблица по наборам признаков.
- Есть вывод, какие признаки оставить для финальной модели.

## Фаза 9. Финальный отчет

Цель: собрать исследование в понятный документ.

### Структура

1. Постановка задачи.
2. Описание исходных данных.
3. Разделение GPS как target и sensor features как input.
4. Подготовка координат и окон.
5. Baseline-модели.
6. Нейросетевые модели.
7. Метрики `dx/dy/dz`.
8. Накопленная траектория и drift.
9. Анализ признаков.
10. Ограничения.
11. Что нужно для улучшения.

### Артефакты

- `reports/final_report.md`;
- таблицы метрик;
- ссылки на replay;
- графики GPS vs predicted;
- список команд для воспроизведения.

## Ближайшие конкретные шаги

1. Создать `src/build_flight_index.py`.
2. Создать `derived/datasets/flight_index.csv/json`.
3. Создать `src/prepare_flight_tracks.py`.
4. Сохранить `track.csv` для каждого полета.
5. Создать базовый формат `pred.csv`.
6. Добавить в replay поддержку второй траектории `GPS vs predicted`.
7. Перенести текущий ridge baseline на новый формат `flight_id`.
8. Запустить первый честный holdout:
   - train: `module_data` segments 1-5;
   - validation: segment 6;
   - test: segment 7.

## Риски

1. Разные источники имеют разные наборы сенсоров.
   - Решение: отдельно вести `module` и `dataflash` эксперименты.

2. GPS появляется не с начала полета.
   - Решение: обучающие окна строить только там, где есть валидный future target.

3. Optical flow может быть плохо масштабирован.
   - Решение: проверить диапазон, производные, интегралы и корреляцию с GPS displacement.

4. Высота из GPS и барометра может иметь разные смещения.
   - Решение: использовать относительную высоту от старта и отдельно сравнить `up_m`.

5. Хорошая ошибка `dx/dy/dz` не гарантирует хорошую траекторию.
   - Решение: обязательный replay и drift-метрики.

## Минимальный успешный результат

Минимально работающий результат считается достигнутым, если:

1. Все полеты индексированы и не смешаны.
2. Есть train/test split по разным полетам.
3. Есть baseline лучше `zero displacement` на holdout хотя бы для одного горизонта.
4. Есть replay `GPS vs predicted`.
5. Есть таблица drift по test-полету.

## Хороший результат

Хороший результат:

1. Sequence-модель лучше ridge/MLP на holdout.
2. Моторные признаки дают измеримое улучшение.
3. Ошибка накопленной траектории объяснена по участкам полета.
4. Есть воспроизводимые команды от raw CSV/log до финального отчета.

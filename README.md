# IMU navigation practice

## Структура данных и Git

Правила хранения данных описаны в `docs/data_layout.md`.

Коротко:

- `artifacts/` - локальные исходные файлы: большие CSV, DataFlash log/bin, POLI_NA.zip, фото; в Git не коммитятся.
- `artifacts/data.csv` - комбинированный файл с несколькими полетами, использовать только через segment-aware скрипты.
- `derived/` и `artifacts/generated/` - воспроизводимые результаты, датасеты, предсказания и HTML-визуализации; в Git не коммитятся.
- В Git храним код, README/docs, requirements, небольшие осмысленные Markdown-отчеты.
- Активные итоговые отчеты лежат в `reports/navigation/` и `reports/final/`, exploratory notebooks в `jupyter/exploration/`, итоговые reproducible notebooks в `jupyter/final/`.

## Что от нас хотят

По переписке задача такая: проверить, можно ли оценивать перемещение БПЛА без GNSS по временным рядам с платы позиционирования.

Доступные источники:

- IMU: акселерометр, гироскоп, магнитометры, барометр, лидар.
- Optical flow: `Xflow`, `Yflow`.
- GPS/GNSS: широта, долгота, высота. Это не вход для навигации без GNSS, а разметка и проверка качества.
- Лог ArduPilot `2025-01-15 16-46-48.log`, где есть сообщения `RCOU`, `MOTB`, `BAT`, `IMU`, `POS`, `ATT` и т.п. Его стоит использовать для гипотез с мощностью/выходами моторов.
- Архив `POLI_NA.zip` с примером уже обученной LSTM: вход `(sequence, 10 features)`, выход `dx, dy, dz`.

Формальная постановка:

```text
dx, dy, dz = f(p(t-window ... t))
```

где `p` - признаки с датчиков и моторов, а `dx/dy/dz` - смещение, полученное из GPS как обучающая цель.

## Реалистичный маршрут реализации

1. Подготовить GPS-разметку:
   - перевести `LatDegrees + LatMinutes/60`, `LonDegrees + LonMinutes/60` в десятичные градусы;
   - перевести координаты в локальные метры East/North относительно первой валидной GPS-точки;
   - посчитать целевые `dx, dy, dz` через фиксированный горизонт, например 1 секунда.

2. Сделать baseline без нейросети:
   - потоково читать большие CSV, не загружая 1+ ГБ целиком;
   - обучить линейную ridge-регрессию;
   - сравнить с нулевым baseline;
   - получить первичную важность признаков по стандартизованным весам.

3. Расширить признаки:
   - нормы `acc/gyro/mag/flow`;
   - производные и скользящие средние;
   - интегралы ускорений/flow на окне;
   - признаки моторов из `RCOU`/`MOTB`: средняя тяга, разброс по моторам, пары/дифференциалы, мощность от батареи.

4. Перейти к временной модели:
   - MLP по агрегатам окна как быстрый вариант;
   - LSTM/GRU/1D-CNN по сырым окнам;
   - сравнить наборы признаков: IMU only, IMU+flow, IMU+flow+motors.

5. Проверять не только `dx/dy/dz`, но и накопленную траекторию:
   - ошибка через 10/30/60 секунд;
   - drift на полном маршруте;
   - отдельная валидация на другом полете.

## Первый эксперимент

Скрипт `src/imu_baseline.py` делает потоковый baseline по CSV без `pandas/scikit-learn`.

Пример запуска:

```bash
python3 src/imu_baseline.py \
  --csv artifacts/linear_15_01_2025.csv artifacts/triangle_15_01_2025.csv \
  --test-file artifacts/triangle_15_01_2025.csv \
  --sample-ms 100 \
  --horizon-ms 1000 \
  --report reports/baseline_report.md
```

Что получится:

- модель учится на одном полете и проверяется на другом;
- цель - смещение GPS в метрах через 1 секунду;
- в отчете будут MAE/RMSE и топ признаков по важности.

Текущий результат записан в `reports/baseline_report.md`. Простая ridge-регрессия по текущему сэмплу оказалась хуже нулевого baseline на переносе с линейного маршрута на треугольный, поэтому следующий маршрут реализации - оконные признаки, моторы из DataFlash и LSTM/GRU.

## Экспорт моторов из DataFlash

```bash
python3 src/dataflash_extract.py \
  --log artifacts/"2025-01-15 16-46-48.log" \
  --out-dir derived/dataflash \
  --summary reports/dataflash_summary.md
```

Скрипт экспортирует `RCOU`, `MOTB`, `BAT`, `POS`, `GPS`, `IMU`, `ATT`, `BARO` и создает `derived/dataflash/RCOU_motor_features.csv`.

## Карта полета по GPS

```bash
python3 src/gps_flight_map.py
```

По умолчанию скрипт читает `derived/dataflash/GPS.csv` и пишет производные файлы в `artifacts/generated/gps/flights/GPS/`.

Для отдельного источника:

```bash
python3 src/gps_flight_map.py \
  --gps artifacts/triangle_15_01_2025.csv \
  --name triangle_15_01_2025
```

Для склеенного файла с несколькими полетами:

```bash
python3 src/gps_flight_map.py \
  --gps artifacts/data.csv \
  --name module_data \
  --split-all
```

Результаты:

- `map.html` - интерактивная карта маршрута;
- `simulation.html` - визуальный replay/моделирование записанного полета с выбором скорости 1x/5x/10x/30x/60x;
- `path.svg` - офлайн-график траектории, высоты и скорости;
- `track.geojson` - трек для GIS/картографических инструментов;
- `manifest.json` - привязка результата к исходному файлу.

Инвентаризация исходных GPS-файлов:

```bash
python3 src/gps_flight_inventory.py
```

Текущий отчет: `reports/navigation/gps_flight_inventory.md`. Он показывает, что `artifacts/data.csv` содержит 7 сегментов по сбросам `TimeStamp`, а `linear`, `triangle` и DataFlash GPS/POS выглядят как отдельные непрерывные треки.

Каталог полетов для дальнейшего обучения:

```bash
python3 src/build_flight_index.py
```

Результаты:

- `derived/datasets/flight_index.csv`;
- `derived/datasets/flight_index.json`;
- `reports/navigation/flight_index.md`.

Единые GPS-треки в локальных метрах:

```bash
python3 src/prepare_flight_tracks.py
```

Результаты:

- `derived/datasets/tracks/{flight_id}_track.csv` - `time_s`, `source_time_s`, `lat/lon/alt`, `east/north/up`, скорость и накопленная дистанция;
- `artifacts/generated/gps/flights/index.html` - единая страница выбора и просмотра треков;
- `artifacts/generated/gps/flights/{flight_id}/map.html`;
- `artifacts/generated/gps/flights/{flight_id}/simulation.html`;
- `reports/navigation/flight_tracks.md`.

Важно: GPS-координаты в `track.csv` остаются эталоном/target, а не входными признаками для навигации без GNSS.

Viewer можно пересобрать отдельно:

```bash
python3 src/build_track_viewer.py
```

## Оконные датасеты для обучения

```bash
python3 src/build_window_dataset.py
```

Текущий режим собирает module-only датасеты по 9 полетам: `module_data_s01` ... `module_data_s07`, `linear_15_01_2025`, `triangle_15_01_2025`.

Результаты:

- `derived/datasets/windows_module_h1000_l1000.npz`;
- `derived/datasets/windows_module_h3000_l3000.npz`;
- `derived/datasets/windows_module_h5000_l5000.npz`;
- рядом `*_features.json` и `*_meta.csv`;
- `reports/window_datasets.md`.

Вход `X` содержит только сенсоры модуля и агрегаты окна. GPS/GNSS не входит в признаки и используется только для target `dx_east_m`, `dy_north_m`, `dz_up_m`.

## Baseline по оконным датасетам

```bash
python3 src/run_window_baselines.py
```

Скрипт запускает flight-level holdout split без перемешивания окон между train/test:

- `module_data_holdout`: train `module_data_s01..s05`, validation `module_data_s06`, test `module_data_s07`;
- `route_holdout_triangle`: test `triangle_15_01_2025`;
- `route_holdout_linear`: test `linear_15_01_2025`.

Модели:

- `zero`;
- `train_mean`;
- `ridge` с выбором `alpha` по validation-полету.

Результаты:

- `reports/experiments/module_window_baselines.md`;
- `derived/predictions/module_window_baselines/.../*_pred.csv`.

Текущий вывод: на честных holdout-полетах ridge пока не лучше `zero/train_mean`, значит переносимость признаков и target нужно улучшать до перехода к более сложным моделям.

Диагностика target и отладочные проверки:

```bash
python3 src/analyze_window_targets.py
python3 src/filter_window_datasets.py
python3 src/run_chronological_debug.py
```

Результаты:

- `reports/experiments/module_window_target_diagnostics.md`;
- `reports/window_dataset_filters.md`;
- `reports/experiments/module_window_baselines_trim5.md`;
- `reports/experiments/module_window_chronological_debug.md`.

Проверка показала, что в отдельных треках есть стартовые скачки GPS-высоты, но простое удаление первых 5 секунд не меняет общий вывод: ridge все равно хуже простых baseline. Даже chronological-debug внутри тех же полетов не дал улучшения, поэтому следующий шаг - менять постановку target/признаков, а не просто запускать более сложную модель.

## Наложение IMU/flow-траектории на GPS

```bash
python3 src/build_trajectory_overlay.py
```

Скрипт берет готовые `*_pred.csv`, выбирает неперекрывающиеся окна, накапливает предсказанные `dx/dy/dz` как GNSS-free rollout и сравнивает полученную траекторию с GPS/POS.

Результаты:

- `artifacts/generated/navigation/trajectory_overlay/index.html` - интерактивное наложение GPS и накопленной IMU/flow-траектории;
- `derived/predictions/trajectory_overlay/rollout.csv` - точки rollout и ошибки;
- `reports/navigation/trajectory_overlay.md` - итоговые метрики drift/error.

Если между неперекрывающимися prediction-окнами есть большой разрыв, rollout автоматически разбивается на сегменты, чтобы не изображать отсутствие предсказаний как непрерывную инерциальную навигацию.

## Чистая IMU-навигация от стартовой точки

```bash
python3 src/build_imu_dead_reckoning.py
```

Этот эксперимент берет DataFlash `IMU + ATT + POS`: стартует из первой POS/GPS-точки, поворачивает ускорения IMU из корпуса в локальную ENU-систему по `Roll/Pitch/Yaw`, вычитает гравитацию, оценивает постоянный начальный bias ускорения и дальше дважды интегрирует ускорение без GPS-коррекции.

Результаты:

- `artifacts/generated/navigation/imu_dead_reckoning/index.html` - наложение GPS/POS и чистой IMU-траектории;
- `derived/predictions/imu_dead_reckoning/dataflash_imu_dr.csv` - точки, скорости, ускорения и ошибки;
- `reports/navigation/imu_dead_reckoning.md` - финальная ошибка и интерпретация.

Это намеренно не EKF и не ML-модель: GPS используется только для стартовой точки и проверки ошибки в конце.

## Навигация от стартовой точки по optical flow

```bash
python3 src/build_flow_dead_reckoning.py
```

Скрипт строит open-loop траекторию от первой GPS-точки: обучает преобразование сенсорного окна в локальную GPS-скорость на отдельных source CSV, затем на тестовом полете интегрирует предсказанную скорость без GPS-коррекции. Реальная GPS-траектория обязательно рисуется рядом как эталон.

По умолчанию общий `artifacts/data.csv` исключен, потому что это файл с несколькими полетами. Его можно включить только явно:

```bash
python3 src/build_flow_dead_reckoning.py --include-combined-data
```

Результаты:

- `artifacts/generated/navigation/flow_dead_reckoning/index.html` - наложение реальной GPS-траектории и расчетной flow/IMU-траектории;
- `derived/predictions/flow_dead_reckoning/flow_dr.csv` - точки траекторий, скорости и ошибки;
- `reports/navigation/flow_dead_reckoning.md` - итоговые метрики.

## Проверка POLI_NA

```bash
PYTHONPATH=/tmp/poli_deps python3 src/run_poli_na_rollout.py
```

Скрипт запускает модель из `artifacts/POLI_NA.zip` через ONNX Runtime, накапливает ее выходы `dx/dy/dz` от первой GPS-точки и рисует рядом реальную GPS-траекторию.

Важно: архив POLI_NA документирует только форму входа `(sequence, batch, 10)`, но не порядок и нормализацию 10 каналов. Поэтому скрипт проверяет несколько вероятных raw-пресетов признаков из module CSV.

Результаты:

- `artifacts/generated/navigation/poli_na_rollout/index.html` - наложение реальной GPS и POLI_NA rollout;
- `derived/predictions/poli_na_rollout/poli_na_rollout.csv` - точки rollout;
- `reports/navigation/poli_na_rollout.md` - метрики по пресетам.

Проверка альтернативных target:

```bash
python3 src/run_window_baselines.py \
  --target-mode xy \
  --report reports/experiments/module_window_baselines_xy.md \
  --pred-dir derived/predictions/module_window_baselines_xy_base

python3 src/build_path_relative_datasets.py

python3 src/run_window_baselines.py \
  --datasets derived/datasets/windows_module_h1000_l1000_pathrel.npz derived/datasets/windows_module_h3000_l3000_pathrel.npz derived/datasets/windows_module_h5000_l5000_pathrel.npz \
  --report reports/experiments/module_window_baselines_pathrel.md \
  --pred-dir derived/predictions/module_window_baselines_pathrel_clean
```

Результаты:

- `reports/experiments/module_window_baselines_xy.md`;
- `reports/path_relative_datasets.md`;
- `reports/experiments/module_window_baselines_pathrel.md`.

Вывод: исключение `dz` и переход к `along/cross` сами по себе не решают проблему. Простые `zero/train_mean` остаются сильнее ridge, а path-relative отчет показывает много окон с невалидным направлением движения, то есть в данных много почти неподвижных участков.

Moving-only проверка:

```bash
python3 src/filter_window_datasets.py \
  --suffix move1 \
  --min-time-s 5 \
  --min-horizontal-target-m 1 \
  --report reports/window_dataset_filters_move1.md

python3 src/run_window_baselines.py \
  --datasets derived/datasets/windows_module_h1000_l1000_move1.npz derived/datasets/windows_module_h3000_l3000_move1.npz derived/datasets/windows_module_h5000_l5000_move1.npz \
  --report reports/experiments/module_window_baselines_move1.md \
  --pred-dir derived/predictions/module_window_baselines_move1
```

Результат: на moving-only окнах ridge впервые стал лучшим на `route_holdout_linear`, но не на `route_holdout_triangle` и не на `module_data_holdout`. Значит сенсоры содержат полезный сигнал на движении, но переносимость между маршрутами пока нестабильна.

Дальше проверены двухрежимный baseline и простая нелинейная модель без новых зависимостей:

```bash
python3 src/run_two_stage_baseline.py

python3 src/run_rff_baseline.py
```

Результаты:

- `reports/experiments/module_window_two_stage.md`;
- `reports/experiments/module_window_rff_move1.md`;
- `derived/predictions/module_window_two_stage/.../*_pred.csv`;
- `derived/predictions/module_window_rff_move1/.../*_pred.csv`.

Вывод: идеальный gate `hover/moving` (`oracle_gate_clipped`) улучшает `route_holdout_linear`, но реальный классификатор движения часто ошибается между маршрутами. Нелинейный `rff_ridge` на moving-only тоже не обошел лучший test baseline: на `route_holdout_linear` остается лучше обычный `ridge`, на остальных split-ах сильнее `zero/train_mean`.

## Baseline по DataFlash

Более перспективный эксперимент использует синхронизированные окна из DataFlash: `IMU`, `ATT`, `BARO`, `BAT`, `MOTB`, `RCOU_motor_features`.

```bash
python3 src/dataflash_baseline.py \
  --feature-set all \
  --lookback-ms 5000 \
  --horizon-ms 5000 \
  --ridge-alpha 1000 \
  --report reports/dataflash_baseline_all_h5000_l5000.md
```

Текущий результат на хронологическом split внутри одного лога:

| model | MAE 3D | RMSE 3D |
| --- | ---: | ---: |
| zero displacement | 13.970 m | 22.442 m |
| IMU+ATT+BARO ridge | 11.345 m | 14.417 m |
| IMU+ATT+BARO+motors ridge | 10.041 m | 13.668 m |

Вывод: на горизонте 5 секунд оконная модель уже лучше нулевого baseline, а добавление моторных/питающих признаков улучшает результат относительно IMU+ATT+BARO.

После этого добавлен более честный DataFlash sweep с отдельным validation:

```bash
python3 src/run_dataflash_sweep.py
python3 src/build_dataflash_prediction_viewer.py
```

Результаты:

- `reports/experiments/dataflash_sweep.md`;
- `derived/predictions/dataflash_sweep/.../*_pred.csv`;
- `artifacts/generated/dataflash/predictions/sweep/index.html`.

Новый вывод: при split `60/20/20` и подборе `alpha` на validation устойчивее всего выглядит `imu_att`. Набор `all` с моторными/питающими признаками чувствителен к регуляризации: validation может выбрать слишком слабую регуляризацию, а test при этом ухудшается. Это не отменяет пользу моторов из старого `80/20` baseline, но показывает, что следующий шаг должен проверять устойчивость по временным участкам, а не только одну точку split.

Добавлены rolling validation и sparse rollout:

```bash
python3 src/run_dataflash_rolling_validation.py
python3 src/build_dataflash_prediction_viewer.py \
  --pred-dir derived/predictions/dataflash_rolling_validation \
  --output artifacts/generated/dataflash/predictions/rolling/index.html
python3 src/build_dataflash_rollout.py
```

Результаты:

- `reports/experiments/dataflash_rolling_validation.md`;
- `reports/experiments/dataflash_rollout_summary.md`;
- `artifacts/generated/dataflash/predictions/rolling/index.html`;
- `artifacts/generated/dataflash/rollouts/ridge/index.html`.

Rolling validation подтвердил: по оконным метрикам лучший устойчивый кандидат сейчас `imu_att h5000_l5000 ridge` (`MAE 3D 16.293` против `zero 16.335`, но лучше RMSE/P95). Sparse rollout показал, что этого недостаточно: ridge лучше на folds 1 и 3, но резко проваливается на fold 2. Следующий критерий для моделей - не только ошибка `dx/dy/dz`, но и drift/rollout.

Проверен sequence baseline без новых зависимостей:

```bash
python3 src/run_dataflash_sequence_baseline.py
python3 src/run_dataflash_sequence_baseline.py \
  --fixed-alpha 100 \
  --report reports/experiments/dataflash_sequence_imu_att_h5000_fixed100.md \
  --pred-dir derived/predictions/dataflash_sequence_fixed100
```

Результаты:

- `reports/experiments/dataflash_sequence_imu_att_h5000.md`;
- `reports/experiments/dataflash_sequence_imu_att_h5000_fixed100.md`;
- `artifacts/generated/dataflash/predictions/sequence_fixed100/index.html`;
- `artifacts/generated/dataflash/rollouts/sequence_fixed100/index.html`.

Вывод: sequence с per-fold выбором `alpha` нестабилен. Sequence с фиксированным `alpha=100`, выбранным по mean rolling-validation sensitivity, лучше по локальным метрикам (`MAE 3D 15.316` против `zero 16.251`) и дает лучший mean rollout error (`95.471 m` против `train_mean 97.994 m`), но fold 3 все еще сильно дрейфует.

Добавлена validation-residual bias correction для sequence:

```bash
python3 src/run_dataflash_sequence_baseline.py \
  --fixed-alpha 100 \
  --report reports/experiments/dataflash_sequence_imu_att_h5000_fixed100_bias.md \
  --pred-dir derived/predictions/dataflash_sequence_fixed100_bias
```

Результат: `sequence_ridge_bias_corrected` улучшил локальные метрики до `MAE 3D 14.295`, `RMSE 3D 16.684`, `P95 29.954`. В rollout final error стал `52.818 m`, лучше всех предыдущих вариантов, но mean rollout error `106.143 m` все еще хуже `train_mean`. Следующий слабый участок - fold 2.

Диагностика fold residuals записана в `reports/experiments/dataflash_fold_residuals_sequence_fixed100.md`: fold 3 исправляется за счет east-bias, а fold 2 после такой correction переисправляется по east.

Добавлен tuned shrinkage для bias correction:

```bash
python3 src/run_dataflash_sequence_baseline.py \
  --fixed-alpha 100 \
  --tune-bias-shrink \
  --report reports/experiments/dataflash_sequence_imu_att_h5000_fixed100_shrink.md \
  --pred-dir derived/predictions/dataflash_sequence_fixed100_shrink
```

Это текущий лучший вариант: `sequence_ridge_bias_tuned` получил локальную `MAE 3D 13.990`, rollout final error `52.818 m`, mean rollout error `93.538 m`, max rollout error `187.322 m`. Для fold 2 shrink был выбран `0.5`, для folds 1 и 3 - `1.0`.

Также проверен выбор shrink по validation rollout mean: `reports/experiments/dataflash_sequence_imu_att_h5000_fixed100_shrink_rollout.md`. Он улучшил fold 1, но хуже защитил fold 2; текущим best остается выбор shrink по validation MAE.

Финальная сводка текущего DataFlash-кандидата:

- `reports/final/final_dataflash_report.md`;
- `artifacts/generated/dataflash/final_report/index.html`.

Воспроизводимый one-command pipeline для текущего best DataFlash-кандидата:

```bash
python3 src/run_best_dataflash_pipeline.py
```

Он пересобирает:

- `reports/experiments/dataflash_sequence_imu_att_h5000_fixed100_shrink.md`;
- `artifacts/generated/dataflash/predictions/sequence_fixed100_shrink/index.html`;
- rollout CSV и rollout HTML по baseline/model variants;
- `reports/experiments/dataflash_rollout_summary.md`;
- `reports/final/final_dataflash_report.md`;
- `artifacts/generated/dataflash/final_report/index.html`.

Единая comparison page по GPS/POS, IMU, flow, POLI_NA и best DataFlash rollout:

```bash
python3 src/build_navigation_comparison.py
```

Результаты:

- `artifacts/generated/navigation/comparison/index.html`;
- `reports/navigation/navigation_comparison.md`.

## Что уточнить у преподавателя

- Какие именно моторные признаки считать "мощностью": PWM/`RCOU`, `MOTB.ThrOut`, ток/напряжение из `BAT`, или уже готовые значения из другого файла?
- Нужно ли оценивать смещение в глобальной ENU-системе или в системе координат корпуса дрона?
- Какой горизонт предсказания важнее: 0.1 с, 1 с, 5 с или восстановление полной траектории без GPS?
- Есть ли остальные 7 полетов с координатами, о которых говорится в переписке?

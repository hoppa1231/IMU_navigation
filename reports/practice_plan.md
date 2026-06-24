# Практика: план и первый результат

## Постановка

Нужно исследовать навигацию без GNSS: по IMU, optical flow и признакам моторов оценивать локальное перемещение `dx, dy, dz`. GPS используется как эталон для обучения и проверки.

## Что уже сделано

1. Разобрана переписка и вложения.
2. Найдена примерная архитектура из `POLI_NA.zip`: LSTM, вход `(sequence, 10 features)`, выход `3` значения.
3. Сделан потоковый baseline для больших CSV: `src/imu_baseline.py`.
4. Сделан экспорт ArduPilot DataFlash лога в CSV: `src/dataflash_extract.py`.
5. Получены моторные признаки из `RCOU`: `derived/dataflash/RCOU_motor_features.csv`.
6. Сделан оконный baseline по DataFlash: `src/dataflash_baseline.py`.

## Первый эксперимент

Команда:

```bash
python3 src/imu_baseline.py \
  --csv artifacts/linear_15_01_2025.csv artifacts/triangle_15_01_2025.csv \
  --test-file artifacts/triangle_15_01_2025.csv \
  --sample-ms 100 \
  --horizon-ms 1000 \
  --report reports/baseline_report.md
```

Результат на holdout-полете:

| model | MAE 3D | RMSE 3D |
| --- | ---: | ---: |
| zero displacement | 3.086 m | 5.115 m |
| ridge baseline | 19.404 m | 20.190 m |

Интерпретация: простая линейная модель по текущему сэмплу не переносится с линейного маршрута на треугольный. Это не провал задачи, а нормальный первый baseline: он показывает, что нужна временная модель, нормальная ориентация/система координат и дополнительные признаки, особенно моторы.

## Второй эксперимент: DataFlash с моторами

Команда:

```bash
python3 src/dataflash_baseline.py \
  --feature-set all \
  --lookback-ms 5000 \
  --horizon-ms 5000 \
  --ridge-alpha 1000 \
  --report reports/dataflash_baseline_all_h5000_l5000.md
```

Результат на хронологическом split внутри одного лога:

| model | MAE 3D | RMSE 3D |
| --- | ---: | ---: |
| zero displacement | 13.970 m | 22.442 m |
| IMU+ATT+BARO ridge | 11.345 m | 14.417 m |
| IMU+ATT+BARO+motors ridge | 10.041 m | 13.668 m |

Интерпретация: на горизонте 5 секунд оконные признаки уже дают выигрыш. Добавление `BAT`, `MOTB` и `RCOU_motor_features` улучшает результат относительно `IMU+ATT+BARO`. В топе важности появились yaw/roll, `GyrZ`, `AccX` и моторный дифференциал `motor_diff_c2_c4`, то есть модель действительно использует ориентацию и моторную часть.

## Что делать дальше

1. Синхронизировать `derived/dataflash/*.csv` по `TimeUS` и собрать единый датасет из:
   - `IMU`: `GyrX/GyrY/GyrZ`, `AccX/AccY/AccZ`;
   - `ATT`: `Roll/Pitch/Yaw`;
   - `BARO`;
   - `RCOU_motor_features`;
   - `BAT`, `MOTB`;
   - `GPS/POS` как целевые координаты.

2. Улучшить признаки на окнах:
   - mean/std/min/max/delta;
   - интегралы ускорений;
   - нормы ускорения/угловой скорости;
   - моторная средняя тяга, разброс, дифференциалы пар моторов;
   - optical flow и его производные, если используем CSV с flow.

3. Проверить нелинейные модели:
   - ridge/RandomForest/GradientBoosting как быстрые baseline;
   - MLP по агрегатам окна;
   - LSTM/GRU/1D-CNN по последовательностям.

4. Сравнивать гипотезы:
   - IMU only;
   - IMU + optical flow;
   - IMU + motors;
   - IMU + optical flow + motors.

5. Основная метрика:
   - ошибка `dx/dy/dz` на горизонтах 1/5/10 секунд;
   - ошибка восстановленной траектории без GPS;
   - проверка на полете, которого не было в обучении.

## Вопросы преподавателю

- Есть ли остальные 7 полетов с GPS, о которых говорится в переписке?
- Какую систему координат считать целевой: глобальную ENU или body-frame относительно корпуса?
- Какие каналы `RCOU` соответствуют моторам именно на этом аппарате?
- Можно ли использовать `ATT/XKF` из автопилота как вход, или нужны только сырые датчики?

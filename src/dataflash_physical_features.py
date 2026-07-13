"""Shared physical feature transforms for DataFlash sequence models."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SequenceData(Protocol):
    x: np.ndarray
    feature_names: list[str]


def sequence_cube(dataset: SequenceData, sequence_len: int) -> tuple[np.ndarray, list[str]]:
    per_step = len(dataset.feature_names) // sequence_len
    names = [name.split(".", 1)[1] for name in dataset.feature_names[:per_step]]
    return dataset.x.reshape(len(dataset.x), sequence_len, per_step), names


def rotation_body_to_ned(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    result = np.empty(roll.shape + (3, 3), dtype=np.float64)
    result[..., 0, 0] = cp * cy
    result[..., 0, 1] = sr * sp * cy - cr * sy
    result[..., 0, 2] = cr * sp * cy + sr * sy
    result[..., 1, 0] = cp * sy
    result[..., 1, 1] = sr * sp * sy + cr * cy
    result[..., 1, 2] = cr * sp * sy - sr * cy
    result[..., 2, 0] = -sp
    result[..., 2, 1] = sr * cp
    result[..., 2, 2] = cr * cp
    return result


def physical_feature_cube(
    dataset: SequenceData, sequence_len: int, lookback_s: float
) -> tuple[np.ndarray, list[str]]:
    cube, names = sequence_cube(dataset, sequence_len)
    index = {name: idx for idx, name in enumerate(names)}

    def get(name: str) -> np.ndarray:
        return cube[:, :, index[name]]

    direct_names = [
        "IMU.GyrX", "IMU.GyrY", "IMU.GyrZ", "IMU.AccX", "IMU.AccY", "IMU.AccZ",
        "ATT.Roll", "ATT.Pitch", "ATT.att_sin_yaw", "ATT.att_cos_yaw",
        "BARO.Alt", "BARO.CRt", "BAT.Volt", "BAT.Curr",
        "MOTB.ThrOut", "MOTB.ThLimit", "RCOU_motor_features.motor_mean_norm",
        "RCOU_motor_features.motor_std", "RCOU_motor_features.motor_range",
        "RCOU_motor_features.motor_diff_c1_c3", "RCOU_motor_features.motor_diff_c2_c4",
    ]
    channels = [get(name) for name in direct_names]
    channel_names = list(direct_names)

    gyro = np.stack([get(f"IMU.Gyr{axis}") for axis in "XYZ"], axis=-1)
    acc = np.stack([get(f"IMU.Acc{axis}") for axis in "XYZ"], axis=-1)
    gyro_norm = np.linalg.norm(gyro, axis=-1)
    acc_norm = np.linalg.norm(acc, axis=-1)
    dt = lookback_s / max(sequence_len - 1, 1)
    jerk = np.gradient(acc, dt, axis=1)
    angular_acc = np.gradient(gyro, dt, axis=1)

    roll = np.deg2rad(get("ATT.Roll"))
    pitch = np.deg2rad(get("ATT.Pitch"))
    yaw = np.arctan2(get("ATT.att_sin_yaw"), get("ATT.att_cos_yaw"))
    ned_specific_force = np.einsum("...ij,...j->...i", rotation_body_to_ned(roll, pitch, yaw), acc)
    ned_linear_acc = ned_specific_force.copy()
    ned_linear_acc[..., 2] += 9.80665
    enu_linear_acc = np.stack(
        [ned_linear_acc[..., 1], ned_linear_acc[..., 0], -ned_linear_acc[..., 2]], axis=-1
    )

    thrust = get("RCOU_motor_features.motor_mean_norm")
    climb_rate = get("BARO.CRt")
    derived = {
        "gyro_norm": gyro_norm,
        "acc_norm": acc_norm,
        "jerk_norm": np.linalg.norm(jerk, axis=-1),
        "angular_acc_norm": np.linalg.norm(angular_acc, axis=-1),
        "linear_acc_east": enu_linear_acc[..., 0],
        "linear_acc_north": enu_linear_acc[..., 1],
        "linear_acc_up": enu_linear_acc[..., 2],
        "linear_acc_norm": np.linalg.norm(enu_linear_acc, axis=-1),
        "thrust_x_gyro_norm": thrust * gyro_norm,
        "thrust_x_acc_norm": thrust * acc_norm,
        "thrust_x_climb_rate": thrust * climb_rate,
    }
    channels.extend(derived.values())
    channel_names.extend(derived)
    return np.stack(channels, axis=-1), channel_names


def physical_features(
    dataset: SequenceData, sequence_len: int, lookback_s: float
) -> tuple[np.ndarray, list[str]]:
    cube, channel_names = physical_feature_cube(dataset, sequence_len, lookback_s)
    flat_names = [f"t{step:02d}.{name}" for step in range(sequence_len) for name in channel_names]
    return cube.reshape(len(cube), -1), flat_names

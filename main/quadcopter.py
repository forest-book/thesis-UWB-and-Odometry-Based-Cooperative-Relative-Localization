import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from enum import Enum, auto

class Scenario(Enum):
    CONTINUOUS = auto()
    SUDDEN_TURN = auto()

class UAV:
    """
    各UAVの状態と機能を管理するクラス
    論文 V-A-1節「Configuration」に基づき、UAVのダイナミクスを定義
    """
    def __init__(self, uav_id: int,
                 initial_position: np.ndarray,
                 neighbors: List[int],
                 trajectory_config: Optional[Dict[str, str]] = None):
        self.id: int = uav_id
        self.true_position = np.array(initial_position, dtype=float)
        self.neighbors: List[int] = neighbors
        self.trajectory_config = trajectory_config

        # --- 高速化のための事前処理 ---
        self.compiled_vx = None
        self.compiled_vy = None
        self.eval_context = {
                "np": np,
                "sin": np.sin,
                "cos": np.cos,
                "k": 0.0
            }

        if self.trajectory_config and self.trajectory_config.get('type') == 'formula':
            try:
                # 文字列を「バイトコード」にコンパイルしておく
                # filename='<string>' はエラー表示用、mode='eval' は式評価モード
                self.compiled_vx = compile(self.trajectory_config['vx'], '<string>', 'eval')
                self.compiled_vy = compile(self.trajectory_config['vy'], '<string>', 'eval')
            except Exception as e:
                print(f"Compile Error for UAV {self.id}: {e}")

        # 初期速度の計算
        self.update_velocity(t=0, dt=0)  # dtは初期化時は0でOK

        # 推定値を保持する辞書 {target_id: estimate_vector}
        self.direct_estimates: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.fused_estimates: Dict[str, List[np.ndarray]] = defaultdict(list)

    def update_velocity(self, t: int, dt: float):
        """現在の時刻kに基づいて速度ベクトルを更新"""
        # 注: 添え字のkは離散時間ステップだが，速度式内部のkは実時間であるみたい
        # 速度は [m/s] 単位として解釈し、dt を掛けて位置を更新
        k = t * dt  # 速度式内部のkなので実時間に変換

        if self.compiled_vx and self.compiled_vy:
            self.eval_context["k"] = k
            try:
                vx = eval(self.compiled_vx, {"__builtins__": {}}, self.eval_context)
                vy = eval(self.compiled_vy, {"__builtins__": {}}, self.eval_context)
                self.true_velocity = np.array([vx, vy], dtype=float)
            except Exception as e:
                print(f"Error calculating trajectory for UAV {self.id}: {e}")
                raise
        else:
            self.true_velocity = np.zeros(2)

    def update_state(self, t: int, dt: float, event: Scenario = Scenario.CONTINUOUS):
        """UAVの真の位置と速度を更新する"""
        k = t * dt  # 速度式内部のkなので実時間に変換

        # 速度の更新
        self.update_velocity(t=t, dt=dt)

        # シナリオ2: UAV4の急な機動変更イベント
        if self.id == 4 and event == Scenario.SUDDEN_TURN and 100 <= k < 101:
            self.true_velocity += np.array([5.0, 5.0]) # 外乱を追加

        # 位置の更新: v [m/s] × dt [s] = 変位 [m]
        self.true_position += self.true_velocity * dt

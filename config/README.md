# 設定ファイルディレクトリ

このディレクトリにはシミュレーションの設定ファイルが格納されています。

## ファイル一覧

### デフォルト設定

- **simulation_config.yaml**: 論文準拠のスター型トポロジによるデフォルト設定（フィルタ有効）

### グラフトポロジ × 軌道の組み合わせ設定

ファイル名は `graph_[トポロジ]_traj_[軌道].yaml` の形式です。

| ファイル名 | グラフ | 軌道 |
|-----------|--------|------|
| graph_direct_traj_baseline.yaml | direct | baseline |
| graph_direct_traj_linear.yaml | direct | linear |
| graph_direct_traj_sync.yaml | direct | sync |
| graph_direct_traj_highturn.yaml | direct | highturn |
| graph_direct_traj_suddenturn.yaml | direct | suddenturn |
| graph_chain_traj_baseline.yaml | chain | baseline |
| graph_chain_traj_linear.yaml | chain | linear |
| graph_chain_traj_sync.yaml | chain | sync |
| graph_chain_traj_highturn.yaml | chain | highturn |
| graph_chain_traj_suddenturn.yaml | chain | suddenturn |
| graph_ring_traj_baseline.yaml | ring | baseline |
| graph_ring_traj_linear.yaml | ring | linear |
| graph_ring_traj_sync.yaml | ring | sync |
| graph_ring_traj_highturn.yaml | ring | highturn |
| graph_ring_traj_suddenturn.yaml | ring | suddenturn |
| graph_dense_traj_baseline.yaml | dense | baseline |
| graph_dense_traj_linear.yaml | dense | linear |
| graph_dense_traj_sync.yaml | dense | sync |
| graph_dense_traj_highturn.yaml | dense | highturn |
| graph_dense_traj_suddenturn.yaml | dense | suddenturn |

## ファイル形式

YAML形式のみサポートしています。読み込みには `pyyaml` パッケージが必要です。

```bash
pip install pyyaml
```

## パラメータ説明

### 基本設定

| パラメータ | 型 | 説明 | 推奨値 |
|-----------|-----|------|--------|
| DURATION | float | シミュレーション時間 [秒] | 300 |
| T | float | サンプリング周期 [秒] | 0.02〜0.05 |
| GAMMA | float | 推定ゲイン γ | 0.01〜0.1 |
| TARGET_ID | int | 推定目標UAVのID | 1 |

### シナリオ設定

| 値 | 内容 |
|----|------|
| CONTINUOUS | 通常の連続飛行 |
| SUDDEN_TURN | UAV4が t=100s 時点で急機動（外乱 +5.0, +5.0 m/s を付加） |

### UAV配置

- **INITIAL_POSITIONS**: 各UAVの初期位置 `{id: [x, y]}` [m]
- **NEIGHBORS**: 各UAVの隣接機IDリスト `{id: [neighbor_id, ...]}`

UAV1 は推定対象のため、NEIGHBORS は常に空リスト `[]` にします。

### 軌道設定

各UAVの速度を数式で指定します。数式内では `np`（NumPy）と `k`（実時間 [s]）が使用できます。

```yaml
UAV_TRAJECTORIES:
  1:
    type: "formula"
    vx: "np.cos(k / 3)"
    vy: "-5/3 * np.sin(k / 3)"
```

### ノイズ設定

| パラメータ | 説明 |
|-----------|------|
| enabled | ノイズの有効/無効 |
| delta_bar | 速度ノイズの境界値 [m/s]（ガウス分布の 3σ 相当） |
| dist_bound | 距離ノイズの境界値 [m]（ガウス分布の 3σ 相当） |
| initialization_bound | 初期推定値の一様ノイズ境界 [m] |

### フィルタ設定

指数移動平均（EMA）フィルタの設定です。

| パラメータ | 説明 |
|-----------|------|
| enabled | フィルタの有効/無効 |
| alpha | EMA係数（0〜1）。大きいほど応答速度が速く、小さいほど平滑化が強い |

## グラフトポロジの説明

### direct（スター型・直接のみ）

全UAVがUAV1のみを隣接機とするトポロジです。全UAVが直接推定のみを行い、間接推定は発生しません。

```
UAV1（ターゲット）
├── UAV2
├── UAV3
├── UAV4
├── UAV5
└── UAV6
```

### chain（チェーン型）

UAV2→3→4→5→6 の一方向チェーンに加え、UAV2のみUAV1に直接接続します。

```
UAV1 - UAV2 - UAV3 - UAV4 - UAV5 - UAV6
```

### ring（リング型）

UAV2〜6 が環状に接続し、UAV2のみUAV1に直接接続します。

```
UAV1 - UAV2 - UAV3 - UAV4 - UAV5 - UAV6 - (UAV2に戻る)
```

### dense（密結合型）

各UAVが3機の隣接機を持ち、全UAVがUAV1に直接接続します。`simulation_config.yaml` の論文準拠トポロジよりも接続が密になっています。

```yaml
NEIGHBORS:
  2: [1, 3, 4]
  3: [1, 2, 5]
  4: [1, 2, 5]
  5: [1, 3, 6]
  6: [1, 4, 5]
```

## 軌道の説明

### baseline（論文準拠の各UAV異なる軌道）

各UAVが異なる速度式を持つ、論文 Section V-A に記載の設定です。

| UAV | 速度式 | 軌跡の特徴 |
|-----|--------|-----------|
| 1 | `cos(k/3), -5/3 sin(k/3)` | 楕円軌道 |
| 2 | `-2 sin(k), 2 cos(k)` | 円軌道 |
| 3 | `cos(k/5)-sin(k/5)cos(k), sin(k/5)+cos(k/5)cos(k)` | 複合軌道 |
| 4 | `-3 sin(k), 3 cos(k)` | 円軌道（大） |
| 5 | `1/6, 0` | 一定速度の直線 |
| 6 | `-10/3 sin(k/3), 5/3 cos(k/3)` | 楕円軌道 |

### linear（直線移動）

UAV2〜6 が全て一定速度 `vx=1/6, vy=0` で直線移動します。

### sync（同期楕円軌道）

UAV2〜6 が全て同一の楕円軌道 `vx=-10/3 sin(k/3), vy=5/3 cos(k/3)` を描きます。

### highturn（高旋回率軌道）

UAV2〜6 が全て複合軌道 `vx=cos(k/5)-sin(k/5)cos(k), vy=sin(k/5)+cos(k/5)cos(k)` を描きます。baseline よりも旋回が激しい条件です。

### suddenturn（急機動あり）

`EVENT: SUDDEN_TURN` を設定し、t=100s にUAV4に外乱を付加します。軌道自体は baseline と同じです。

## 使用方法

```python
from config_loader import ConfigLoader

# YAML形式で読み込み
params = ConfigLoader.load('../config/simulation_config.yaml')

# 自動判定（拡張子で判別）
params = ConfigLoader.load('../config/graph_ring_traj_baseline.yaml')
```

## カスタム設定の作成

既存の設定ファイルをコピーして編集することで、異なる実験条件を作成できます。

```bash
cp simulation_config.yaml my_experiment.yaml
# my_experiment.yaml を編集
```

```python
# run_single.py でカスタム設定を指定
config_path = PathProvider.get_config_filepath("my_experiment.yaml")
```

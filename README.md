# UWB and Odometry-Based Cooperative Relative Localization Simulation

このリポジトリは，IEEE論文 "Ultra-Wideband and Odometry-Based Cooperative Relative Localization With Application to Multi-UAV Formation Control" (Guo et al., 2020) のシミュレーションを追試実装したものです．

## 📚 論文概要

本実装は，論文の **Section V-A: Cooperative RL Simulation Results** に基づいており，以下の技術を再現しています：

- **直接相対位置推定 (Direct RL Estimation)**: 式(1)に基づく，UWB測距とオドメトリを用いた相対位置推定
- **融合相対位置推定 (Fused RL Estimation)**: 式(5)に基づく，コンセンサスベースの融合推定手法
- **マルチUAVシナリオ**: 6機のUAVによる協調的な相対位置推定

### 主要な特徴

✅ インフラストラクチャフリー（GPS不要）\
✅ UWB測距とオドメトリのみを使用 \
✅ ガウスノイズモデルによるリアルな環境 \
✅ 直接推定と間接推定の融合による高精度化 \
✅ 指数移動平均フィルタによるノイズ低減 \
✅ YAML設定ファイルによる柔軟なパラメータ管理 \

## 🗂️ プロジェクト構成

```
thesis-UWB-and-Odometry-Based-Cooperative-Relative-Localization/
├── config/                     # 設定ファイル（YAML/JSON）
│   ├── simulation_config.yaml # デフォルト設定
│   ├── simulation_config.json # JSON形式の設定
│   ├── star_config.yaml       # スター型トポロジ設定
│   ├── test_config.yaml       # テスト用設定
│   └── README.md              # 設定ファイルの説明
│
├── main/                       # メインプロジェクトディレクトリ
│   ├── run_single.py          # 単一設定でのシミュレーション実行
│   ├── run_batch.py           # 複数設定の一括実行
│   ├── controller.py          # メインコントローラー（シミュレーション制御）
│   ├── quadcopter.py          # UAVクラス定義
│   ├── estimator.py           # RL推定アルゴリズム実装
│   ├── data_logger.py         # データロギング機能
│   ├── plotter.py             # グラフ生成機能
│   ├── measurement_filter.py  # 測定値フィルタ（EMA）
│   ├── config_loader.py       # 設定ファイルローダー
│   └── path_provider.py       # パス管理ユーティリティ
│
├── sandbox/                    # 実験的なコードと単体テスト
│   ├── test_directRL.py       # 直接推定の検証
│   ├── test_fusedRL.py        # 融合推定の検証
│   ├── test_error.py          # エラー解析
│   └── simulation.py          # 旧バージョンのシミュレーション
│
├── data/                       # 実行結果の保存先（自動生成）
│   └── [設定名]_[タイムスタンプ]/
│       ├── csv/               # CSV形式の生データ
│       │   ├── trajectories/  # UAV軌跡データ
│       │   └── RL_errors/     # 推定誤差データ
│       ├── graph/             # 生成されたグラフ
│       │   ├── trajectories/  # 軌跡プロット
│       │   └── RL_errors/     # 誤差プロット
│       ├── statistics/        # 統計情報
│       │   ├── json/          # JSON形式の統計
│       │   └── txt/           # テキスト形式の統計
│       └── *.yaml             # 使用した設定ファイルのコピー
│
├── requirements.txt           # 依存パッケージ
├── .gitignore                 # Git除外設定
├── .editorconfig              # エディタ設定
└── README.md                  # このファイル
```

## 🚀 セットアップ

### 必要要件

- Python 3.8以上（Python 3.13.3で動作確認済み）
- pip

### インストール

1. リポジトリのクローン
```bash
git clone https://github.com/<USERNAME>/thesis-UWB-and-Odometry-Based-Cooperative-Relative-Localization.git
cd thesis-UWB-and-Odometry-Based-Cooperative-Relative-Localization
```

2. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

必要なパッケージ：
- numpy: 数値計算
- matplotlib: グラフ描画
- pandas: データ処理
- pyyaml: YAML設定ファイルの読み込み（オプション）

## 🎮 実行方法

### 基本的な実行（単一設定）

```bash
cd main
python run_single.py
```

デフォルトでは `config/simulation_config.yaml` が使用されます。

### 一括実行（複数設定）

```bash
cd main
python run_batch.py
```

`config/` ディレクトリ内の全ての `.yaml` ファイルで自動的にシミュレーションを実行します。

### 設定ファイルの選択

`run_single.py` を編集して任意の設定ファイルを指定できます：

```python
# run_single.py の該当行を変更
config_path = PathProvider.get_config_filepath("star_config.yaml")
```

## ⚙️ 設定パラメータ

設定は `config/` ディレクトリ内のYAMLまたはJSONファイルで管理します。

### 主要パラメータ

```yaml
# シミュレーション時間設定
DURATION: 300          # シミュレーション時間 [秒]
T: 0.02               # サンプリング周期 [秒] (推奨: 0.02)

# 推定アルゴリズム設定
GAMMA: 0.06           # 推定ゲイン γ (0.01～0.1を推奨)
TARGET_ID: 1          # 推定対象のUAV ID

# シナリオ設定
EVENT: CONTINUOUS     # CONTINUOUS | SUDDEN_TURN

# UAV初期位置 [x, y] (m)
INITIAL_POSITIONS:
  1: [0, 0]
  2: [2, -30]
  3: [20, -15]
  4: [-20, 8]
  5: [-14, 8]
  6: [-10, -30]

# 隣接関係設定（センシンググラフ）
NEIGHBORS:
  1: []              # UAV1は推定対象のため隣接機なし
  2: [1]             # UAV2の隣接機
  3: [1, 4, 5]       # UAV3の隣接機
  4: [1]             # UAV4の隣接機
  5: [3, 4]          # UAV5の隣接機
  6: [4]             # UAV6の隣接機

# UAV軌道設定（速度式）
UAV_TRAJECTORIES:
  1:
    type: "formula"
    vx: "np.cos(k / 3)"
    vy: "-5/3 * np.sin(k / 3)"
  # ... 他のUAVも同様

# ノイズパラメータ
NOISE:
  enabled: true                  # ノイズの有効/無効
  delta_bar: 0.5                # 速度ノイズ境界 [m/s]
  dist_bound: 0.05              # 距離測定ノイズ境界 [m]
  initialization_bound: 25      # 初期推定ノイズ境界 [m]

# フィルタ設定
FILTER:
  enabled: true     # フィルタの有効/無効
  alpha: 0.8        # EMA係数 (0～1, 大きいほど応答速度が速い)
```

### シナリオの説明

- **CONTINUOUS**: 連続的な軌道（論文 Fig.4(a)相当）
- **SUDDEN_TURN**: UAV4が100秒時点で急機動（論文 Fig.4(d)相当）

### パラメータチューニングのヒント

- **GAMMA（推定ゲイン）**:
  - 小さい値（0.01～0.03）: 安定だが収束が遅い
  - 大きい値（0.06～0.1）: 収束が速いが振動しやすい

- **T（サンプリング周期）**:
  - 推奨値: 0.02～0.05秒
  - 小さすぎると計算負荷増、大きすぎると精度低下

- **FILTER alpha（フィルタ係数）**:
  - 小さい値（0.2～0.5）: ノイズ除去効果大、応答遅い
  - 大きい値（0.7～0.9）: 応答速い、ノイズ除去効果小

## 📊 出力結果

実行後、`data/[設定名]_[タイムスタンプ]/` ディレクトリに以下のファイルが生成されます：

### 1. CSVファイル（`csv/`）

#### 軌跡データ（`trajectories/`）
- `uav_trajectories_YYYY-MM-DD-HH-MM-SS.csv`
  - 全UAVの位置データ
  - 列: time, uav1_true_pos_x, ..., uav6_true_pos_x, uav1_true_pos_y, ..., uav6_true_pos_y

#### 誤差データ（`RL_errors/`）
- `fused_RL_error_YYYY-MM-DD-HH-MM-SS.csv`
  - UAV 2～6からUAV 1への推定誤差
  - 列: time, uav2_fused_error, ..., uav6_fused_error

### 2. グラフ（`graph/`）

#### 軌跡プロット（`trajectories/`）
- `uav_trajectories_graph_*.png`
  - 全UAVの飛行軌跡を2次元プロット
  - 開始点（○）と終了点（×）をマーク
  - 論文 Fig.4(a)/(d) に相当

#### 誤差プロット（`RL_errors/`）
- `fused_RL_errors_graph_*.png`
  - 各UAVの推定誤差の時間変化
  - ズームインセット図付き（98～110秒）
  - 論文 Fig.4(b)/(e) に相当

### 3. 統計情報（`statistics/`）

#### JSON形式（`json/`）
- `fused_RL_error_statistics_*.json`
  - 機械可読形式の統計データ
  - 平均誤差、分散、標準偏差、サンプル数

#### テキスト形式（`txt/`）
- `fused_RL_error_statistics_*.txt`
  - 人間が読みやすい形式の統計
  - 論文 Table I に相当

### コンソール出力例

```
initialize simulation settings...
simulation progress: 10%
simulation progress: 20%
...
simulation progress: 100%

======================================================================
  融合RL推定誤差の統計 (120秒後から安定状態)
======================================================================
UAV Pair   | Mean Error (m)     | Variance        | Std Dev (m)
----------------------------------------------------------------------
 2→1       | 0.234567          | 0.012345        | 0.111111
 3→1       | 0.345678          | 0.023456        | 0.153139
 4→1       | 0.456789          | 0.034567        | 0.185925
 5→1       | 0.567890          | 0.045678        | 0.213732
 6→1       | 0.678901          | 0.056789        | 0.238306
======================================================================
```

## 🧪 単体テスト

個別のアルゴリズムを検証するためのテストコードが `sandbox/` に用意されています：

```bash
cd sandbox

# 直接推定（式1）のテスト
python test_directRL.py
# → 単一ステップテストと収束性テストを実行
# → グラフ 'equation_1_convergence_test.png' を生成

# 融合推定（式5）のテスト
python test_fusedRL.py
# → 融合推定の計算例を出力

# エラー解析（旧実装）
python test_error.py
# → 初期化戦略の検証
```

## 🔬 実装の詳細

### アーキテクチャ

本実装は**Model-View-Controller (MVC)** パターンに基づいています：

- **Model**: `UAV`, `Estimator` - UAVの状態と推定アルゴリズム
- **View**: `Plotter` - 結果の可視化
- **Controller**: `MainController` - シミュレーション制御とデータフロー管理

### 主要クラス

#### `UAV` (quadcopter.py)
```python
class UAV:
    def __init__(self, uav_id, initial_position, neighbors, trajectory_config)
    def update_velocity(self, t, dt)
    def update_state(self, t, dt, event)
```
- UAVの状態（位置，速度）を管理
- 論文の速度式に基づく運動モデル
- 直接推定値と融合推定値を保持
- **高速化**: 速度式をバイトコードにプリコンパイル

#### `Estimator` (estimator.py)
```python
class Estimator:
    def calc_direct_RL_estimate(chi_hat, noisy_v, noisy_d, noisy_d_dot, T, gamma)
    def calc_fused_RL_estimate(pi, direct_est, indirect_ests, noisy_v, T, kappa_D, kappa_I)
    def calc_estimation_kappa(uav_neighbors, target_id)
```
- **`calc_direct_RL_estimate()`**: 式(1)の実装
- **`calc_fused_RL_estimate()`**: 式(5)の実装
- **`calc_estimation_kappa()`**: 重みκの計算

#### `MainController` (controller.py)
```python
class MainController:
    def __init__(self, params, save_dir, is_result_show)
    def initialize()
    def run()
    def build_measurements_cache()
    def exec_direct_estimation(measurements_cache, loop)
    def exec_fused_estimation(measurements_cache, loop)
```
- シミュレーション全体の制御
- **最適化**: 測定値の事前計算とキャッシュ
- メインループの実行
- ノイズ生成とフィルタリング

#### `MeasurementFilter` (measurement_filter.py)
```python
class MeasurementFilter:
    def __init__(self, alpha)
    def apply(key, measured_v, measured_d, measured_d_dot)
    def reset()
```
- 指数移動平均（EMA）フィルタの実装
- UAVペアごとに状態を保持
- ノイズの影響を軽減

#### `DataLogger` (data_logger.py)
```python
class DataLogger:
    def logging_timestamp(time)
    def logging_uav_trajectories(uav_id, uav_position)
    def logging_fused_RL_error(uav_id, error)
    def save_UAV_trajectories_data_to_csv()
    def save_fused_RL_errors_to_csv()
    def calc_fused_RL_error_statistics(transient_time)
    def print_fused_RL_error_statistics(transient_time)
    def save_fused_RL_error_statistics(transient_time, file_format)
```
- データのロギングとCSV出力
- 統計情報の計算と表示

#### `Plotter` (plotter.py)
```python
class Plotter:
    @staticmethod
    def plot_UAV_trajectories_from_csv(csv_path, save_dir, save_filename, is_result_show)
    @staticmethod
    def plot_fused_RL_errors_from_csv(csv_path, save_dir, save_filename, is_result_show)
```
- CSVからのグラフ生成
- 論文形式のプロット（ズームインセット付き）

#### `ConfigLoader` (config_loader.py)
```python
class ConfigLoader:
    @staticmethod
    def load_from_json(filepath)
    @staticmethod
    def load_from_yaml(filepath)
    @staticmethod
    def load(filepath)
```
- JSON/YAML設定ファイルの読み込み
- 自動フォーマット判定

### アルゴリズムの詳細

#### 直接相対位置推定（式1）

```
χ̂ᵢⱼᵢ,ₖ₊₁ = χ̂ᵢⱼᵢ,ₖ + T(νᵢⱼᵢ,ₖ + εₖ)
          + γT(νᵢⱼᵢ,ₖ + εₖ)[(dᵢⱼₖ + εᵈₖ)(ḋᵢⱼₖ + εᵈ̇ₖ)
          - (νᵢⱼᵢ,ₖ + εₖ)ᵀχ̂ᵢⱼᵢ,ₖ]
```

**各項の意味：**
- 第1項: 現在の推定値
- 第2項: 相対速度による予測項（オドメトリ）
- 第3項: UWB測定による補正項

**実装のポイント：**
- UWB測距 `dᵢⱼₖ` と距離変化率 `ḋᵢⱼₖ` を利用
- イノベーション = UWB測定値 - 推定予測値
- ゲイン `γ` で補正の強さを調整

#### 融合相対位置推定（式5）

```
πᵢⱼᵢ,ₖ₊₁ = πᵢⱼᵢ,ₖ + T(νᵢⱼᵢ,ₖ + εₖ)
          + κᴰᵢⱼ[χ̂ᵢⱼᵢ,ₖ - πᵢⱼᵢ,ₖ]
          + Σᵣ∈Nᵢ\{j} κᴵᵢᵣ[χ̂ᵢⱼᵣ,ₖ - πᵢⱼᵢ,ₖ]
```

**各項の意味：**
- 第1項: 現在の融合推定値
- 第2項: 相対速度による予測項
- 第3項: 直接推定による補正項
- 第4項: 間接推定による補正項（総和）

**重みの計算：**
```
κᴰᵢⱼ = αᵢⱼ / (|Nᵢ| + 1 + αᵢⱼ)
κᴵᵢᵣ = 1 / (|Nᵢ| + 1 + αᵢⱼ)

ここで、αᵢⱼ = 1 (jがiの隣接機の場合), 0 (それ以外)
```

#### ノイズモデル

**ガウスノイズ（改良版）:**
```python
# 元の一様分布の境界値を3σに対応させる
σ_v = delta_bar / 6.0           # 速度ノイズの標準偏差
σ_d = dist_bound / 6.0          # 距離ノイズの標準偏差

vel_noise = np.random.normal(0, σ_v, size=2)
dist_noise = np.random.normal(0, σ_d)
dist_rate_noise = np.random.normal(0, σ_d)
```

**特徴：**
- 99.7%の値が±3σ（元の境界値）以内に収まる
- 時折真値に近い測定も得られる（一様分布より現実的）

#### 指数移動平均フィルタ

```python
filtered_k = α × raw_k + (1-α) × filtered_{k-1}
```

**パラメータ α の効果：**
- α = 0.2: 強い平滑化、応答遅い
- α = 0.8: 弱い平滑化、応答速い（デフォルト）

## 📈 論文との対応

| 論文の要素 | 実装ファイル | 関数/クラス |
|-----------|------------|-----------|
| 式(1) 直接推定 | `estimator.py` | `calc_direct_RL_estimate()` |
| 式(5) 融合推定 | `estimator.py` | `calc_fused_RL_estimate()` |
| 定理1 推定誤差境界 | `estimator.py` | パラメータ検証ロジック |
| 速度式（V-A節） | `quadcopter.py` | `update_velocity()` |
| ノイズモデル（4.1節） | `controller.py` | `get_noisy_measurements()` |
| Fig.4(a)/(d) 軌跡 | `plotter.py` | `plot_UAV_trajectories_from_csv()` |
| Fig.4(b)/(e) 誤差 | `plotter.py` | `plot_fused_RL_errors_from_csv()` |
| Table I 統計 | `data_logger.py` | `calc_fused_RL_error_statistics()` |

## 🔧 トラブルシューティング

### 推定が発散する

**原因と対策：**
1. **GAMMAが大きすぎる**
   - 解決策: 0.01～0.06の範囲に設定

2. **サンプリング周期Tが大きすぎる**
   - 解決策: T < 0.05秒に設定

3. **初期推定値が真値から遠すぎる**
   - 解決策: `initialization_bound` を25m以下に設定

### フィルタ効果が見られない

**原因と対策：**
1. **alpha係数が大きすぎる（0.9以上）**
   - 解決策: alpha = 0.5～0.8に調整

2. **フィルタが無効になっている**
   - 解決策: `FILTER.enabled: true` を確認

### グラフが表示されない

**原因と対策：**
1. **バッチ実行モード**
   - `run_batch.py` では `is_result_show=False` がデフォルト
   - 解決策: `run_single.py` を使用するか、コード内で変更

2. **matplotlibのバックエンド問題**
   - 解決策: 環境変数 `MPLBACKEND=TkAgg` を設定

## 🤝 貢献

バグ報告や改善提案は Issue または Pull Request でお願いします。

### 開発ガイドライン

- コードスタイル: `.editorconfig` に従う
- PRテンプレート: `.github/pull_request_template.md` を使用
- コミットメッセージ: 日本語で簡潔に

## 📄 ライセンス

本プロジェクトは学術研究目的で作成されています。商用利用の際は原論文の著者に確認してください。

## 📄 参考文献

```bibtex
@article{guo2020uwb,
  title={Ultra-Wideband and Odometry-Based Cooperative Relative Localization With Application to Multi-UAV Formation Control},
  author={Guo, Kexin and Li, Xiuxian and Xie, Lihua},
  journal={IEEE Transactions on Cybernetics},
  volume={50},
  number={6},
  pages={2590--2603},
  year={2020},
  publisher={IEEE},
  doi={10.1109/TCYB.2019.2905931}
}
```

## 📞 お問い合わせ

質問や提案がある場合は、GitHubのIssueでお知らせください。

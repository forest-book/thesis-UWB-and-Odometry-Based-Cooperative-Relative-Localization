# UWB and Odometry-Based Cooperative Relative Localization Simulation

このリポジトリは，IEEE論文 "Ultra-Wideband and Odometry-Based Cooperative Relative Localization With Application to Multi-UAV Formation Control" (Guo et al., 2020) のシミュレーションを追試実装したものです．

## 📚 論文概要

本実装は，論文の **Section V-A: Cooperative RL Simulation Results** に基づいており，以下の技術を再現しています：

- **直接相対位置推定 (Direct RL Estimation)**: 式(1)に基づく，UWB測距とオドメトリを用いた相対位置推定
- **融合相対位置推定 (Fused RL Estimation)**: 式(5)に基づく，コンセンサスベースの融合推定手法
- **マルチUAVシナリオ**: 6機のUAVによる協調的な相対位置推定

### 主要な特徴

✅ インフラストラクチャフリー（GPS不要）\
✅ UWB測距とオドメトリのみを使用\
✅ ガウスノイズモデルによるリアルな環境\
✅ 直接推定と間接推定の融合による高精度化\
✅ 指数移動平均フィルタによるノイズ低減\
✅ YAML設定ファイルによる柔軟なパラメータ管理

## 🗂️ プロジェクト構成

```
thesis-UWB-and-Odometry-Based-Cooperative-Relative-Localization/
├── config/                          # 設定ファイル（YAML）
│   ├── simulation_config.yaml      # デフォルト設定（論文準拠のスター型トポロジ）
│   ├── graph_chain_traj_baseline.yaml   # チェーン型グラフ × ベースライン軌道
│   ├── graph_chain_traj_highturn.yaml   # チェーン型グラフ × 高回転軌道
│   ├── graph_chain_traj_linear.yaml     # チェーン型グラフ × 直線軌道
│   ├── graph_chain_traj_suddenturn.yaml # チェーン型グラフ × 急機動
│   ├── graph_chain_traj_sync.yaml       # チェーン型グラフ × 同期軌道
│   ├── graph_dense_traj_*.yaml          # 密結合グラフ × 各軌道 (5種)
│   ├── graph_direct_traj_*.yaml         # 全直接接続グラフ × 各軌道 (5種)
│   ├── graph_ring_traj_*.yaml           # リング型グラフ × 各軌道 (5種)
│   └── README.md
│
├── config_archive/                  # 過去の実験設定ファイル（参考用）
│
├── main/                            # メインプロジェクトディレクトリ
│   ├── run_single.py               # 単一設定でのシミュレーション実行
│   ├── run_batch.py                # 複数設定の一括実行
│   ├── controller.py               # メインコントローラー（シミュレーション制御）
│   ├── quadcopter.py               # UAVクラス・Scenarioクラス定義
│   ├── estimator.py                # RL推定アルゴリズム実装（式(1)・式(5)）
│   ├── data_logger.py              # データロギング・統計計算・CSV出力
│   ├── plotter.py                  # グラフ生成（軌跡・誤差プロット）
│   ├── measurement_filter.py       # 測定値フィルタ（EMA）
│   ├── config_loader.py            # 設定ファイルローダー（YAML/JSON対応）
│   ├── path_provider.py            # パス管理ユーティリティ
│   └── filesystem_adapter.py       # ファイルシステム操作ユーティリティ
│
├── sandbox/                         # 実験的なコードと単体テスト
│   ├── test_directRL.py            # 直接推定（式1）の検証
│   ├── test_fusedRL.py             # 融合推定（式5）の検証
│   ├── test_error.py               # エラー解析（旧実装）
│   ├── test.py                     # 簡易テスト
│   └── simulation.py               # 旧バージョンのシミュレーション
│
├── data/                            # 実行結果の保存先（自動生成）
│   └── [設定名]_[タイムスタンプ]/
│       ├── csv/
│       │   ├── trajectories/       # UAV軌跡データ（CSV）
│       │   └── RL_errors/          # 推定誤差データ（CSV）
│       ├── graph/
│       │   ├── trajectories/       # 軌跡プロット（PNG・SVG）
│       │   └── RL_errors/          # 誤差プロット（PNG・SVG）
│       ├── statistics/
│       │   ├── json/               # 統計データ（JSON）
│       │   └── txt/                # 統計データ（テキスト）
│       └── *.yaml                  # 使用した設定ファイルのコピー
│
├── requirements.txt
├── .gitignore
├── .editorconfig
└── README.md
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
- pyyaml: YAML設定ファイルの読み込み

## 🎮 実行方法

### 基本的な実行（単一設定）

```bash
cd main
python run_single.py
```

デフォルトでは `config/simulation_config.yaml` が使用されます。別の設定ファイルを使う場合は `run_single.py` 内の `config_path` を変更してください。

```python
# run_single.py の該当行を変更
config_path = PathProvider.get_config_filepath("graph_ring_traj_baseline.yaml")
```

### 一括実行（複数設定）

```bash
cd main
python run_batch.py
```

`config/` ディレクトリ内の全 `.yaml` ファイルでシミュレーションを自動実行し、それぞれ `data/` 以下に結果を保存します。バッチ実行時はグラフの画面表示は行われません（ファイル保存のみ）。

## ⚙️ 設定パラメータ

設定は `config/` ディレクトリ内の YAML ファイルで管理します。

### 主要パラメータ

```yaml
# シミュレーション時間設定
DURATION: 300          # シミュレーション時間 [秒]
T: 0.02               # サンプリング周期 [秒]（推奨: 0.02〜0.05）

# 推定アルゴリズム設定
GAMMA: 0.06           # 推定ゲイン γ（0.01〜0.1を推奨）
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
  1: []
  2: [1]
  3: [1, 4, 5]
  4: [1]
  5: [3, 4]
  6: [4]

# UAV軌道設定（速度式）
UAV_TRAJECTORIES:
  1:
    type: "formula"
    vx: "np.cos(k / 3)"
    vy: "-5/3 * np.sin(k / 3)"
  # ... 他のUAVも同様

# ノイズパラメータ
NOISE:
  enabled: true
  delta_bar: 0.5                # 速度ノイズ境界 [m/s]
  dist_bound: 0.05              # 距離測定ノイズ境界 [m]
  initialization_bound: 25      # 初期推定ノイズ境界 [m]

# フィルタ設定
FILTER:
  enabled: true     # フィルタの有効/無効
  alpha: 0.8        # EMA係数（0〜1、大きいほど応答速度が速い）
```

### シナリオの説明

- **CONTINUOUS**: 連続的な軌道（論文 Fig.4(a) 相当）
- **SUDDEN_TURN**: UAV4 が t=100s 時点で急機動（外乱 +5.0, +5.0 m/s を付加）

### 設定ファイルの命名規則

`config/` 内のファイルは `graph_[トポロジ]_traj_[軌道]` の形式で命名されています。

| トポロジ名 | NEIGHBORS の構造 |
|----------|----------------|
| `direct` | 全UAVがUAV1のみに接続（純粋な直接推定） |
| `chain`  | 1→2→3→4→5→6 の一方向チェーン |
| `ring`   | 2↔3↔4↔5↔6↔2 のリング（UAV1は孤立） |
| `dense`  | 各UAVが3機に接続する密結合グラフ |

| 軌道名 | 内容 |
|--------|------|
| `baseline` | 論文記載の各UAV異なる軌道 |
| `linear`   | UAV2〜6が一定速度の直線移動 |
| `sync`     | UAV2〜6が同一の楕円軌道 |
| `highturn` | UAV2〜6が高旋回率の複合軌道 |
| `suddenturn` | UAV4が t=100s で急機動 |

### パラメータチューニングのヒント

**GAMMA（推定ゲイン）:**
- 小さい値（0.01〜0.03）: 安定だが収束が遅い
- 大きい値（0.06〜0.1）: 収束が速いが振動しやすい

**T（サンプリング周期）:**
- 推奨値: 0.02〜0.05 秒
- 定理1の条件: `0 < T < 1 / (γ(2v̄ + δ̄)²)`（詳細は論文参照）

**FILTER alpha（EMA係数）:**
- 小さい値（0.2〜0.5）: ノイズ除去効果大、応答遅い
- 大きい値（0.7〜0.9）: 応答速い、ノイズ除去効果小（デフォルト: 0.8）

## 📊 出力結果

実行後、`data/[設定名]_[タイムスタンプ]/` に以下のファイルが生成されます。

### CSVファイル

**軌跡データ** (`csv/trajectories/uav_trajectories_*.csv`)

| 列名 | 内容 |
|------|------|
| time | 時刻 [s] |
| uav1_true_pos_x 〜 uav6_true_pos_x | 各UAVのX座標 [m] |
| uav1_true_pos_y 〜 uav6_true_pos_y | 各UAVのY座標 [m] |

**誤差データ** (`csv/RL_errors/fused_RL_error_*.csv`)

| 列名 | 内容 |
|------|------|
| time | 時刻 [s] |
| uav2_fused_error 〜 uav6_fused_error | UAV2〜6からUAV1への推定誤差ノルム [m] |

### グラフ

- **軌跡プロット** (`graph/trajectories/uav_trajectories_graph_*.png/.svg`): 全UAVの飛行軌跡（開始点○・終了点×付き）
- **誤差プロット** (`graph/RL_errors/fused_RL_errors_graph_*.png/.svg`): 推定誤差の時間変化（t=98〜110s のズームインセット付き）

### 統計情報

過渡状態（デフォルト: 120秒）を除いた安定状態における統計をJSON・TXT形式で保存します。コンソールにも以下の形式で出力されます。

```
======================================================================
  融合RL推定誤差の統計 (120秒後から安定状態)
======================================================================
UAV Pair   | Mean Error (m)     | Variance        | Std Dev (m)
----------------------------------------------------------------------
 2→1       | 0.234567          | 0.012345        | 0.111111
 ...
======================================================================
```

## 🔬 実装の詳細

### アーキテクチャ

**Model-View-Controller (MVC)** パターンに基づいて設計されています。

- **Model**: `UAV`（quadcopter.py）、`Estimator`（estimator.py）
- **View**: `Plotter`（plotter.py）
- **Controller**: `MainController`（controller.py）

### 主要クラスの概要

**`UAV`** (quadcopter.py)

UAVの状態（位置・速度）と推定値を管理します。速度式の文字列をバイトコードに事前コンパイルすることで計算を高速化しています。`direct_estimates` と `fused_estimates` の辞書に各ステップの推定値を時系列で蓄積します。

**`Estimator`** (estimator.py)

論文の推定式を実装します。`calc_direct_RL_estimate()` が式(1)、`calc_fused_RL_estimate()` が式(5)に対応します。`calc_estimation_kappa()` では隣接関係から重みκを計算します。

**`MainController`** (controller.py)

シミュレーション全体を制御します。各ステップで全UAVペアの測定値を一括計算してキャッシュし（`build_measurements_cache()`）、直接推定→融合推定→状態更新→ロギングの順で処理を進めます。

**`MeasurementFilter`** (measurement_filter.py)

指数移動平均（EMA）フィルタを実装します。UAVペアごとに前回のフィルタ値を保持し、`apply()` を呼ぶたびに更新します。

**`DataLogger`** (data_logger.py)

シミュレーション中のデータを収集してCSVに保存し、統計計算・表示・保存を行います。

**`ConfigLoader`** (config_loader.py)

YAML / JSON 設定ファイルを読み込み、`EVENT` 文字列を `Scenario` Enum に変換するなどの前処理を行います。

### アルゴリズムの詳細

#### 直接相対位置推定（式1）

```
χ̂ᵢⱼᵢ,ₖ₊₁ = χ̂ᵢⱼᵢ,ₖ + T(νᵢⱼᵢ,ₖ + εₖ)
           + γT(νᵢⱼᵢ,ₖ + εₖ)[(dᵢⱼₖ + εᵈₖ)(ḋᵢⱼₖ + εᵈ̇ₖ)
           - (νᵢⱼᵢ,ₖ + εₖ)ᵀχ̂ᵢⱼᵢ,ₖ]
```

第2項がオドメトリによる予測項、第3項がUWB測定値によるイノベーション補正項です。補正ゲイン γ で補正量を調整します。

#### 融合相対位置推定（式5）

```
πᵢⱼᵢ,ₖ₊₁ = πᵢⱼᵢ,ₖ + T(νᵢⱼᵢ,ₖ + εₖ)
           + κᴰᵢⱼ[χ̂ᵢⱼᵢ,ₖ - πᵢⱼᵢ,ₖ]
           + Σᵣ∈Nᵢ\{j} κᴵᵢᵣ[χ̂ᵢⱼᵣ,ₖ - πᵢⱼᵢ,ₖ]
```

間接推定値 `χ̂ᵢⱼᵣ,ₖ` は「自機→中継機の直接推定」+「中継機→ターゲットの融合推定」で構成されます。重みκは以下の式で計算します。

```
κᴰᵢⱼ = αᵢⱼ / (|Nᵢ| + 1 + αᵢⱼ)   （αᵢⱼ = 1: j が隣接機, 0: それ以外）
κᴵᵢᵣ = 1 / (|Nᵢ| + 1 + αᵢⱼ)
```

#### ノイズモデル

一様分布の境界値を3σに対応させたガウスノイズを採用しています。

```python
σ_v = delta_bar / 6.0    # 速度ノイズの標準偏差
σ_d = dist_bound / 6.0   # 距離ノイズの標準偏差
```

99.7%の値が±3σ（元の境界値）以内に収まり、一様分布より現実的な測定を模擬します。

#### 指数移動平均フィルタ

```
filtered_k = α × raw_k + (1 - α) × filtered_{k-1}
```

`FILTER.enabled: false` の設定では適用をスキップし、生の測定値をそのまま使用します。

### 論文との対応

| 論文の要素 | 実装ファイル | 関数・メソッド |
|-----------|------------|-------------|
| 式(1) 直接推定 | `estimator.py` | `calc_direct_RL_estimate()` |
| 式(5) 融合推定 | `estimator.py` | `calc_fused_RL_estimate()` |
| 重みκの計算 | `estimator.py` | `calc_estimation_kappa()` |
| 速度式（V-A節） | `quadcopter.py` | `update_velocity()` |
| ノイズモデル（4.1節） | `controller.py` | `get_noisy_measurements()` |
| EMAフィルタ | `measurement_filter.py` | `apply()` |
| Fig.4(a)/(d) 軌跡 | `plotter.py` | `plot_UAV_trajectories_from_csv()` |
| Fig.4(b)/(e) 誤差 | `plotter.py` | `plot_fused_RL_errors_from_csv()` |
| Table I 統計 | `data_logger.py` | `calc_fused_RL_error_statistics()` |

## 🔧 トラブルシューティング

**推定が発散する**

- `GAMMA` を 0.01〜0.06 の範囲に下げる
- `T` を 0.05 秒以下に設定する
- `initialization_bound` を 25m 以下に設定する

**フィルタ効果が見られない**

- `FILTER.alpha` を 0.5〜0.8 に調整する
- `FILTER.enabled: true` になっているか確認する

**グラフが表示されない（バッチ実行時）**

`run_batch.py` は `is_result_show=False` で動作します。グラフは `data/` 以下にファイルとして保存されています。画面表示が必要な場合は `run_single.py` を使用してください。

**`pyyaml` が見つからないエラー**

```bash
pip install pyyaml
```

## 🧪 単体テスト（sandbox/）

```bash
cd sandbox

# 直接推定（式1）の検証
python test_directRL.py
# → 単一ステップテストと収束性テストを実行
# → グラフ 'equation_1_convergence_test.png' を生成

# 融合推定（式5）の検証
python test_fusedRL.py
# → 融合推定の計算例を出力
```

## 🤝 貢献

バグ報告や改善提案は Issue または Pull Request でお願いします。

### 開発ガイドライン

- コードスタイル: `.editorconfig` に従う（インデント4スペース、UTF-8、LF改行）
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

質問や提案がある場合は、GitHub の Issue でお知らせください。

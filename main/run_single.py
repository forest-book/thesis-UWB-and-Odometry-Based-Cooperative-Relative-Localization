import os
import sys
import datetime
import shutil

from config_loader import ConfigLoader
from controller import MainController
from path_provider import PathProvider

if __name__ == '__main__':
    # 実行したい設定ファイルを指定
    config_path = PathProvider.get_config_filepath("simulation_config.yaml")
    # または '../config/trajectory_circle.yaml' など

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit()

    print(f"Running single simulation with: {config_path}")

    # --- データ保存ディレクトリの生成 ---
    # configファイル名から拡張子を除いた部分を取得
    config_filename = os.path.splitext(os.path.basename(config_path))[0]
    # 現在の日時を取得
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # 保存ディレクトリ名を生成
    save_dir = PathProvider.get_saved_data_dir_path(config_filename=config_filename, timestamp=timestamp)

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- シミュレーション実行 ---
    # 設定読み込み
    simulation_params = ConfigLoader.load(config_path)

    # コントローラー初期化・実行
    controller = MainController(
        params=simulation_params,
        save_dir=save_dir
    )
    controller.run()

    # --- 使用した設定ファイルのコピー ---
    shutil.copy(config_path, save_dir)
    print(f"Config file copied to {save_dir}")

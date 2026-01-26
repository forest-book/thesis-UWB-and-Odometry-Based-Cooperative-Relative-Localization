import sys
import datetime

from config_loader import ConfigLoader
from controller import MainController
from path_provider import PathProvider
from filesystem_adapter import FileSystemAdapter

if __name__ == '__main__':
    # 実行したい設定ファイルを指定
    config_path = PathProvider.get_config_filepath("simulation_config.yaml")
    # または '../config/trajectory_circle.yaml' など

    if not FileSystemAdapter.file_exists(path=config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit()

    print(f"Running single simulation with: {config_path}")

    # --- データ保存ディレクトリの生成 ---
    # configファイル名から拡張子を除いた部分を取得
    config_filename = FileSystemAdapter.get_filename(file_path=config_path)
    # 現在の日時を取得
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # 保存ディレクトリ名を生成
    save_dir = PathProvider.get_saved_data_dir_path(config_filename=config_filename, timestamp=timestamp)

    # ディレクトリが存在しない場合は作成
    if not FileSystemAdapter.directory_exists(path=save_dir):
        FileSystemAdapter.create_directory(path=save_dir)

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
    FileSystemAdapter.file_copy(source_path=config_path, destination_path=save_dir)
    print(f"Config file copied to {save_dir}")

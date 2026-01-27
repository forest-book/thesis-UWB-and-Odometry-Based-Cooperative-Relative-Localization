import traceback
import sys
import datetime

from config_loader import ConfigLoader
from controller import MainController
from path_provider import PathProvider
from filesystem_adapter import FileSystemAdapter


if __name__ == '__main__':
    # 設定ファイルが格納されているディレクトリ
    config_dir = PathProvider.get_config_dir_path()

    # .yaml ファイルを全て取得
    config_files = FileSystemAdapter.get_files_with_extension(directory=config_dir, extension='*.yaml')

    if not config_files:
        print("No configuration files found in", config_dir)
        sys.exit()

    print(f"Found {len(config_files)} configuration files.")

    for config_file in config_files:
        print(f"\n{'='*20}")
        print(f"Running simulation with config: {FileSystemAdapter.get_filename(config_file)}.yaml")
        print(f"{'='*20}")

        try:
            # --- データ保存ディレクトリの生成 ---
            config_filename = FileSystemAdapter.get_filename(file_path=config_file)
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            save_dir = PathProvider.get_saved_data_dir_path(config_filename=config_filename, timestamp=timestamp)

            if not FileSystemAdapter.directory_exists(path=save_dir):
                FileSystemAdapter.create_directory(path=save_dir)

            # --- シミュレーション実行 ---
            simulation_params = ConfigLoader.load(config_file)
            controller = MainController(
                params=simulation_params,
                save_dir=save_dir,
                is_result_show=False
            )
            controller.run()

            # --- 使用した設定ファイルのコピー ---
            FileSystemAdapter.file_copy(source_path=config_file, destination_path=save_dir)
            print(f"Config file copied to {save_dir}")

        except Exception as e:
            print(f"Error occurred in {config_file}: {e}")
            traceback.print_exc()

    print("\nAll simulations completed.")

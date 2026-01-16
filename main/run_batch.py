import os
import glob

from config_loader import ConfigLoader
from controller import MainController

if __name__ == '__main__':
    # 設定ファイルが格納されているディレクトリ
    config_dir = "../config"
    # .yaml ファイルを全て取得
    config_files = glob.glob(os.path.join(config_dir, '*.yaml'))

    if not config_files:
        print("No configuration files found in", config_dir)
        exit()

    print(f"Found {len(config_files)} configuration files.")

    for config_file in config_files:
        print(f"\n{'='*20}")
        print(f"Running simulation with config: {os.path.basename(config_file)}")
        print(f"{'='*20}")

        try:
            # 設定ファイルから読み込む
            simulation_params = ConfigLoader.load(config_file)

            # コントローラーの初期化と実行
            controller = MainController(simulation_params)
            controller.run()

        except Exception as e:
            print(f"Error occurred in {config_file}: {e}")
            import traceback
            traceback.print_exc()

print("\nAll simulations completed.")

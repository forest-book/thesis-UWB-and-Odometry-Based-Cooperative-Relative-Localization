import os
import glob
import traceback
import sys
import datetime
import shutil

from config_loader import ConfigLoader
from controller import MainController

if __name__ == '__main__':
    # 設定ファイルが格納されているディレクトリ
    config_dir = "../config"
    # .yaml ファイルを全て取得
    config_files = glob.glob(os.path.join(config_dir, '*.yaml'))

    if not config_files:
        print("No configuration files found in", config_dir)
        sys.exit()

    print(f"Found {len(config_files)} configuration files.")

    for config_file in config_files:
        print(f"\n{'='*20}")
        print(f"Running simulation with config: {os.path.basename(config_file)}")
        print(f"{'='*20}")

        try:
            # --- データ保存ディレクトリの生成 ---
            config_filename = os.path.splitext(os.path.basename(config_file))[0]
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            save_dir = f'../data/{config_filename}_{timestamp}'

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # --- シミュレーション実行 ---
            simulation_params = ConfigLoader.load(config_file)
            controller = MainController(
                params=simulation_params,
                save_dir=save_dir,
                is_result_show=False
            )
            controller.run()
            
            # --- 使用した設定ファイルのコピー ---
            shutil.copy(config_file, save_dir)
            print(f"Config file copied to {save_dir}")

        except Exception as e:
            print(f"Error occurred in {config_file}: {e}")
            traceback.print_exc()

    print("\nAll simulations completed.")

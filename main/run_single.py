import os
from config_loader import ConfigLoader
from controller import MainController

if __name__ == '__main__':
    # 実行したい設定ファイルを指定
    config_path = '../config/simulation_config.yaml'
    # または '../config/trajectory_circle.yaml' など

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        exit()

    print(f"Running single simulation with: {config_path}")

    # 設定読み込み
    simulation_params = ConfigLoader.load(config_path)

    # コントローラー初期化・実行
    controller = MainController(simulation_params)
    controller.run()

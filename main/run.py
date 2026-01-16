from config_loader import ConfigLoader
from controller import MainController

if __name__ == '__main__':
    # 設定ファイルから読み込む
    simulation_params = ConfigLoader.load('../config/simulation_config.yaml')

    controller = MainController(simulation_params)
    controller.run()

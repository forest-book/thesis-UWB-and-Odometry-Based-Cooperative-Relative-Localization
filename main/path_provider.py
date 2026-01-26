import os

class PathProvider:
    @staticmethod
    def get_config_filepath(filename: str):
        return f'../config/{filename}'

    @staticmethod
    def get_config_dir_path():
        return "../config"

    @staticmethod
    def get_saved_data_dir_path(config_filename: str, timestamp: str) -> str:
        return f'../data/{config_filename}_{timestamp}'

    @staticmethod
    def get_trajectory_graph_dir_path(save_dir: str) -> str:
        return os.path.join(save_dir, 'graph', 'trajectories')

    @staticmethod
    def get_graph_filepath(graph_dir: str, save_filename) -> str:
        return os.path.join(graph_dir, save_filename)

    @staticmethod
    def get_RL_error_graph_dir_path(save_dir: str) -> str:
        return os.path.join(save_dir, 'graph', 'RL_errors')

    @staticmethod
    def get_statistics_dir_path(save_dir: str, format: str) -> str:
        return os.path.join(save_dir, 'statistics', format)

    @staticmethod
    def get_save_filepath(save_dir: str, save_filename: str) -> str:
        return os.path.join(save_dir, save_filename)

    @staticmethod
    def get_trajectory_csv_dir_path(save_dir: str) -> str:
        return os.path.join(save_dir, 'csv', 'trajectories')

    @staticmethod
    def get_RL_error_csv_dir_path(save_dir: str) -> str:
        return os.path.join(save_dir, 'csv', 'RL_errors')

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


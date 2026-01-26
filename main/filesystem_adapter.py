import os
import shutil

class FileSystem_Adapter:
    """ファイル・フォルダ関連の操作"""
    @staticmethod
    def file_exists(path: str) -> bool:
        return os.path.exists(path)

    @staticmethod
    def file_copy(source_path: str, destination_path) -> None:
        shutil.copy(source_path, destination_path)

    @staticmethod
    def get_filename(file_path: str) -> str:
        """ファイル名から拡張子を除いた部分を取得"""
        return os.path.splitext(os.path.basename(file_path))[0]

    @staticmethod
    def directory_exists(path: str) -> bool:
        return os.path.exists(path)

    @staticmethod
    def create_directory(path: str) -> None:
        os.makedirs(path)

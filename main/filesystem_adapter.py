import os
import shutil
import glob
from typing import List

class FileSystemAdapter:
    """ファイル・フォルダ関連の操作"""
    @staticmethod
    def file_exists(path: str) -> bool:
        return os.path.isfile(path)

    @staticmethod
    def file_copy(source_path: str, destination_path: str) -> None:
        shutil.copy(source_path, destination_path)

    @staticmethod
    def get_filename(file_path: str) -> str:
        """ファイル名から拡張子を除いた部分を取得"""
        return os.path.splitext(os.path.basename(file_path))[0]

    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """ファイル名から拡張子を取得"""
        return os.path.splitext(file_path)[1].lower()

    @staticmethod
    def get_files_with_extension(directory: str, extension: str) -> List[str]:
        """指定したディレクトリ内の特定の拡張子のファイルをすべて取得"""
        extension = extension.lower()
        return glob.glob(os.path.join(directory, extension))

    @staticmethod
    def directory_exists(path: str) -> bool:
        return os.path.isdir(path)

    @staticmethod
    def create_directory(path: str, exist_ok: bool = True) -> None:
        os.makedirs(path, exist_ok=exist_ok)

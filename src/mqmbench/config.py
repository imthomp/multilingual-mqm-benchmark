from pathlib import Path

from dynaconf import Dynaconf

settings = Dynaconf()


def init_settings(settings_file_path=None):
    global settings
    if settings_file_path:
        settings.load_file(path=[str(settings_file_path)])
        settings.data.settings_file_path = str(Path(settings_file_path).resolve())

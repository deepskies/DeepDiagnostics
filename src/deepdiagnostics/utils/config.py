from typing import Optional
import os
import yaml

from deepdiagnostics.utils.defaults import Defaults


def get_item(section, item, raise_exception=True):
    return Config().get_item(section, item, raise_exception)


def get_section(section, raise_exception=True):
    return Config().get_section(section, raise_exception)


class Config:
    ENV_VAR_PATH = "DeepDiagnostics_Config"

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is not None:
            # Add it to the env vars in case we need to get it later.
            os.environ[self.ENV_VAR_PATH] = config_path
            self.config = self._read_config(config_path)
            self._validate_config()

        else:
            # Get it from the env vars
            try:
                config_path = os.environ[self.ENV_VAR_PATH]
                self.config = self._read_config(config_path)
                self._validate_config()

            except KeyError:
                print("Warning: Cannot load config from environment. Hint: Have you set the config path by passing a str path to Config?")
                self.config = Defaults

    def _validate_config(self):
        # Validate common
        # TODO
        pass

    def _read_config(self, path):
        assert os.path.exists(path), f"Config path at {path} does not exist."
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def get_item(self, section, item, raise_exception=True):
        try:
            return self.config[section][item]
        except KeyError as e:
            if raise_exception:
                raise KeyError(f"Configuration File missing parameter {e}")
            else:
                return Defaults[section][item]

    def get_section(self, section, raise_exception=True):
        try:
            return self.config[section]
        except KeyError as e:
            if raise_exception:
                raise KeyError(e)
            else:
                return Defaults[section]

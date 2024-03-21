
from typing import Optional
import os 
import yaml 


def get_item(section, item, raise_exception=True, default=None): 
    return Config().get_item(section, item, raise_exception, default) 

def get_section(section): 
    return Config().get_section(section)


class Config: 
    ENV_VAR = "DeepDiagnostics_Config"
    def __init__(self, config_path:Optional[str]) -> None:
        if config_path is not None: 
            # Add it to the env vars in case we need to get it later. 
            os.environ[self.ENV_VAR] = config_path
        else: 
            # Get it from the env vars 
            try: 
                config_path =  os.environ[self.ENV_VAR]
            except KeyError: 
                assert False, "Cannot load config from enviroment. Hint: Have you set the config path by pasing a str path to Config?"
        
        self.config = self._read_config(config_path)

    def _read_config(self, path): 
        assert os.path.exist(path), f"Config path at {path} does not exist."
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def get_item(self, section, item, raise_exception=True, default=None): 
        try: 
            return self.config[section][item]
        except KeyError as e: 
            if raise_exception: 
                raise KeyError(e)
            else: 
                return default

    def get_section(self, section): 
        return self.config[section]
from pathlib import Path
import cv2 as cv
import yaml
import json
import logging
import pickle
YAML_SUPPORT = True
YAML_NOT_DETECTED_MESSAGE = "yaml is not installed, consider installing it by pip install PyYAML"
try:
    import yaml
    from yaml.loader import SafeLoader
except:
    YAML_SUPPORT = False
    logging.warning(YAML_NOT_DETECTED_MESSAGE)

class Image:
    @staticmethod
    def load(path: Path):
        assert path.exists()
        image = cv.imread(str(path))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return image

    @staticmethod
    def write(path:Path, img):
        image = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite(str(path), image)

class Dump:
    @staticmethod
    def load_yaml(path:Path,) -> dict:
        assert YAML_SUPPORT, YAML_NOT_DETECTED_MESSAGE
        with open(path) as file:
            params = yaml.load(file, Loader=SafeLoader)
        return params
    
    @staticmethod
    def save_yaml(data: dict, path:Path, **kwargs):
        assert YAML_SUPPORT, YAML_NOT_DETECTED_MESSAGE
        with open(path, 'w') as outfile:
            yaml.dump(data, outfile, **kwargs)
    
    @staticmethod
    def load_json(path:Path,) -> dict:
        with open(path) as file:
            params = json.load(file)
        return params

    @staticmethod
    def save_json(data: dict, path:Path):
        with open(path, 'w') as outfile:
            json.dump(data, outfile)
    
    @staticmethod
    def load_pickle(path:Path,) -> dict:
        with open(path, "rb") as file:
            params = pickle.load(file)
        return params

    @staticmethod
    def save_pickle(data: dict, path:Path):
        with open(path, 'wb') as outfile:
            pickle.dump(data, outfile)
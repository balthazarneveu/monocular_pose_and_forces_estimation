from pathlib import Path
import cv2 as cv
class Image:
    @staticmethod
    def load(path: Path):
        assert path.exists()
        image = cv.imread(str(path))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return image

    @staticmethod
    def write(path:Path, img):
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imwrite(str(path), img)
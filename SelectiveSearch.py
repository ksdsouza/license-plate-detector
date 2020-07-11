import cv2


class SelectiveSearch:
    def __init__(self):
        cv2.setUseOptimized(True)
        self.selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.SS_IMG_SIZE = (224, 224)

    def process_image(self, path: str):
        image = cv2.imread(path)
        self.selective_search.setBaseImage(image)
        self.selective_search.switchToSelectiveSearchFast()
        return self.selective_search.process(), image.copy()

    def resize(self, img):
        return cv2.resize(img, self.SS_IMG_SIZE, interpolation=cv2.INTER_AREA)

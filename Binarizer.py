import numpy as np
from sklearn.preprocessing import LabelBinarizer


class Binarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1 - Y))
        return Y

    def inverse_transform(self, Y, threshold=None):
        y_value = Y[:, 0] if self.y_type_ == 'binary' else Y
        return super().inverse_transform(y_value, threshold)
import numpy as np


class LearningData:
    def __init__(self, image, age, gender, skin_status):
        self.image = image
        self.age = age
        self.gender = gender
        self.status = skin_status

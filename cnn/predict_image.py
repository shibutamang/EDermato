from predict import SkinDiseaseModel
import numpy as np
from keras.preprocessing import image

TEST_IMG = "D:\\Projects\\Project X\\dataset\\testing_set\\bkl\\ISIC_0033761.jpg"

model = SkinDiseaseModel("model.json", "model_weights.h5")

test_image = image.load_img(TEST_IMG, target_size = (64,64),color_mode = 'grayscale')
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

model.predict_disease(test_image)   

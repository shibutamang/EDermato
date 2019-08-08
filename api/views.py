from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .predict import SkinDiseaseModel
import numpy as np
from keras.preprocessing import image
import json

from .serializers import ImageSerializer


class FileUploadView(APIView):
    parser_class = (FileUploadParser,)
    DISEASE_LIST = ["akiec", "bcc", "bkl", "mel", "nv", "vasc"]

    def post(self, request, *args, **kwargs):

      file_serializer = ImageSerializer(data=request.data)

      if file_serializer.is_valid():
          file_serializer.save()

          TEST_IMG = file_serializer.data

          img = '.' + TEST_IMG['image']
          print(img)

          model = SkinDiseaseModel("./model/model.json", "./model/model_weights.h5")

          test_image = image.load_img(img, target_size = (64,64),color_mode = 'grayscale')
          test_image = image.img_to_array(test_image)
          test_image = np.expand_dims(test_image, axis = 0)
          
          pred = model.predict_disease(test_image) 
          print(pred) 
          accuracy = np.amax(pred) * 100
          dtype = FileUploadView.DISEASE_LIST[np.argmax(pred)]
          diagnosis = ''
          description = ''

          if (dtype == 'akiec'):
              diagnosis = 'Actinic keratoses'
              description = 'An actinic keratosis (AK), also known as a solar keratosis, is a crusty, scaly growth caused by damage from exposure to ultraviolet (UV) radiation. You’ll often see the plural, “keratoses,” because there is seldom just one. AK is considered a precancer because if left alone, it could develop into a skin cancer, most often the second most common form of the disease, squamous cell carcinoma (SCC).  More than 419,000 cases of skin cancer in the U.S. each year are linked to indoor tanning, including about 168,000 squamous cell carcinomas'
          elif (dtype == 'bcc'):
              diagnosis = 'Basal cell carcinoma'
              description = 'Basal cell carcinoma is a type of skin cancer. Basal cell carcinoma begins in the basal cells — a type of cell within the skin that produces new skin cells as old ones die off'
          elif (dtype == 'bkl'):
              diagnosis = 'Seborrheic keratoses (benign keratosis-like lesions)'
              description = 'Seborrheic keratosis (seb-o-REE-ik ker-uh-TOE-sis) is one of the most common noncancerous skin growths in older adults'
          elif (dtype == 'mel'):
              diagnosis = 'Melanoma'
              description = 'The most dangerous form of skin cancer, these cancerous growths develop when unrepaired DNA damage to skin cells (most often caused by ultraviolet radiation from sunshine or tanning beds) triggers mutations (genetic defects) that lead the skin cells to multiply rapidly and form malignant tumors'
          elif (dtype == 'nv'):
              diagnosis = 'Melanocytic nevi'
              description = 'Melanocytic nevi are benign neoplasms or hamartomas composed of melanocytes, the pigment-producing cells that constitutively colonize the epidermis'
          elif (dtype == 'vasc'):
              diagnosis = 'Vascular lesions'
              description = 'Vascular lesions in childhood are comprised of vascular tumors and vascular malformations. Vascular tumors encompass neoplasms of the vascular system, of which infantile hemangiomas (IHs) are the most common'

          res = {'diagnosis': diagnosis, 'type':'skin cancer', 'accuracy': accuracy, 'description': description}
          
          return Response(res, status=status.HTTP_201_CREATED)
      else:
          return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    
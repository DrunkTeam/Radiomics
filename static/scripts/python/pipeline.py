import subprocess
import SimpleITK as sitk
from  radiomics import featureextractor
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

class RadiomicsProcessor:
    def __init__(self,image_path, mask_path,data_path=None):
        self.data_path = data_path
        self.image_path = image_path
        self.mask_path = mask_path
        self.data = {}

    def get_radiomics(self):
      self.data[0] = {}
      image = sitk.ReadImage(self.image_path)
      mask = sitk.ReadImage(self.mask_path)
      extractor = featureextractor.RadiomicsFeatureExtractor()
      extractor.enableAllFeatures()
      extractor.settings['correctMask'] = True
      features = extractor.execute(image, mask)
      for k, v in features.items():
        self.data[0][k] = v

    def get_data(self):
      self.get_radiomics()

    def data_to_csv(self):
      return pd.DataFrame.from_dict(self.data, orient='index')



    def run(self):
      self.get_data()
      return self.data_to_csv()


class Pipeline:
    def __init__(self):
        self.classifier_TabPFN = torch.load('/content/model_TabPFN.pth')
        self.classifier_adaboost = torch.load('/content/model_Adaboost.pth')
        self.net = torch.load('/content/model_net.pth')

        self.processor = RadiomicsProcessor('/content/ABUBAKAROVA.nii','/content/ABUBAKAROVA_label.nii','/content/pyradiomics/data/')
        self.convert_mri = self.processor.run()

        self.input=self.convert_mri.iloc[:, 22:]
        self.input=self.input.iloc[:, :107]
        pass


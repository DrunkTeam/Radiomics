from fastapi import FastAPI, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles

# from static.scripts.python.pipeline import Pipeline
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

import subprocess
import SimpleITK as sitk
from  radiomics import featureextractor
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import mode

from pathlib import Path

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(107, 32)  # Входной слой: 107 входных признаков, 64 нейрона
        self.fc3 = nn.Linear(32, 1)    # Выходной слой: 32 нейрона, 1 выход (бинарная классификация)

    def forward(self, x):
        x = torch.relu(self.fc1(x))   # Применяем ReLU к выходу первого слоя
        x = torch.sigmoid(self.fc3(x))  # Применяем сигмоиду к выходу третьего слоя
        return x


features=['original_shape_Maximum2DDiameterColumn', 'original_glcm_JointAverage',
       'original_glszm_LargeAreaLowGrayLevelEmphasis',
       'original_glcm_MaximumProbability', 'original_glcm_DifferenceEntropy',
       'original_glrlm_ShortRunLowGrayLevelEmphasis',
       'original_glszm_SmallAreaEmphasis',
       'original_glrlm_LongRunLowGrayLevelEmphasis',
       'original_glcm_JointEntropy', 'original_glcm_ClusterProminence',
       'original_shape_Maximum2DDiameterSlice', 'original_glcm_Contrast',
       'original_glcm_DifferenceVariance',
       'original_glrlm_HighGrayLevelRunEmphasis',
       'original_gldm_DependenceNonUniformityNormalized',
       'original_shape_SurfaceArea', 'original_glcm_ClusterTendency',
       'original_shape_SurfaceVolumeRatio', 'original_glszm_ZoneVariance',
       'original_gldm_DependenceEntropy', 'original_shape_MajorAxisLength',
       'original_glrlm_RunLengthNonUniformity',
       'original_gldm_GrayLevelVariance', 'original_firstorder_Range',
       'original_glszm_GrayLevelVariance', 'original_glrlm_LongRunEmphasis',
       'original_glcm_SumAverage', 'original_glrlm_RunEntropy',
       'original_gldm_HighGrayLevelEmphasis', 'original_ngtdm_Busyness',
       'original_firstorder_InterquartileRange',
       'original_gldm_GrayLevelNonUniformity', 'original_firstorder_Entropy',
       'original_ngtdm_Strength', 'original_firstorder_Median',
       'original_glrlm_RunLengthNonUniformityNormalized',
       'original_gldm_SmallDependenceEmphasis',
       'original_glszm_LargeAreaEmphasis', 'original_firstorder_Mean',
       'original_glcm_Idn', 'original_shape_Maximum3DDiameter',
       'original_glrlm_LowGrayLevelRunEmphasis', 'original_glcm_JointEnergy',
       'original_firstorder_TotalEnergy',
       'original_glszm_GrayLevelNonUniformityNormalized',
       'original_firstorder_Uniformity',
       'original_glrlm_GrayLevelNonUniformity', 'original_glcm_MCC',
       'original_glrlm_GrayLevelVariance', 'original_glcm_Imc2',
       'original_firstorder_RobustMeanAbsoluteDeviation',
       'original_gldm_LargeDependenceLowGrayLevelEmphasis',
       'original_glszm_ZonePercentage',
       'original_glrlm_GrayLevelNonUniformityNormalized',
       'original_glcm_Autocorrelation',
       'original_glrlm_ShortRunHighGrayLevelEmphasis',
       'original_shape_MinorAxisLength', 'original_glrlm_RunPercentage',
       'original_firstorder_Energy', 'original_firstorder_Maximum']

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

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# pipeline = Pipeline('data/ABUBAKAROVA.nii', 'data/ABUBAKAROVA_label.nii')
# print(pipeline.predict())

@app.post("/upload")
async def upload_files(file1: UploadFile, file2: UploadFile):
    # if 1:
    #     return {"Result": 1}
    # else:
    #     return {"Result": 0}
    
    # Читаем содержимое файлов
    content1 = await file1.read()
    content2 = await file2.read()

    with open("data/ABUBAKAROVA.nii", "wb") as f1:
        f1.write(content1)
    
    with open("data/ABUBAKAROVA_label.nii", "wb") as f2:
        f2.write(content2)
    print("start pipeline")

    path_series = "data/ABUBAKAROVA.nii"
    path_mask = "data/ABUBAKAROVA_label.nii"

    classifier_TabPFN = torch.load('models/model_TabPFN.pth')
    # classifier_TabPFN = torch.load('models/model_TabPFN.pth', map_location=lambda storage, loc: storage)
    classifier_adaboost = torch.load('models/model_Adaboost.pth')
    net = SimpleNet()
    net.load_state_dict(torch.load('models/model_net.pth'))
    # net = torch.load('models/model_net.pth')

    processor = RadiomicsProcessor(path_series, path_mask,'')
    convert_mri = processor.run()

    convert_mri.to_csv("data/test.csv", index=False)

    input=convert_mri.iloc[:, 22:]
    input=input.iloc[:, :107]
    loader_test=torch.tensor(input.to_numpy().astype(np.float32), dtype=torch.float32)

    n_trees = 3
    # n_trees = 2
    base_pred = np.zeros((input.shape[0], n_trees), dtype="int")
    classifiers = [classifier_TabPFN,classifier_adaboost, net]
    # classifiers = [classifier_adaboost, net]
    # classifiers = [classifier_TabPFN,classifier_adaboost]

    for i in range(n_trees):
        # obtain the predictions from each tree
        if classifiers[i]==classifier_TabPFN:
          base_pred[:,i], p_eval = classifiers[i].predict(input[features], return_winning_probability=True)

        if classifiers[i]==classifier_adaboost:
          base_pred[:,i]= classifiers[i].predict(input)

        if classifiers[i]==net:
          outputs = classifiers[i](loader_test)
          base_pred[:,i]= torch.round(outputs).squeeze().tolist()
    print(base_pred)

    # aggregate predictions by majority voting
    pred = mode(base_pred, axis=1)[0].ravel()

    print('предсказания - ', pred[0], '\n')

    # pipeline = Pipeline('data/ABUBAKAROVA.nii', 'data/ABUBAKAROVA_label.nii')
    # predict = pipeline.predict()

    if pred[0]:
        return {"Result": 1}
    else:
        return {"Result": 0}
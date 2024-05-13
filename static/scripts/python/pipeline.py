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
    def __init__(self, path_series, path_mask):
        self.classifier_TabPFN = torch.load('models/model_TabPFN.pth')
        self.classifier_adaboost = torch.load('models/model_Adaboost.pth')
        self.net = torch.load('models/model_net.pth')

        self.processor = RadiomicsProcessor(path_series, path_mask,'')
        self.convert_mri = self.processor.run()

        self.convert_mri.to_csv("data/test.csv", index=False)

        self.input=self.convert_mri.iloc[:, 22:]
        self.input=self.input.iloc[:, :107]
        self.loader_test=torch.tensor(self.input.to_numpy().astype(np.float32), dtype=torch.float32)
        
    def predict(self):
        n_trees = 3
        base_pred = np.zeros((self.input.shape[0], n_trees), dtype="int")
        classifiers = [self.classifier_TabPFN,self.classifier_adaboost, self.net]

        for i in range(n_trees):
            # obtain the predictions from each tree
            if classifiers[i]==self.classifier_TabPFN:
              base_pred[:,i], p_eval = classifiers[i].predict(self.input[features], return_winning_probability=True)

            if classifiers[i]==self.classifier_adaboost:
              base_pred[:,i]= classifiers[i].predict(self.input)

            if classifiers[i]==self.net:
              outputs = classifiers[i](self.loader_test)
              base_pred[:,i]= torch.round(outputs).squeeze().tolist()
        print(base_pred)

        # aggregate predictions by majority voting
        pred = mode(base_pred, axis=1)[0].ravel()

        print('предсказания - ', pred[0], '\n')
        return pred[0]


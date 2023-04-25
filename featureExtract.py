import radiomics
from radiomics import featureextractor
import os
import sys
import six
import SimpleITK as sitk
import numpy as np
import pandas as pd
import pickle
from glob import glob

if len(sys.argv) < 3:
    print('image and mask directory required!')
    sys.exit()

imagePathDir = sys.argv[1]
maskPathDir = sys.argv[2]

cwd = os.getcwd()
paramsFile = os.path.join(cwd, 'exampleCT.yaml')

# setting pyradiomics parameter
extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
extractor.enableAllFeatures()

print("Extraction parameters:\n\t", extractor.settings)
print("Enabled filters:\n\t", extractor.enabledImagetypes)
print("Enabled features:\n\t", extractor.enabledFeatures)

case_list = sorted(glob(os.path.join(imagePathDir, "*.nii.gz")))
print("number of image: ", len(case_list))

records = []
failed_case = []

for c in case_list:
    baseName = os.path.basename(c)
    case_name = baseName[:baseName.find('.nii.gz')]
    print(case_name)
    try:
      # get image file
      imagePath = os.path.join(imagePathDir, case_name + ".nii.gz")
      # get mask file
      labelPath = os.path.join(maskPathDir, case_name + ".nii.gz")
      result = extractor.execute(imagePath, labelPath)
      record = {}
      record["case_name"] = case_name
      # store pyradiomics feature
      for featureName, featureValue in six.iteritems(result):
        record[featureName] = featureValue
      records.append(record)
    except Exception as e:
      print(case_name, "failed")
      print(e)
      failed_case.append(case_name)

# output feature
df = pd.DataFrame(records)
outputPath = os.path.join(cwd, 'featureResult')
if not os.path.isdir(outputPath):
    os.mkdir(outputPath)
df.to_csv(os.path.join(cwd, 'featureResult', 'feature.csv'))
df.to_pickle(os.path.join(cwd, 'featureResult', 'feature.pkl'))
print("Failed case", failed_case)
print("Finish!")
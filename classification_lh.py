import pandas as pd
import os
from sklearn import ensemble, preprocessing, metrics
import joblib

cwd = os.getcwd()
featureFile = os.path.join(cwd, 'featureResult', 'feature.pkl')
with open(featureFile, 'rb') as f:
    tumorData = joblib.load(f)
case_list = tumorData["case_name"]
tumorData = tumorData.iloc[:, 38:]

modelFile = os.path.join(cwd, 'random_forest_classification_model_lh.joblib')

## feature type
wavelet_column = []
log_column = []
original_column = []
all_cols = []

for item in tumorData.columns:
    all_cols.append(item)
    if 'wavelet' in item:
        wavelet_column.append(item)
    elif 'log' in item:
        log_column.append(item)
    elif 'original' in item:
        original_column.append(item)

log_matrix = [[],[],[],[],[]]

for log in log_column:
#     print(log)
    if '1-0' in log:
        log_matrix[0].append(log)
    elif '2-0' in log:
        log_matrix[1].append(log)
    elif '3-0' in log:
        log_matrix[2].append(log)
    elif '4-0' in log:
        log_matrix[3].append(log)
    elif '5-0' in log:
        log_matrix[4].append(log)
#     break

wavelet_matrix = [[], [], [], [], [], [], [], []]
for wavelet in wavelet_column:
#     print(log)
    if 'LLL' in wavelet:
        wavelet_matrix[0].append(wavelet)
    elif 'LLH' in wavelet:
        wavelet_matrix[1].append(wavelet)
    elif 'LHL' in wavelet:
        wavelet_matrix[2].append(wavelet)
    elif 'HLL' in wavelet:
        wavelet_matrix[3].append(wavelet)
    elif 'LHH' in wavelet:
        wavelet_matrix[4].append(wavelet)
    elif 'HLH' in wavelet:
        wavelet_matrix[5].append(wavelet)
    elif 'HHL' in wavelet:
        wavelet_matrix[6].append(wavelet)
    elif 'HHH' in wavelet:
        wavelet_matrix[7].append(wavelet)
        
all_columns = [all_cols, original_column, wavelet_column, log_column, wavelet_matrix[0], wavelet_matrix[1], wavelet_matrix[2], wavelet_matrix[3], wavelet_matrix[4], wavelet_matrix[5], wavelet_matrix[6], wavelet_matrix[7], log_matrix[0], log_matrix[1], log_matrix[2], log_matrix[3], log_matrix[4]]

## mix filters
shape_columns = [item for item in tumorData.columns if 'shape' in item]
firstorder_columns = [item for item in tumorData.columns if 'firstorder' in item]
glcm_columns = [item for item in tumorData.columns if 'glcm' in item]
glrlm_columns = [item for item in tumorData.columns if 'glrlm' in item]
glszm_columns = [item for item in tumorData.columns if 'glszm' in item]
gldm_columns = [item for item in tumorData.columns if 'gldm' in item]
ngtdm_columns = [item for item in tumorData.columns if 'ngtdm' in item]
## mix filters

## split diff filters
split_original_column = []
split_original_column.append([item for item in original_column if 'firstorder' in item])
split_original_column.append([item for item in original_column if 'glcm' in item])
split_original_column.append([item for item in original_column if 'glrlm' in item])
split_original_column.append([item for item in original_column if 'glszm' in item])
split_original_column.append([item for item in original_column if 'gldm' in item])
split_original_column.append([item for item in original_column if 'ngtdm' in item])
split_original_column.append([item for item in original_column if 'firstorder' not in item and 'shape' not in item])

split_wavelet_matrix = []
for i in range(len(wavelet_matrix)):
    split_wavelet_matrix.append([item for item in wavelet_matrix[i] if 'firstorder' in item])
    split_wavelet_matrix.append([item for item in wavelet_matrix[i] if 'glcm' in item])
    split_wavelet_matrix.append([item for item in wavelet_matrix[i] if 'glrlm' in item])
    split_wavelet_matrix.append([item for item in wavelet_matrix[i] if 'glszm' in item])
    split_wavelet_matrix.append([item for item in wavelet_matrix[i] if 'gldm' in item])
    split_wavelet_matrix.append([item for item in wavelet_matrix[i] if 'ngtdm' in item])
    split_wavelet_matrix.append([item for item in wavelet_matrix[i] if 'firstorder' not in item])

split_log_matrix = []
for i in range(len(log_matrix)):
    split_log_matrix.append([item for item in log_matrix[i] if 'firstorder' in item])
    split_log_matrix.append([item for item in log_matrix[i] if 'glcm' in item])
    split_log_matrix.append([item for item in log_matrix[i] if 'glrlm' in item])
    split_log_matrix.append([item for item in log_matrix[i] if 'glszm' in item])
    split_log_matrix.append([item for item in log_matrix[i] if 'gldm' in item])
    split_log_matrix.append([item for item in log_matrix[i] if 'ngtdm' in item])
    split_log_matrix.append([item for item in log_matrix[i] if 'firstorder' not in item])

all_columns = [shape_columns, *split_original_column, *split_wavelet_matrix, *split_log_matrix]

## load model

model = joblib.load(modelFile)
predictProba = []
for j in [17]:
    tumorDataFeature = tumorData[all_columns[j]].copy(deep=True) ## shape_columns 
    predict_y = model.predict_proba(tumorDataFeature)
    predictProba = predict_y[:, 1]

out_df = {'case_name': case_list, 'predict_proba': predictProba}
out_df = pd.DataFrame(out_df)
outputPath = os.path.join(cwd, 'classificationResult')
if not os.path.isdir(outputPath):
    os.mkdir(outputPath)
out_df.to_csv(os.path.join(cwd, 'classificationResult', 'classificationResult_lh.csv'))

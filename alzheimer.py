from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os, itertools, subprocess
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing,tree
import plotly.plotly as py
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  cross_val_score, KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier

"""
The aim of the project is to find machine/deep learning algorithms that best predict the onset of
Alzheimer's Disease (AD) given patients' biomarkers data, including MRI, PET, CSF, cognitive tests,
and demographic data (age, gender, etc.). This file only runs tree-based and
some ensemble algorithms. More implementations (Neural Nets and SVM) can be seen here:
- Neural Nets: https://raw.githubusercontent.com/mbindhi3/CS229Project/master/NN_wth_cv.py
- SVM: https://raw.githubusercontent.com/kechavez/AD_Classification/master/Alzheimer%20Visualization%20and%20Learning.ipynb

Before feeding the input to machine learning algorithm, we pre-process the data by performing 
standard scaling, clean-up, and principal component analysis (PCA) to reduce the size of the 
features and reduce overall variance. The algorithm is run with k-fold cross validation to
tackle overfitting issue. The resulting prediction is whether a person is having an Alzheimer's
Disease or Normal.

More information about the case can be viewed here:
Report: http://cs229.stanford.edu/proj2017/final-reports/5233661.pdf
Poster: http://cs229.stanford.edu/proj2017/final-posters/5145027.pdf
"""

def load_data():
    """
    This function loads all raw features including MRI, PET, CSF, cognitive tests, and demographic data (age, gender, etc.)
    and label (having AD or not). The input file is stored on-line.
    :return: A pandas dataframe of raw data
    """
    url='https://raw.githubusercontent.com/titaristanto/Alzheimer-s-Disease-Prediction/master/alzheimer_input.csv'
    df = pd.read_csv(url,low_memory=False)
    #df=shuffle(df)
    raw_gen=df.loc[:,['AGE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY','APOE4','DX']]
    raw_cognitive_test=df.loc[:,['ADAS11', 'MMSE', 'RAVLT_immediate']]
    raw_MRI=df.loc[:,['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform','MidTemp']]
    raw_PET=df.loc[:,['FDG','AV45']]
    raw_CSF=df.loc[:,['ABETA_UPENNBIOMK9_04_19_17','TAU_UPENNBIOMK9_04_19_17','PTAU_UPENNBIOMK9_04_19_17']]

    # Other Raw Data
    raw_biomarkers=df.loc[:,['CEREBELLUMGREYMATTER_UCBERKELEYAV45_10_17_16',
                            'WHOLECEREBELLUM_UCBERKELEYAV45_10_17_16',
                            'ERODED_SUBCORTICALWM_UCBERKELEYAV45_10_17_16',
                            'FRONTAL_UCBERKELEYAV45_10_17_16',
                            'CINGULATE_UCBERKELEYAV45_10_17_16',
                            'PARIETAL_UCBERKELEYAV45_10_17_16',
                            'TEMPORAL_UCBERKELEYAV45_10_17_16',
                            'SUMMARYSUVR_WHOLECEREBNORM_UCBERKELEYAV45_10_17_16',
                            'SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF_UCBERKELEYAV45_10_17_16',
                            'SUMMARYSUVR_COMPOSITE_REFNORM_UCBERKELEYAV45_10_17_16',
                            'SUMMARYSUVR_COMPOSITE_REFNORM_0.79CUTOFF_UCBERKELEYAV45_10_17_16',
                            'BRAINSTEM_UCBERKELEYAV45_10_17_16',
                            'BRAINSTEM_SIZE_UCBERKELEYAV45_10_17_16',
                            'VENTRICLE_3RD_UCBERKELEYAV45_10_17_16',
                            'VENTRICLE_3RD_SIZE_UCBERKELEYAV45_10_17_16',
                            'VENTRICLE_4TH_UCBERKELEYAV45_10_17_16',
                            'VENTRICLE_4TH_SIZE_UCBERKELEYAV45_10_17_16',
                            'VENTRICLE_5TH_UCBERKELEYAV45_10_17_16',
                            'VENTRICLE_5TH_SIZE_UCBERKELEYAV45_10_17_16',
                            'CC_ANTERIOR_UCBERKELEYAV45_10_17_16',
                            'CC_ANTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
                            'CC_CENTRAL_UCBERKELEYAV45_10_17_16',
                            'CC_CENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CC_MID_ANTERIOR_UCBERKELEYAV45_10_17_16',
                            'CC_MID_ANTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
                            'CC_MID_POSTERIOR_UCBERKELEYAV45_10_17_16',
                            'CC_MID_POSTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
                            'CC_POSTERIOR_UCBERKELEYAV45_10_17_16',
                            'CC_POSTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
                            'CSF_UCBERKELEYAV45_10_17_16',
                            'CSF_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_BANKSSTS_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_BANKSSTS_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_CAUDALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_CAUDALANTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_CAUDALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_CAUDALMIDDLEFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_CUNEUS_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_CUNEUS_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_ENTORHINAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_ENTORHINAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_FRONTALPOLE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_FRONTALPOLE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_FUSIFORM_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_FUSIFORM_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_INFERIORPARIETAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_INFERIORPARIETAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_INFERIORTEMPORAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_INFERIORTEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_INSULA_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_INSULA_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_ISTHMUSCINGULATE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_ISTHMUSCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_LATERALOCCIPITAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_LATERALOCCIPITAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_LATERALORBITOFRONTAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_LATERALORBITOFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_LINGUAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_LINGUAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_MEDIALORBITOFRONTAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_MEDIALORBITOFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_MIDDLETEMPORAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_MIDDLETEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PARACENTRAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PARACENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PARAHIPPOCAMPAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PARAHIPPOCAMPAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PARSOPERCULARIS_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PARSOPERCULARIS_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PARSORBITALIS_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PARSORBITALIS_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PARSTRIANGULARIS_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PARSTRIANGULARIS_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PERICALCARINE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PERICALCARINE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_POSTCENTRAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_POSTCENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_POSTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_POSTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PRECENTRAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PRECENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PRECUNEUS_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_PRECUNEUS_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_ROSTRALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_ROSTRALANTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_ROSTRALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_ROSTRALMIDDLEFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_SUPERIORFRONTAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_SUPERIORFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_SUPERIORPARIETAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_SUPERIORPARIETAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_SUPERIORTEMPORAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_SUPERIORTEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_SUPRAMARGINAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_SUPRAMARGINAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_TEMPORALPOLE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_TEMPORALPOLE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_TRANSVERSETEMPORAL_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_TRANSVERSETEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_UNKNOWN_UCBERKELEYAV45_10_17_16',
                            'CTX_LH_UNKNOWN_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_BANKSSTS_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_BANKSSTS_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_CAUDALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_CAUDALANTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_CAUDALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_CAUDALMIDDLEFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_CUNEUS_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_CUNEUS_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_ENTORHINAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_ENTORHINAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_FRONTALPOLE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_FRONTALPOLE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_FUSIFORM_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_FUSIFORM_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_INFERIORPARIETAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_INFERIORPARIETAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_INFERIORTEMPORAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_INFERIORTEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_INSULA_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_INSULA_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_ISTHMUSCINGULATE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_ISTHMUSCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_LATERALOCCIPITAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_LATERALOCCIPITAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_LATERALORBITOFRONTAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_LATERALORBITOFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_LINGUAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_LINGUAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_MEDIALORBITOFRONTAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_MEDIALORBITOFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_MIDDLETEMPORAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_MIDDLETEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PARACENTRAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PARACENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PARAHIPPOCAMPAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PARAHIPPOCAMPAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PARSOPERCULARIS_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PARSOPERCULARIS_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PARSORBITALIS_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PARSORBITALIS_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PARSTRIANGULARIS_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PARSTRIANGULARIS_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PERICALCARINE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PERICALCARINE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_POSTCENTRAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_POSTCENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_POSTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_POSTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PRECENTRAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PRECENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PRECUNEUS_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_PRECUNEUS_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_ROSTRALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_ROSTRALANTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_ROSTRALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_ROSTRALMIDDLEFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_SUPERIORFRONTAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_SUPERIORFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_SUPERIORPARIETAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_SUPERIORPARIETAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_SUPERIORTEMPORAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_SUPERIORTEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_SUPRAMARGINAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_SUPRAMARGINAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_TEMPORALPOLE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_TEMPORALPOLE_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_TRANSVERSETEMPORAL_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_TRANSVERSETEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_UNKNOWN_UCBERKELEYAV45_10_17_16',
                            'CTX_RH_UNKNOWN_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_ACCUMBENS_AREA_UCBERKELEYAV45_10_17_16',
                            'LEFT_ACCUMBENS_AREA_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_AMYGDALA_UCBERKELEYAV45_10_17_16',
                            'LEFT_AMYGDALA_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_CAUDATE_UCBERKELEYAV45_10_17_16',
                            'LEFT_CAUDATE_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_CEREBELLUM_CORTEX_UCBERKELEYAV45_10_17_16',
                            'LEFT_CEREBELLUM_CORTEX_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_CEREBELLUM_WHITE_MATTER_UCBERKELEYAV45_10_17_16',
                            'LEFT_CEREBELLUM_WHITE_MATTER_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_CEREBRAL_WHITE_MATTER_UCBERKELEYAV45_10_17_16',
                            'LEFT_CEREBRAL_WHITE_MATTER_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_CHOROID_PLEXUS_UCBERKELEYAV45_10_17_16',
                            'LEFT_CHOROID_PLEXUS_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16',
                            'LEFT_HIPPOCAMPUS_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_INF_LAT_VENT_UCBERKELEYAV45_10_17_16',
                            'LEFT_INF_LAT_VENT_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_LATERAL_VENTRICLE_UCBERKELEYAV45_10_17_16',
                            'LEFT_LATERAL_VENTRICLE_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_PALLIDUM_UCBERKELEYAV45_10_17_16',
                            'LEFT_PALLIDUM_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_PUTAMEN_UCBERKELEYAV45_10_17_16',
                            'LEFT_PUTAMEN_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_THALAMUS_PROPER_UCBERKELEYAV45_10_17_16',
                            'LEFT_THALAMUS_PROPER_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_VENTRALDC_UCBERKELEYAV45_10_17_16',
                            'LEFT_VENTRALDC_SIZE_UCBERKELEYAV45_10_17_16',
                            'LEFT_VESSEL_UCBERKELEYAV45_10_17_16',
                            'LEFT_VESSEL_SIZE_UCBERKELEYAV45_10_17_16',
                            'NON_WM_HYPOINTENSITIES_UCBERKELEYAV45_10_17_16',
                            'NON_WM_HYPOINTENSITIES_SIZE_UCBERKELEYAV45_10_17_16',
                            'OPTIC_CHIASM_UCBERKELEYAV45_10_17_16',
                            'OPTIC_CHIASM_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_ACCUMBENS_AREA_UCBERKELEYAV45_10_17_16',
                            'RIGHT_ACCUMBENS_AREA_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_AMYGDALA_UCBERKELEYAV45_10_17_16',
                            'RIGHT_AMYGDALA_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_CAUDATE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_CAUDATE_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_CEREBELLUM_CORTEX_UCBERKELEYAV45_10_17_16',
                            'RIGHT_CEREBELLUM_CORTEX_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_CEREBELLUM_WHITE_MATTER_UCBERKELEYAV45_10_17_16',
                            'RIGHT_CEREBELLUM_WHITE_MATTER_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_CEREBRAL_WHITE_MATTER_UCBERKELEYAV45_10_17_16',
                            'RIGHT_CEREBRAL_WHITE_MATTER_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_CHOROID_PLEXUS_UCBERKELEYAV45_10_17_16',
                            'RIGHT_CHOROID_PLEXUS_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16',
                            'RIGHT_HIPPOCAMPUS_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_INF_LAT_VENT_UCBERKELEYAV45_10_17_16',
                            'RIGHT_INF_LAT_VENT_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_LATERAL_VENTRICLE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_LATERAL_VENTRICLE_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_PALLIDUM_UCBERKELEYAV45_10_17_16',
                            'RIGHT_PALLIDUM_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_PUTAMEN_UCBERKELEYAV45_10_17_16',
                            'RIGHT_PUTAMEN_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_THALAMUS_PROPER_UCBERKELEYAV45_10_17_16',
                            'RIGHT_THALAMUS_PROPER_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_VENTRALDC_UCBERKELEYAV45_10_17_16',
                            'RIGHT_VENTRALDC_SIZE_UCBERKELEYAV45_10_17_16',
                            'RIGHT_VESSEL_UCBERKELEYAV45_10_17_16',
                            'RIGHT_VESSEL_SIZE_UCBERKELEYAV45_10_17_16',
                            'WM_HYPOINTENSITIES_UCBERKELEYAV45_10_17_16',
                            'WM_HYPOINTENSITIES_SIZE_UCBERKELEYAV45_10_17_16'
                                    ]]

    raw_data=pd.concat([raw_gen,raw_cognitive_test,raw_biomarkers],axis=1,join='inner')
    return raw_data

def preprocess_data(raw_data):
    """This function 'cleans' the raw data from missing data points and converts some
    variables from numerical into categorical using label encoder"""
    # Drops missing values
    raw_data_cleaned=raw_data.dropna(how='any')

    # Converts 'DX' to 2 labels only: MCI is considered Dementia
    raw_data_cleaned=conv_binary(raw_data_cleaned)

    # Sets some features as categorical
    # Remarks: PTGENDER: 0:Female; 1: Male -- #PTMARRY: 0:Divorced; 1: Married; 2: Never Married 4: Widowed
    xcat_p = raw_data_cleaned[['PTGENDER','PTMARRY','APOE4']]
    raw_data_cleaned.drop(['PTGENDER','PTMARRY','APOE4'], axis=1, inplace=True)

    # Extracts label. Remarks: #DX: 0: Dementia, 1:Normal
    y_p = raw_data_cleaned[['DX']]
    raw_data_cleaned.drop(['DX'], axis=1, inplace=True)

    le = preprocessing.LabelEncoder()
    xcat=xcat_p.apply(le.fit_transform)
    x=pd.concat([xcat,raw_data_cleaned],axis=1,join='inner')

    # Sets 'DX' (AD or Not) as categorical
    y=y_p.apply(le.fit_transform)
    comb=pd.concat([x,y],axis=1,join='inner')
    clean_comb=clean_data(comb)

    y = clean_comb[['DX']]
    clean_comb.drop(['DX'], axis=1, inplace=True)
    return clean_comb,y

def clean_data(raw_data):
    """Additional data clean-up"""
    xnum= raw_data.apply(pd.to_numeric, errors='coerce')
    xnum = xnum.dropna()
    return xnum

def conv_binary(raw_data_cleaned):
    """Converts 'DX' to 2 labels only: MCI is considered Dementia"""
    raw_data_cleaned=raw_data_cleaned.replace('Dementia to MCI', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('MCI', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('MCI to Dementia', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('NL to MCI', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('MCI to NL', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('NL to Dementia', 'Dementia')
    return raw_data_cleaned

def conv_binary2(raw_data_cleaned):
    """Converts 'DX' to 2 labels only: MCI is considered NL"""
    raw_data_cleaned=raw_data_cleaned.replace('Dementia to MCI', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('MCI', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('MCI to Dementia', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('NL to MCI', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('MCI to NL', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('NL to Dementia', 'Dementia')
    return raw_data_cleaned

def split_data(x,y):
    """This function splits the data into training and test data"""
    train_split=0.8 # fraction of the data set used in the training set
    m=x.shape[0] # number of data points

    x_train=x.iloc[0:int(m*train_split),:]
    y_train=y.iloc[0:int(m*train_split),:]
    x_test=x.iloc[int(m*train_split)+1:m-1,:]
    y_test=y.iloc[int(m*train_split)+1:m-1,:]
    return x_train, y_train, x_test, y_test

def decision_tree(x_train,y_train):
    """This function trains the inputted pair of features and label, then returns the trained classifier.
    Cross-validation is performed to avoid overfitting"""
    #clf=DecisionTreeClassifier(criterion="gini",min_samples_split=15,random_state=0)
    clf = RandomForestClassifier(n_estimators=80,max_depth=5, random_state=0)
    #clf = ExtraTreesClassifier(n_estimators=80, max_depth=30, random_state=0)
    #clf=AdaBoostClassifier(n_estimators=25, random_state=0)
    clf.fit(x_train,y_train)

    scores=cross_val_score(clf, x_train, y_train['DX'], cv=10)
    print("cross_val_score mean: {:.3f} (std: {:.3f})".format(scores.mean(),scores.std()),end="\n\n" )

    return clf

def kfold_CV(models, x, y, k=10):
    """ Performs k-fold cross-validation to training data"""
    rs = KFold(k, shuffle=True, random_state=0)


    for name, model in models.items():
      print('\n\nModel: ', name)
      sum_train = 0
      sum_dev_test = 0
      for train_index, dev_test_index in rs.split(x):
        x_pca_train,x_lda_train, x_pca_dev_test, x_lda_dev_test = \
          run_PCA_LDA(x.iloc[train_index],y.iloc[train_index], \
                      x.iloc[dev_test_index], components=10)

        model.fit(x_lda_train, y.iloc[train_index])

        predicted_labels = model.predict(x_lda_dev_test)
        training_score = \
           accuracy_score(y.iloc[train_index], model.predict(x_lda_train))
        dev_testing_score = accuracy_score(y.iloc[dev_test_index], predicted_labels)

        cnf_matrix=confusion_matrix(y.iloc[dev_test_index], predicted_labels)
        class_names=list(['Dementia','NL'])
        # plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')

        sum_train = sum_train + training_score;
        sum_dev_test = sum_dev_test + dev_testing_score;
        # print('train score', training_score, ' dev test score', dev_testing_score)

      print("Average Training Score : ", sum_train/k)
      print("Average Dev Testing Score : ", sum_dev_test/k)


def run_PCA_LDA(X,y,xtest,components):
    """
    This function runs both PCA and LDA to compress the number of features, aiming to reduce the
    overall variance of the data. PCA only requires feature matrix (unsupervised learning), while
    LDA requires feature amtrix and label vector (supervised learning).
    :param X: training feature matrix
    :param y: training label
    :param xtest: feature matrix used for prediction
    :param components: number of desired compressed features
    :return:
    """
    y=np.ravel(y)
    target_names = ['Dementia', 'NL'] # 'MCI','NL','MCI to Dementia']

    pca = PCA(n_components=components)
    pca1 =  pca.fit(X)
    X_r = pca1.transform(X)
    Xtest_r = pca1.transform(xtest)

    lda = LinearDiscriminantAnalysis(n_components=10)
    lda1= lda.fit(X, y)
    X_r2 = lda1.transform(X)
    # print('xr2', X_r2.shape)
    Xtest_r2 = lda1.transform(xtest)

    x_pca=pd.DataFrame(X_r)
    x_lda=pd.DataFrame(X_r2)
    xtest_pca=pd.DataFrame(Xtest_r)
    xtest_lda=pd.DataFrame(Xtest_r2)
    y=pd.DataFrame(y)
    return x_pca,x_lda,xtest_pca,xtest_lda

def run_PCA_LDA1(X,y,components):
    """
    This function runs both PCA and LDA to compress the number of features, aiming to reduce the
    overall variance of the data. This function also displays the transformed dataset on a 2D-plot in which
    the axes are the two strongest eigen-components.
    :param X: training feature matrix
    :param y: training label
    :param xtest: feature matrix used for prediction
    :param components: number of desired compressed features
    :return:
    """
    y=np.ravel(y)
    target_names = ['Dementia','Normal']

    pca = PCA(n_components=components)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=components)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each component
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Tadpole dataset')
    plt.savefig('PCA_tadpole.png')

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of Tadpole dataset')
    plt.savefig('LDA_tadpole.png')

    plt.show()

    x_pca=pd.DataFrame(X_r)
    x_lda=pd.DataFrame(X_r2)

    return pca,lda,x_pca,x_lda

def build_pipe(x_train,y_train,x_test,y_test):
    """
    This function takes:
    :param x_train: matrix of features in training set
    :param y_train: vector of label in training set
    :param x_test: matrix of features in test set
    :param y_test: vector of label in test set

    and trains 4 different classifiers, returning score comparison of training, dev,
    and test set for each of them.
    """
    clf_dt=tree.DecisionTreeClassifier(random_state=0)
    pipe_dt = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=10)),
			('clf', clf_dt)])
    scores_dt=cross_val_score(clf_dt, x_train, y_train['DX'], cv=10)

    clf_rf=RandomForestClassifier(n_estimators=2, max_depth=1,random_state=0)
    pipe_rf = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=10)),
			('clf',clf_rf)])
    scores_rf=cross_val_score(clf_rf, x_train, y_train['DX'], cv=10)

    clf_et=ExtraTreesClassifier(n_estimators=80, max_depth=30, random_state=0)
    pipe_et = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=10)),
			('clf', clf_et)])
    scores_et=cross_val_score(clf_et, x_train, y_train['DX'], cv=10)

    clf_ab=AdaBoostClassifier(n_estimators=25, random_state=0)
    pipe_ab = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=10)),
			('clf', clf_ab)])
    scores_ab=cross_val_score(clf_ab, x_train, y_train['DX'], cv=10)

    pipelines = [pipe_dt, pipe_rf, pipe_et,pipe_ab]
    pipe_dict = {0: 'Decision Tree', 1: 'Random Forest', 2: 'Extra Tree',3: "AdaBoost"}
    for pipe in pipelines:
	    pipe.fit(x_train, y_train)
    cval_list=[np.average(scores_dt),np.average(scores_rf),np.average(scores_et),np.average(scores_ab)]

    # Compare accuracies
    for idx, val in enumerate(pipelines):
        print('%s pipeline training accuracy: %.3f' % (pipe_dict[idx], val.score(x_train, y_train)))
        print('%s pipeline dev accuracy: %.3f' % (pipe_dict[idx], cval_list[idx]))
        print('%s pipeline test accuracy: %.3f' % (pipe_dict[idx], val.score(x_test, y_test)))

    # Identify the most accurate model on test data
    best_acc = 0.0
    best_clf = 0
    best_pipe = ''
    for idx, val in enumerate(pipelines):
        if val.score(x_test, y_test) > best_acc:
            best_acc = val.score(x_test, y_test)
            best_pipe = val
            best_clf = idx
    print('Classifier with best accuracy: %s' % pipe_dict[best_clf])


def feature_importances(x, clf):
    """
    Given
    :param x: feature matrix
    :param clf: trained classifier
    we can rank features importance (based on information gain calculation)
     and visualize it in a bar chart.
    """
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    header_sorted=[]
    for f in range(len(list(x))):
        header_sorted.append(list(x)[indices[f]])
        print("%d. Feature: %s (%f)" % (f + 1, list(x)[indices[f]], importances[indices[f]]))

    # Plot the feature importance
    plt.figure()
    plt.title("Feature importance")
    plt.barh(range(x.shape[1]), importances[indices], color="r", align="center")
    plt.yticks(range(x.shape[1]), header_sorted)
    plt.ylim([-1, x.shape[1]])
    plt.savefig('feature_importance_tadpole.png')
    plt.show()

def bar_chart():
    """Visualizes scores recorded from previous computations"""
    url='https://raw.githubusercontent.com/titaristanto/Alzheimer-s-Disease-Prediction/master/Recap.csv'
    df = pd.read_csv(url)
    n_groups=df.shape[0]
    index_df=df.loc[:,['Method(s)']]
    index_name=[index_df.values[i][0] for i in range(index_df.shape[0])]
    train_acc=df.loc[:,['Training Score']]
    dev_acc=df.loc[:,['Dev Score']]
    test_acc=df.loc[:,['Test Score']]
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8
    rects1 = plt.barh(index, train_acc.values, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Training Acc')
    rects2 = plt.barh(index + bar_width, dev_acc.values, bar_width,
                     alpha=opacity,
                     color='orange',
                     label='Dev Acc')

    rects3 = plt.barh(index + bar_width*2, test_acc.values, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Test Acc')
    plt.xlabel('Accuracy')
    plt.ylabel('Methods')
    plt.title('Accuracy Comparison')
    plt.xlim([0,1])
    plt.yticks(index + bar_width, index_name)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
          fancybox=True, shadow=True, ncol=5)
    plt.tight_layout()
    plt.show()
    plt.savefig('alg comparison.png')

def scatterplot_matrix(x,y):
    dat=pd.concat([x,y],axis=1,join='inner')
    fig = ff.create_scatterplotmatrix(dat, diag='histogram', index='Group',
                                  height=800, width=800)
    py.iplot(fig, filename='Histograms along Diagonal Subplots')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('dementia_Tadpole.png')

def main():
    # Initialization
    raw_data=load_data()
    x,y=preprocess_data(raw_data)
    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=.2, random_state=123)
    # pca,lda,x_pca_train,x_lda_train=run_PCA_LDA(x_train,y_train,components=10)
    clf=decision_tree(x_train, y_train)
    # x_lda_test=lda.transform(x_test)
    # x_pca_test=pca.transform(x_test)
    y_pred=clf.predict(x_test)

    # Show confusion Matrix
    cnf_matrix=confusion_matrix(y_test, y_pred)
    #DX: 0: Dementia, 1:MCI to Dementia; 2: MCI; 3: NL
    class_names=list(['Dementia','Normal'])
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')

    # Perform accuracy Calculation
    train_ac_score=accuracy_score(y_train,clf.predict(x_train))
    test_ac_score=accuracy_score(y_test,y_pred)
    print('Training Data Accuracy Score: %1.4f' % train_ac_score)
    print('Test Data Score: %1.4f' % test_ac_score)

    # Feature Importances
    feature_importances(x_train,clf)

    # Perform Comparison using Pipeline
    build_pipe(x_train,y_train,x_test,y_test)

    # Algorithm Comparison
    bar_chart()

    # Tree Visualization
    tree.export_graphviz(clf, out_file = 'tree2.dot', feature_names = list(x_train))
    # open .dot file in a text editor and copy all the text to http://webgraphviz.com to generate a tree structure

if __name__ == '__main__':
    main()


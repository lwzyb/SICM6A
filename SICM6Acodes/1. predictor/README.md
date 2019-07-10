# SICM6A

SICM6A is a deep-learning-based framework to predict m6A sites across species. 

# Dependency
- Python3.7
- numpy
- pytorch 1.0 
- pylab
- sklearn

It is highly recommended to install based on anaconda £¨python 3.7£© 

# model folder

  Five models that have been trained.

# config.txt
  Parameters that the software needs to load£º
- Model file: XXXpklfile =model/...
- Three level threshold (high 95%,midum 90%,low 85%) 
- Content as follows£º

hepg2_brainmodelpklfile=model/HepG2_model_ind_hs_mm_NAGCT_85_8320_29_101.pkl
hepg2_brain_threshold_high=-0.09196107
hepg2_brain_threshold_midum=-0.14587191
hepg2_brain_threshold_low=-0.19978276
cerevisiaemodelpklfile=model/S.cerevisiae_model_ind_-AGCU_200_8170_28_51.pth
cerevisiaethreshold_high=-0.27075472
cerevisiaethreshold_midum=-0.52439857
cerevisiaethreshold_low=-0.66015166
thalianamodelpklfile=model/A.thaliana_model_ind_-AGCU_200_9433_11_101.pth
thalianathreshold_high=-0.1773178
thalianathreshold_midum=-0.59249824
thalianathreshold_low=-0.9193424
Mature_mRNAmodelpklfile=model/MaturemRNA_model_ind_-AGCU_1000_8323_14_251.pth
Mature_mRNAthreshold_high=-1.1005155
Mature_mRNAthreshold_midum=-1.4586651
Mature_mRNAthreshold_low=-1.7554178
Full_transcriptmodelpklfile=model/Fulltranscript_model_ind_-AGCU_400_9122_11_501.pth
Full_transcriptthreshold_high=-1.052967
Full_transcriptthreshold_midum=-1.6455091
Full_transcriptthreshold_low=-2.186074

#Usage:
- Run sicm6apredictor.py in the "spyder" interface of anaconda

#Copyright:
   If you use this software and methods, please cite this paper in your research results:
  Wenzhong Liu.SICM6A:Identifying m6A Site across Species by Transposed GRU Network.(Submitted)



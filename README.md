# final-report
## Introduction
1. The folder data contains the pre-processed dataset which saved RR interval and R-peaks amplitude value extracted from the original ECG signalas as the feature data to train the model.
2. The folder model contains code for training and testing models.
3. The "test.py" in this folder is used to test model performance by using confusion matrix method.
4. The folder result has some of the completed training models, due to github size limitations, not all models areploaded. It is for reference only.
5. The folder "result/Some of the best model" has some of the best models
   If you want to see more models, you can use the following link to download it from baidu web disk. 
   These models also have good performance, and some of their structures are different from the best ones described in the paper.
   
   Link：https://pan.baidu.com/s/1a7wi9eR-LY07RXa0GnyfGw?pwd=m675 
   Extract code：m675 

6. The floder "datapreprocessing" contain the code to pre-processing the ECG data. Run file "ECG_TXT.py" first, then "preprocess.py". Finally, run"dataset.py" to generate the feature data sets.

    To get the original database which is the  "Physionet Apnea-ECG Database", you can download from following link if you can not download it from offcial website.
    
    Link：https://pan.baidu.com/s/11yopWqMkkA8U_UzN8XQZkQ?pwd=6hw6 
    Extract code：6hw6 

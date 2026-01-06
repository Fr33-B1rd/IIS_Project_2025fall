# IIS_Project_2025fall
DM for TRPG

>[!NOTE]
> Mkae sure to add .env file with Gemini key
> `GEMINI_API_KEY=<api_key>`

## Training

```commandline
Loaded 1292 images from ./dataset/DiffusionFER/DiffusionEmotion_S/original
Using 1292 images after label remap.
Extracting DINOv3 features for 1292 images...
Processed 8/1292
Processed 88/1292
Processed 168/1292
Processed 248/1292
Processed 328/1292
Processed 408/1292
Processed 488/1292
Processed 568/1292
Processed 648/1292
Processed 728/1292
Processed 808/1292
Processed 888/1292
Processed 968/1292
Processed 1048/1292
Processed 1128/1292
Processed 1208/1292
Processed 1288/1292
Feature shape: (1292, 384)
Classes: [np.str_('confused'), np.str_('excited'), np.str_('nervous')]
Train: 1033, Test: 259

Accuracy (3-class): 89.19%

Classification Report:
               precision    recall  f1-score   support

           0       0.83      0.93      0.88        70
           1       0.94      0.96      0.95       105
           2       0.89      0.77      0.83        84

    accuracy                           0.89       259
   macro avg       0.89      0.89      0.88       259
weighted avg       0.89      0.89      0.89       259

Confusion Matrix:
 [[ 65   0   5]
 [  1 101   3]
 [ 12   7  65]]
Model saved → dinov3_svm_3class.joblib
```

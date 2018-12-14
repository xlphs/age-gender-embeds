# TensorFlow Age and Gender Estimation using Facenet Embeddings

This is a TensorFlow experiment for age and gender estimation, designed to take Facenet embeddings as input.

## Dataset

### UTKFace

[UTKFace](https://susanqq.github.io/UTKFace/) cropped and aligned, then converted to 512D embeddings.

[Facenet](https://github.com/davidsandberg/facenet) for generating embeddings.

## Usage

### Prepare tfrecords

```
python prepare.py
```

### Train model

Network is defined in `network.py`, it's a small network that trains very quickly.

Train command: `python train.py`

### Evaluate model

Evaluate command: `python test.py`

Output:

```
Age_MAE:6.57, Gender_Acc:88.28%, Loss:3.52
Age_MAE:6.73, Gender_Acc:88.28%, Loss:3.61
Age_MAE:6.02, Gender_Acc:89.06%, Loss:3.44
Age_MAE:5.18, Gender_Acc:87.50%, Loss:3.36
Age_MAE:6.05, Gender_Acc:92.97%, Loss:3.28
Age_MAE:6.13, Gender_Acc:92.97%, Loss:3.29
Age_MAE:6.12, Gender_Acc:90.62%, Loss:3.46
Age_MAE:5.59, Gender_Acc:88.28%, Loss:3.32
Age_MAE:5.80, Gender_Acc:88.28%, Loss:3.43
Age_MAE:4.95, Gender_Acc:91.41%, Loss:3.33
Age_MAE:5.04, Gender_Acc:94.53%, Loss:3.06
Age_MAE:5.52, Gender_Acc:89.06%, Loss:3.36
Age_MAE:5.29, Gender_Acc:89.06%, Loss:3.30
Age_MAE:6.24, Gender_Acc:93.75%, Loss:3.40
Age_MAE:6.84, Gender_Acc:90.62%, Loss:3.55
Summary:
Age_MAE:5.87, Gender_Acc:90.31%, Loss:3.38
```

### Test model with arbitrary features

Put 512D embeddings in a CSV file, then call run.py with `--features` argument pointing to this CSV file.


### TODO

- Train on IMDB-WIKI dataset (UTKFace has 20k faces)
- Try embeddings from models trained with different loss functions
- Use embeddings to estimate race and/or expression


### Reference

- https://github.com/davidsandberg/facenet
- https://github.com/BoyuanJiang/Age-Gender-Estimate-TF

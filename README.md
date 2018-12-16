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
Age_MAE:5.17, Gender_Acc:92.19%, Loss:3.61
Age_MAE:6.75, Gender_Acc:92.97%, Loss:3.95
Age_MAE:6.01, Gender_Acc:88.28%, Loss:4.02
Age_MAE:5.55, Gender_Acc:92.97%, Loss:3.59
Age_MAE:5.06, Gender_Acc:92.19%, Loss:3.51
Age_MAE:6.16, Gender_Acc:89.84%, Loss:3.77
Age_MAE:5.30, Gender_Acc:92.19%, Loss:3.50
Age_MAE:6.09, Gender_Acc:89.84%, Loss:3.82
Age_MAE:5.35, Gender_Acc:94.53%, Loss:3.69
Age_MAE:5.31, Gender_Acc:88.28%, Loss:3.97
Age_MAE:5.90, Gender_Acc:90.62%, Loss:3.70
Summary:
Age_MAE:5.70, Gender_Acc:91.26%, Loss:3.74
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

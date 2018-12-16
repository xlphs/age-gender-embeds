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
Age_MAE:4.93, Gender_Acc:91.41%, Loss:4.09
Age_MAE:6.54, Gender_Acc:93.75%, Loss:4.40
Age_MAE:6.31, Gender_Acc:91.41%, Loss:4.60
Age_MAE:5.23, Gender_Acc:92.97%, Loss:4.19
Age_MAE:5.08, Gender_Acc:91.41%, Loss:4.09
Age_MAE:5.95, Gender_Acc:89.84%, Loss:4.42
Age_MAE:4.85, Gender_Acc:93.75%, Loss:4.03
Age_MAE:5.68, Gender_Acc:89.84%, Loss:4.14
Age_MAE:5.22, Gender_Acc:89.84%, Loss:4.38
Age_MAE:5.26, Gender_Acc:86.72%, Loss:4.41
Age_MAE:5.94, Gender_Acc:89.84%, Loss:4.13
Summary:
Age_MAE:5.54, Gender_Acc:90.98%, Loss:4.26
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

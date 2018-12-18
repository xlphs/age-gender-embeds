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
Age_MAE:6.08, Gender_Acc:92.97%, Loss:3.22
Age_MAE:5.28, Gender_Acc:93.75%, Loss:3.13
Age_MAE:5.94, Gender_Acc:92.19%, Loss:3.13
Age_MAE:5.15, Gender_Acc:95.31%, Loss:3.10
Age_MAE:5.70, Gender_Acc:95.31%, Loss:3.19
Age_MAE:5.38, Gender_Acc:92.97%, Loss:3.39
Age_MAE:5.30, Gender_Acc:88.28%, Loss:3.38
Age_MAE:4.87, Gender_Acc:89.84%, Loss:3.08
Age_MAE:4.60, Gender_Acc:86.72%, Loss:3.14
Age_MAE:4.81, Gender_Acc:91.41%, Loss:3.07
Age_MAE:6.53, Gender_Acc:92.97%, Loss:3.30
Summary:
Age_MAE:5.42, Gender_Acc:91.97%, Loss:3.19
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

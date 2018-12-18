# TensorFlow Age and Gender Estimation using Facenet Embeddings

This is a TensorFlow experiment for age and gender estimation, designed to take Facenet embeddings as input.

## Dataset

### UTKFace

[UTKFace](https://susanqq.github.io/UTKFace/) cropped and aligned, then converted to 512D embeddings.

[Facenet](https://github.com/davidsandberg/facenet) for generating embeddings.

## Usage

### Prepare tfrecords

Remove faces under 10 years old. Because faces of very young people are hard even for human to identify, especially when faces are tightly cropped, also because facenet pretrained models were not optimized for such faces.

```
python prepare.py
```

### Train model

Network is defined in `network_conv.py`, it's a small network that trains very quickly.

Train command: `python train.py`

### Evaluate model

Evaluate command: `python test.py`

Output:

```
Age_MAE:6.36, Gender_Acc:94.53%, Loss:3.53
Age_MAE:4.77, Gender_Acc:96.88%, Loss:3.36
Age_MAE:5.76, Gender_Acc:96.09%, Loss:3.29
Age_MAE:6.21, Gender_Acc:94.53%, Loss:3.41
Age_MAE:4.96, Gender_Acc:97.66%, Loss:3.07
Age_MAE:6.16, Gender_Acc:96.09%, Loss:3.51
Age_MAE:5.91, Gender_Acc:96.09%, Loss:3.38
Age_MAE:5.71, Gender_Acc:100.00%, Loss:3.18
Age_MAE:5.68, Gender_Acc:96.09%, Loss:3.33
Age_MAE:5.51, Gender_Acc:97.66%, Loss:3.15
Age_MAE:5.90, Gender_Acc:95.31%, Loss:3.33
Summary:
Age_MAE:5.72, Gender_Acc:96.45%, Loss:3.3
```

### Test model with arbitrary features

Put 512D embeddings in a CSV file, then call run.py with `--features` argument.


### TODO

- Try embeddings from models trained with different loss functions
- Use embeddings to estimate race and/or expression


### Reference

- https://github.com/davidsandberg/facenet
- https://github.com/BoyuanJiang/Age-Gender-Estimate-TF

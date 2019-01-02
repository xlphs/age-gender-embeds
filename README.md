# TensorFlow Age and Gender Estimation using Facenet Embeddings

This is a TensorFlow experiment for age and gender estimation, designed to take Facenet embeddings as input. Note the pretrained model uses softmax loss.

## LFW Gender Accuracy

There's manually verified LFW gender labels from [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)

My test result:

```
Male 0.9854, correct 10118, wrong 138
Female 0.9535, correct 2828, wrong 149
```

## Dataset

### UTKFace

[UTKFace](https://susanqq.github.io/UTKFace/) cropped and aligned, then converted to 512D embeddings, you should use embeddings from a model trained with softmax loss.

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

Although UTKFaces are tightly cropped, I have tested with 5% margin for LFW gender and accuracy is not affected.

### TODO

- Try embeddings from models trained with different loss functions
   - Tried ArcFace and it is not suitable for this task
- Use embeddings to estimate race and/or expression


### Reference

- https://github.com/davidsandberg/facenet
- https://github.com/BoyuanJiang/Age-Gender-Estimate-TF

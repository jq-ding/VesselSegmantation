# VesselSegmantation

## A Multi-channel Deep Neural Network for Retina Vessel Segmentation via a Fusion Mechanism
![frame](https://user-images.githubusercontent.com/93422935/139528410-7a066942-c5c5-4590-91d0-4bda14ef8e07.png)

In this study, we propose multi-channel deep neural network for retina vessel segmentation. First, we apply U-net on original and thin(or thick) vessels for multi-objective optimization for purposively training thick and thin vessels. Then, we design a specific fusion mechanism for combining three kinds of prediction probability maps into final binary segmentation map. Experiments show that our method can effectively improve the segmentation performances of thin blood vessels and vascular ends. It outperforms the many current excellent vessel segmentation methods on three public datasets. In particular, it is pretty impressive that we achieve best F1-score with 0.8201 on the DRIVE dataset and 0.8239 on the STARE dataset. The findings of this study have the potential for application in automated retinal image analysis, and it may provide a new, general and high-performance computing framework for image segmentation.

### Data

DRIVE  https://drive.grand-challenge.org/

STARE https://cecas.clemson.edu/~ahoover/stare/

IOSTAR http://www.retinacheck.org/

### Preparing data

```
python prepare_datasets.py
```

### Preparing the label of thin and thick vessels

```
python Thin_ThickLabel.py
```

### Training

Training using labels for original, thick and thin vessels, respectively

```
python training.py
```

### Testing

Predict original, thick and thin vessels separately and save the results

```
python predict.py
```

### Fusion

```
python merge.py
```

### Calculate metrics

```
python pixel.py
```

### Results

<img src="https://user-images.githubusercontent.com/93422935/139528371-5923290d-0b0f-4ec5-8071-f8065194bcc5.png" width="50%" height="50%">


<img src="https://user-images.githubusercontent.com/93422935/139528399-feb5aa93-1f05-496f-bb69-ea0bfbd6836c.png" width="50%" height="50%">


# VesselSegmantation

## [A Multi-channel Deep Neural Network for Retina Vessel Segmentation via a Fusion Mechanism](https://www.frontiersin.org/articles/10.3389/fbioe.2021.697915/full)

![frame](https://user-images.githubusercontent.com/93422935/139528410-7a066942-c5c5-4590-91d0-4bda14ef8e07.png)

In this study, we propose multi-channel deep neural network for retina vessel segmentation. First, we apply U-net on original and thin(or thick) vessels for multi-objective optimization for purposively training thick and thin vessels. Then, we design a specific fusion mechanism for combining three kinds of prediction probability maps into final binary segmentation map. Experiments show that our method can effectively improve the segmentation performances of thin blood vessels and vascular ends. It outperforms the many current excellent vessel segmentation methods on three public datasets. In particular, it is pretty impressive that we achieve best F1-score with 0.8201 on the DRIVE dataset and 0.8239 on the STARE dataset. The findings of this study have the potential for application in automated retinal image analysis, and it may provide a new, general and high-performance computing framework for image segmentation.

### Data

DRIVE  https://drive.grand-challenge.org/

STARE https://cecas.clemson.edu/~ahoover/stare/

IOSTAR http://www.retinacheck.org/

### Preparing data

```
# transform images to .hdf5 files

python prepare_datasets.py
```

### Preparing the label of thin and thick vessels

```
# including extraction of vessel skeleton and separation of thin and thick vessels
python Thin_ThickLabel.py
```
<img src="https://user-images.githubusercontent.com/93422935/139536170-480900ed-40c4-4e43-ba38-096bef642fea.png" width="50%" height="50%">


### Training

Training using labels for original, thick and thin vessels, respectively

```
python training.py

# using focal loss
def catergorical_focal_loss(gamma=2.0, alpha=0.25):
        """
        Formula:
            loss = -alpha*((1-p_t)^gamma)*log(p_t)
        Parameters:
            alpha -- the same as wighting factor in balanced cross entropy
            gamma -- focusing parameter for modulating factor (1-p)
        Default value:
            gamma -- 2.0 as mentioned in the paper
            alpha -- 0.25 as mentioned in the paper
        """

        def focal_loss(y_true, y_pred):

            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
            # Calculate cross entropy
            cross_entropy = -y_true * K.log(y_pred)

            # Calculate weight that consists of modulating factor and weighting factor
            weight = alpha * y_true * K.pow((1 - y_pred), gamma)
            # Calculate focal loss
            loss = weight * cross_entropy
            # Sum the losses in mini_batch
            loss = K.sum(loss, axis=1)
            return loss

        return focal_loss
```

### Testing

Predict original, thick and thin vessels separately and save the results

```
# test and calculate the AUC within the mask
python predict.py
```

### Fusion

```
python merge.py

def fusion2(raw_img, thick, thin):
    rowNum = len(thick)    
    colNum = len(thick[0]) 

    newImg = np.zeros((rowNum, colNum), np.uint8)

    for i in range(rowNum):
        for j in range(colNum):
            if thick[i][j] > 0.5 or thin[i][j] > 0.5 or raw_img[i][j] > 0.5:
                newImg[i][j] = 255
    return newImg
```

### Calculate metrics

```
# Acc,sp,se,F1
python pixel.py
```

### Results
#### Compare different loss functions

<img src="https://user-images.githubusercontent.com/93422935/139536046-acdc644a-4612-4df1-8f6b-60f3c34ea425.png" width="50%" height="50%">

#### Cross-training results on DRIVE and STARE datasets

<img src="https://user-images.githubusercontent.com/93422935/139536086-65263177-5644-4152-b0e4-c2733c8cd66f.png" width="50%" height="50%">

#### Segmentation result

<img src="https://user-images.githubusercontent.com/93422935/139528371-5923290d-0b0f-4ec5-8071-f8065194bcc5.png" width="50%" height="50%">

#### Compare with others

<img src="https://user-images.githubusercontent.com/93422935/139528399-feb5aa93-1f05-496f-bb69-ea0bfbd6836c.png" width="50%" height="50%">


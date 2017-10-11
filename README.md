# AdaptiveAttention
Pytorch Implementation of Adaptive Attention Model for Image Captioning

Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning [[Paper](https://arxiv.org/abs/1612.01887)] [[Review](https://github.com/yufengm/Papers/blob/master/reviews/lu2016knowing.md)]

**Dataset Preparation**

First we will need to download the MS-COCO dataset. So create a data folder and run the download bash script

``` bash
mkdir data && ./download.sh
```

Afterwards, we should create the Karpathy split for training, validation and test.

``` bash
python KarpathySplit.py
```

Then we can build the vocabulary by running 

```
python build_vocab.py
```

The vocab.pkl should be saved in the data folder.

Now we will need to resize all the images in both train and val folder. Here I create a new folder under data, i.e., 'resized'. Then we may run resize.py to resize all images into 256 x 256. You may specify different locations inside resize.py

```
mkdir data/resized && python resize.py
```

After all images are resized. Now we can train our Adaptive Attention model with 

```
python train.py
```

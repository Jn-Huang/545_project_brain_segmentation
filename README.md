# 545 Project: Brain Segmentation



## Install (in a conda environment preferably)

```
pip install -r requirements.txt
```



## Dataset

Dataset used for development and evaluation was made publicly available on Kaggle: [kaggle.com/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation). It contains MR images from [TCIA LGG collection](https://wiki.cancerimagingarchive.net/display/Public/TCGA-LGG) with segmentation masks approved by a board-certified radiologist at Duke University.

Please download the dataset and place it as `./kaggle`.



## Pipeline

![pipeline](/Users/jinhuang/Desktop/pipeline.png)

- Preprocessing: this step is done by function `data_loaders` in `dataset.py`. Load as follows

  ```
  loader_train, loader_valid = data_loaders(batch_size=16, workers=2, image_size=224, aug_scale=0.05, aug_angle=15)
  ```

- Segmentation: We will try different models for segmentation
- Postprocessing: 

## References

https://github.com/mateuszbuda/brain-segmentation-pytorch


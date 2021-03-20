## Baseball Ball Detection with yolov5

preprocessing.ipynb script allows me to create the desired data/folder structure to use train.py

    
    ├── train_data                  
      ├── images                    
         ├── train                    
         └── val                 

      ├── labels                    
         ├── train
         └── val


I saved the google colab that I worked as a .ipynb, you can check baseball_ball_detection_w_yolov5.ipynb for that.


Here are some of the results of fine-tuning yolov5.

| ![14.jpg](14.jpg) | 
|:--:| 
| *Ball Detection 1* |


| ![22.jpg](22.jpg) | 
|:--:| 
| *Ball Detection 2* |



| ![43.jpg](43.jpg) | 
|:--:| 
| *Ball Detection 3* |


| ![9.jpg](9.jpg) | 
|:--:| 
| *Ball Detection 4* |

| ![metrics.png](metrics.png) | 
|:--:| 
| *Results* |

More results at results.zip

## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```


## Tutorials
* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [ONNX and TorchScript Export](https://github.com/ultralytics/yolov5/issues/251)
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)


## Environments
- **Google Colab and Kaggle** notebooks with free GPU: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>

## Training
Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --img 640 --batch 16 --epochs 50 --data baseball.yaml --weights yolov5s.pt --nosave --cache
```


## More information about inference, exporting, plotting results & metrics can be found at  [ultralytics/yolov5](https://github.com/ultralytics/yolov5). The ultralyrics's shared google colab makes everything so much easier, it's easy to follow.


## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)

# UMNet
The Pytorch implementation of CVPR2022 paper “Multi-Source Uncertainty Mining for Deep Unsupervised Saliency Detection”

# Trained Model，Test Data and Results

Please download the trained model, test data and SOD results from [Baidu Cloud](https://pan.baidu.com/s/10YDn5tiexLx4iUjE8zP4wg?pwd=tmzw) (password: tmzw).



# Requirement
•	Python 3.7

•	PyTorch 1.6.1

•	torchvision

•	numpy

•	Pillow

•	Cython

# Run
1. Please download the [trained model](https://pan.baidu.com/s/10YDn5tiexLx4iUjE8zP4wg?pwd=tmzw) and [test datasets](https://pan.baidu.com/s/10YDn5tiexLx4iUjE8zP4wg?pwd=tmzw) (including DUTS-TE, OMRON, ECSSD, and  HKU-IS). Uncompress and put them in the current file.
3. Set the path of testing sets and trained model in [config.py](https://github.com/yifanw90/UMNet/blob/main/config.py). The default setting can be in [config.py](https://github.com/yifanw90/UMNet/blob/main/config.py).
4. Run [main.py](https://github.com/yifanw90/UMNet/blob/main/main.py) to obtain the predicted saliency maps. The results are saved in the save_path (see [config.py](https://github.com/yifanw90/UMNet/blob/main/config.py)). You can also download our saliency results from [Baidu Cloud](https://pan.baidu.com/s/10YDn5tiexLx4iUjE8zP4wg?pwd=tmzw).
4. Run [compute_score.py](https://github.com/yifanw90/UMNet/blob/main/compute_score.py) to obtain the evaluation scores of the predictions in terms of MAE, Fmax, Sm, and Em.  The evaluation codes are referred from  https://github.com/Xiaoqi-Zhao-DLUT/GateNet-RGB-Saliency.  
5. Please be sure that the paths of ground truth and predictions are valid in [compute_score.py](https://github.com/yifanw90/UMNet/blob/main/compute_score.py).


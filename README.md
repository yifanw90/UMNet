# UMNet
The Pytorch implementation of CVPR2022 paper [Multi-Source Uncertainty Mining for Deep Unsupervised Saliency Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Multi-Source_Uncertainty_Mining_for_Deep_Unsupervised_Saliency_Detection_CVPR_2022_paper.pdf)

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


# Train
Note: Our method is trained mainly following the same setting of [DeepUSPS](https://github.com/sally20921/DeepUSPS). We use MSRA-B 2500 training data for network training.
1. Four traditional SOD methods including MC, HS, DSR, and RBD are adopted to generate pseudo labels for the training data, which are refined using the first stage of DeepUSPS. 
2. The four kinds of refined pseudo labels are used for multi-source network learning using our [training code](https://pan.baidu.com/s/18Xsq7MJ_hCNNCLM5Eyxt0g?pwd=a4hh) (extract code: a4hh).

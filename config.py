from easydict import EasyDict as edict
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '4'

config = edict()

### Input
config.input_size = [320, 320]
## PIL Image.open
config.input_mean = [0.49227863, 0.46342391, 0.39742668]
config.input_std = [0.22730138, 0.22451538, 0.22985159] 

#config.snapshot = 'models/UMNet_trained.pth'
config.model_path = 'trained_model/UMNet.pth'

config.data_path = 'Test_Data'

config.datasets = ['DUTS-TE', 'OMRON', 'ECSSD', 'HKU-IS']  #dataset name
#config.datasets = ['DUTS-TE']

config.save_path = 'result_UMNet'   # save path

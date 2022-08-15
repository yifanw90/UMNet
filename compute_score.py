import numpy as np
from libs.util import eval_dataset
from libs.saliency_metric import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm
import os

data_path_gt = 'Test_Data'
data_path_pre = 'result_UMNet'  ##pre_salmap_path


datasets = ['DUTS-TE', 'OMRON', 'ECSSD', 'HKU-IS']  ##test_datasets_name
#datasets = ['DUTS-TE']

for dataset in datasets:
    gt_path = os.path.join(data_path_gt, dataset, 'gt')
    pre_path = os.path.join(data_path_pre, dataset)
    test_loader = eval_dataset(pre_path, gt_path)
    mae, fm, sm, em = cal_mae(), cal_fm(test_loader.size), cal_sm(), cal_em()
    
    for i in range(test_loader.size):
        print('Computing scores for %d / %d' % (i + 1, test_loader.size))
        sal, gt = test_loader.load_data()
        #assert sal.size == gt.size
        if sal.size != gt.size:
            x, y = gt.size
            sal = sal.resize((x, y))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)  # convert gt from [0, 255] to [0,1]
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0  # binarize gt with a threthhold of 0.5
        res = sal
        res = np.array(res)
        if res.max() == res.min():
            res = res / 255.0
        else:
            res = (res - res.min()) / (res.max() - res.min())  # convert res to [0,1]
        mae.update(res, gt)
        sm.update(res, gt)
        fm.update(res, gt) 
        em.update(res, gt)

    MAE = mae.show()
    maxf, _, _, _ = fm.show()
    sm = sm.show()
    em = em.show()                                                                                    
    print('dataset: {} MAE: {:.4f} maxF: {:.4f}  Sm: {:.4f} Em: {:.4f}'.format(dataset, MAE, maxf, sm, em))

import os
import argparse
import yaml
import time
import datetime
import json
import torch
import numpy as np

from pathlib import Path
from solver import Solver
from data_loader import get_loader
from make256 import makePaths, restoreImage
from tps_transformation import tps_transform

from tqdm.auto import tqdm
from torch.backends import cudnn
from torch.utils import data
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from utils import elastic_transform

class FinetuneDataSet(data.Dataset):
    def __init__(self, config, img_transform_gt, img_transform_sketch):
        self.img_transform_gt = img_transform_gt
        self.img_transform_sketch = img_transform_sketch
        self.img_size = (config['MODEL_CONFIG']['IMG_SIZE'], config['MODEL_CONFIG']['IMG_SIZE'], 3)
        self.augment = config['TRAINING_CONFIG']['AUGMENT']

        self.dist = config['TRAINING_CONFIG']['DIST']
        if self.dist == 'uniform':
            self.a = config['TRAINING_CONFIG']['A']
            self.b = config['TRAINING_CONFIG']['B']
        else:
            self.mean = config['TRAINING_CONFIG']['MEAN']
            self.std = config['TRAINING_CONFIG']['STD']
        self.ref_list = []
        self.skt_list = []
        self.ref_cache = {}
        self.skt_cache = {}
        self.base_dir = config['FINETUNE_CONFIG']['DATA_DIR']

        try:
            with open(os.path.join(config['FINETUNE_CONFIG']['FINETUNE_DIR'],'profile.json'),'r') as fin:
                setting = json.load(fin)
        except:
            print("JSON load Erorr:", os.path.join(config['FINETUNE_CONFIG']['FINETUNE_DIR'],'profile.json'))
            return

        train_list, test_list = { 'skt':[], 'ref':[], }, {'skt':[], 'ref':[],}
        for key, val in setting['references'].items():
            skt_path, ref_path = makePaths(val)
            train_list['skt'].append(os.path.join(self.base_dir,skt_path))
            train_list['ref'].append(os.path.join(self.base_dir,ref_path))
        for key, val in setting['tests'].items():
            skt_path, ref_path = makePaths(val)
            test_list['skt'].append(os.path.join(self.base_dir,skt_path))
            test_list['ref'].append(os.path.join(self.base_dir,ref_path))
        self.ref_list = train_list['ref']
        self.skt_list = train_list['skt']
        self.test_list = test_list
        
    def __getitem__(self, index):
        if index not in self.ref_cache.keys(): 
            self.ref_cache[index] = Image.open(self.ref_list[index]).convert('RGB')
        if index not in self.skt_cache.keys(): 
            self.skt_cache[index] = Image.open(self.skt_list[index]).convert('L')
        reference, sketch = self.ref_cache[index], self.skt_cache[index]

        if self.dist == 'uniform':
            noise = np.random.uniform(self.a, self.b, np.shape(reference))
        else:
            noise = np.random.normal(self.mean, self.std, np.shape(reference))

        reference = np.array(reference) + noise
        reference = Image.fromarray(reference.astype('uint8'))

        if self.augment == 'elastic':
            augmented_reference = elastic_transform(np.array(reference), 1000, 8, random_state=None)
            augmented_reference = Image.fromarray(augmented_reference)
        elif self.augment == 'tps':
            augmented_reference = tps_transform(np.array(reference))
            augmented_reference = Image.fromarray(augmented_reference)
        else:
            augmented_reference = reference

        return os.path.split(self.ref_list[index])[1], self.img_transform_gt(augmented_reference),\
               self.img_transform_gt(reference), self.img_transform_sketch(sketch)

    def __len__(self):
        """Return the number of images."""
        return len(self.ref_list)


class TestDataSet(data.Dataset):
    def __init__(self, config, img_transform_gt, img_transform_sketch):
        self.config = config
        self.img_transform_gt = img_transform_gt
        self.img_transform_sketch = img_transform_sketch
        self.img_size = (config['MODEL_CONFIG']['IMG_SIZE'], config['MODEL_CONFIG']['IMG_SIZE'], 3)
        self.augment = config['TRAINING_CONFIG']['AUGMENT']

        self.dist = config['TRAINING_CONFIG']['DIST']
        if self.dist == 'uniform':
            self.a = config['TRAINING_CONFIG']['A']
            self.b = config['TRAINING_CONFIG']['B']
        else:
            self.mean = config['TRAINING_CONFIG']['MEAN']
            self.std = config['TRAINING_CONFIG']['STD']
        self.base_dir = config['FINETUNE_CONFIG']['DATA_DIR']
        self.ref_cache = {}
        self.skt_cache = {}
        self.ref_list, self.skt_list = [], []

        try:
            with open(os.path.join(config['FINETUNE_CONFIG']['FINETUNE_DIR'],'profile.json'),'r') as fin:
                setting = json.load(fin)
        except:
            print("JSON load Erorr:", os.path.join(config['FINETUNE_CONFIG']['FINETUNE_DIR'],'profile.json'))
            return

        for key, val in setting['references'].items():
            _, ref_path = makePaths(val)
            self.ref_list.append(os.path.join(self.base_dir,ref_path))
        for key, val in setting['tests'].items():
            skt_path, _ = makePaths(val)
            self.skt_list.append((os.path.join(self.base_dir,skt_path), setting['patch_settings'][skt_path]))
        
    def __getitem__(self, index):
        step = len(self.skt_list)
        if index//step not in self.ref_cache.keys(): 
            self.ref_cache[index//step] = self.img_transform_gt(Image.open(self.ref_list[index//step]).convert('RGB'))
        if index%step not in self.skt_cache.keys(): 
            self.skt_cache[index%step] = self.img_transform_sketch(Image.open(self.skt_list[index%step][0]).convert('L'))
            
        return (self.ref_cache[index//step], self.skt_cache[index%step],
                os.path.splitext(os.path.basename(self.skt_list[index%step][0]))[0]+('-ref%03d'%(index//step)), # fID
                self.skt_list[index%step][1] # patchSetting
               )
    
    def __len__(self):
        """Return the number of images."""
        return len(self.ref_list)*len(self.skt_list)

def get_test_loader(config):
    img_transform_gt = list()
    img_transform_sketch = list()
    img_size = 256

    img_transform_gt.append(T.Resize((img_size, img_size)))
    img_transform_gt.append(T.ToTensor())
    img_transform_gt.append(T.Normalize(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform_gt = T.Compose(img_transform_gt)

    img_transform_sketch.append(T.Resize((img_size, img_size)))
    img_transform_sketch.append(T.ToTensor())
    img_transform_sketch.append(T.Normalize(mean=0.5, std=0.5))
    img_transform_sketch = T.Compose(img_transform_sketch)

    dataset = TestDataSet(config, img_transform_gt, img_transform_sketch)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  num_workers=1)
    return data_loader

def image_save(gen, fid, sample_dir):
    sample_path = sample_dir + '/' + fid + '.png'
    save_image(denorm(gen.data.cpu()), sample_path, nrow=1, padding=0)

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def finetune(config,base_epoch=-1):
    cudnn.benchmark = True
    device = torch.device(config['TRAINING_CONFIG']['GPU'])
    timeFT_log = {}
    timeTT_log = {}

    for d in tqdm(sorted(Path(config['FINETUNE_CONFIG']['DATA_DIR']).iterdir())):
        if d.is_file() or not os.path.exists(os.path.join(d,'profile.json')): continue
        config['FINETUNE_CONFIG']['FINETUNE_DIR'] = d
        
        test_loader = get_test_loader(config)
        
        for k, fine_epoch in enumerate(config['FINETUNE_CONFIG']['FINETUNE_EPOCHS']):
            if k==0 or config['FINETUNE_CONFIG']['FINETUNE_EPOCHS'][k-1]>fine_epoch:
                solver = Solver(config, get_loader(config,FinetuneDataSet))
                base_epoch = solver.restore_model(base_epoch)
                solver.epoch = base_epoch+fine_epoch
                time_offset = 0
            else: # resume finetuning
                prev_epoch = config['FINETUNE_CONFIG']['FINETUNE_EPOCHS'][k-1]
                base_epoch = solver.epoch
                solver.epoch = base_epoch+fine_epoch-prev_epoch
                time_offset = timeFT_log[(d,prev_epoch)]
            
            solver.sample_dir = os.path.join(d,config['FINETUNE_CONFIG']['SAMPLE_DIR'])
            # solver.log_dir = os.path.join(d,config['FINETUNE_CONFIG']['LOG_DIR'])
            # solver.result_dir = os.path.join(d,config['FINETUNE_CONFIG']['RESULT_DIR'])
            solver.log_dir = os.path.join(config['FINETUNE_CONFIG']['DATA_DIR'])
            solver.result_dir = os.path.join(config['FINETUNE_CONFIG']['DATA_DIR'],config['FINETUNE_CONFIG']['RESULT_DIR'],
                                             config['TRAINING_CONFIG']['TRAIN_DIR'],'ft%03d'%fine_epoch,os.path.split(d)[-1])
            
            # solver.model_dir = os.path.join(d,config['FINETUNE_CONFIG']['MODEL_DIR'])
            for tmp in (solver.log_dir, solver.sample_dir, solver.result_dir, solver.model_dir):
                os.makedirs(tmp,exist_ok=True)

            start_time = time.time()
            solver.train(store_checkpoint=False, store_sample=False, resume_epoch=base_epoch, load_model=(time_offset==0))
            et = time.time() - start_time + time_offset
            print('{}: Epoch {} finetuning is finished. [{}]'.format(d,fine_epoch,str(datetime.timedelta(seconds=et))[:-7]))
            timeFT_log[(d,fine_epoch)] = et
            
            start_time = time.time()
            with torch.no_grad():
                for idx, (ref,skt, fID, patchSetting) in enumerate(test_loader):
                    ref = ref.to(device)
                    skt = skt.to(device)
                    result, _, _ = solver.G.predict(ref, skt)
                    result_org = restoreImage(result, *patchSetting)
                    image_save(result_org,
                               #torch.concat([ref,torch.tile(skt,dims=(1,3,1,1)),result],axis=-1),
                               fID[0],solver.result_dir)
            et = time.time() - start_time
            print('{}: Epoch {} testing is finished. [{}]'.format(d,fine_epoch,str(datetime.timedelta(seconds=et))[:-7]))
            timeTT_log[(d,fine_epoch)] = et
            
    with open(os.path.join(solver.log_dir, 'time.log'), 'w') as fout:
        json.dump({'finetuning':{str(k):str(datetime.timedelta(seconds=t))[:-7] for k, t in timeFT_log.items()}, 
                   'testing':{str(k):str(datetime.timedelta(seconds=t))[:-7] for k, t in timeTT_log.items()}, },
                  fout, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml', help='specifies config yaml file')

    params = parser.parse_args()

    if os.path.exists(params.config):
        config = yaml.load(open(params.config, 'r'), Loader=yaml.FullLoader)
        # make_train_directory(config)
        finetune(config, 40)
    else:
        print("Please check your config yaml file")

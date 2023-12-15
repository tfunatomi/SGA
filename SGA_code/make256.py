import cv2
import json
import glob
import numpy as np
import os 
import sys
import torch
from torchvision import transforms as T
from tqdm.auto import tqdm
from PIL import Image

sys.path.append('../../AnimeStyleFewShotColorizationDemo/python')
from fewshot_anime_colorization.color_def import Color
from fewshot_anime_colorization.palette import Palette

IGNORE_COLORS = [[0,0,0], [30,30,30],[0,0,255],[0,255,0],[255,0,0],[255,255,255],[48,35,35]]

def get_unique_colors(image, prefs):
    if prefs:
        ignore_colors = [Color.WHITE]
        ignore_colors.append(prefs.trace.line_color)
        ignore_colors.append(prefs.trace.shadow_direction)
        ignore_colors.append(prefs.trace.highlight_direction)
        ignore_colors.append(prefs.trace.interfer_direction)
        ignore_colors.append(prefs.paint.line_color)
    else:
        ignore_colors = IGNORE_COLORS

    uniques = np.unique(image.reshape(-1, image.shape[-1]), axis=0).tolist()
    for col in ignore_colors:
        if col in uniques:
            uniques.remove(col)
    return uniques

def resizeImagePair(trace_image,paint_image,IMG_SIZE=256,IMG_Ratio=1.1):
    roi = []
    for d in [1, 0]:
        idx = np.where((trace_image<255).any(axis=d))[0]
        roi.append([int(idx.min()),int(idx.max())])
    roiSize = np.array(roi)[:,1]-np.array(roi)[:,0]
    maxSize = int(roiSize.max()*IMG_Ratio)
    topLeft = (maxSize-roiSize)//2

    ret = {}
    canvasSlice = ((int(topLeft[0]),int(roiSize[0]+topLeft[0])),
                   (int(topLeft[1]),int(roiSize[1]+topLeft[1])))
    for key,img in {'skt':trace_image, 'ref':paint_image,}.items():
        canvas = np.full((maxSize,maxSize,img.shape[2]),255,dtype=np.uint8)
        canvas[slice(*canvasSlice[0]),slice(*canvasSlice[1])] = img[slice(*roi[0]),slice(*roi[1])]
        img = cv2.resize(canvas,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA)
        if (key=='skt'): img = (img==255)*255
        ret[key] = img
    return ret['skt'], ret['ref'], (trace_image.shape,roi,maxSize,canvasSlice)

def makePaths(paint_path):
    skt_path = paint_path.replace('PAINT','skt').replace('tga','png')
    ref_path = paint_path.replace('PAINT','ref').replace('tga','png')
    return skt_path, ref_path

def restoreImage(img,original_shape,roiSlice,maxSize,canvasSlice):
    original_shape, roiSlice, canvasSlice = torch.tensor(original_shape).tolist(), torch.tensor(roiSlice).tolist(), torch.tensor(canvasSlice).tolist()
    canvas = T.Resize(int(maxSize))(img)
    orig_img = torch.ones(list(canvas.shape[:2])+original_shape[:2],dtype=img.dtype,device=img.device)
    orig_img[...,slice(*roiSlice[0]),slice(*roiSlice[1])] = canvas[...,slice(*canvasSlice[0]),slice(*canvasSlice[1])]
    return orig_img

if __name__ == '__main__':
    basedir = './datasetB'
    prefs = None

    for d in tqdm(sorted(os.listdir(basedir))):
        target_dir = os.path.join(basedir, d)
        if os.path.isfile(target_dir) or target_dir.find('.ipynb_checkpoints')!=-1:
            continue

        refs = {}
        tests = {}
        patches = {}
        palette_colors = []

        query = f'{target_dir}/PAINT/*/*.tga'
        for f, paint_path in enumerate(sorted(glob.glob(query))):
            paint_image = Image.open(paint_path).convert('RGB')
            paint_image = cv2.cvtColor(np.array(paint_image), cv2.COLOR_RGB2BGR)

            trace_path = paint_path.replace('PAINT','TRACE')
            trace_image = np.array(Image.open(trace_path).convert('RGB'))[...,::-1]

            unique_colors = get_unique_colors(paint_image, prefs)
            palette_colors.extend(unique_colors)

            useful_for_ref = True if len(unique_colors) > 0 else False
            if useful_for_ref:
                refs[f] = paint_path
            else:
                tests[f] = paint_path

            skt_path = paint_path.replace('PAINT','skt').replace('tga','png')
            ref_path = paint_path.replace('PAINT','ref').replace('tga','png')
            resizedImg = resizeImagePair(trace_image,paint_image)

            os.makedirs(os.path.split(skt_path)[0],exist_ok=True)
            os.makedirs(os.path.split(ref_path)[0],exist_ok=True)
            cv2.imwrite(skt_path,resizedImg[0])
            cv2.imwrite(ref_path,resizedImg[1])
            patches[ref_path] = resizedImg[2]
            patches[skt_path] = resizedImg[2]

        # make palette
        palette_colors.append(Color.BLACK)
        palette_colors.append(Color.WHITE)
        palette_colors = np.unique(np.array(palette_colors), axis=0).tolist()
        pal = Palette()
        print(d, 'palette_colors {}'.format(palette_colors))
        pal.from_array(palette_colors)
        pal.save(os.path.join(target_dir, 'palette.csv'))

        pack = {'references': refs, 'tests': tests, 'patch_settings': patches,}
        with open(os.path.join(target_dir, 'profile.json'), 'w') as fout:
            json.dump(pack, fout, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))


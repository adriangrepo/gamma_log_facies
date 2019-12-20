#!/usr/bin/env python
# coding: utf-8

# # Image segmentaion with masked facies
# 
# Squish rectangular images to square
# 
# Using mask images processed in 01_02_mask_processing
# 
# 550x550 size - images cropped - generated in 01_03_image_mask_cropping



import sys
from numbers import Integral
from random import uniform
from PIL import Image as pil_image
import fastai
from fastai.vision import *
from fastai.vision import Image
from fastai.vision.transform import _minus_epsilon
from fastai.vision.data import SegmentationProcessor
from fastai.vision.interpret import SegmentationInterpretation
from mask_functions import *
from collections import defaultdict
import cv2
from IPython.display import display 
import datetime
import uuid
from scipy.stats import mode




fastai.__version__




torch.cuda.set_device(1)




DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]
print(f'UID: {UID}, DATE: {DATE}')    




UID='d4cb7213'
DATE='20191217'
NB='02_07_3'




SUBSET_DATA=False
SUBSET_LEN=171




tgt_height = 550
data_dir = Path('data')
train_images = data_dir/'train_images'
train_path = train_images/'cropped/mask_fill/train'
train_mask = train_path/'masks'
train_img = train_path/'images'

test_img = train_images/'cropped/mask_fill/test'




filename = 'data/CAX_LogFacies_Train_File.csv'




file_test = 'data/CAX_LogFacies_Test_File.csv'


# #### all data



training_data = pd.read_csv(filename)
training_data.head()
training_data['well_file']='well_'+training_data['well_id'].astype(str)+'.png'
wells=training_data['well_file'].unique()
all_wells_df=pd.DataFrame(wells)
all_wells_df.head()
    




test_df = pd.read_csv(file_test)
test_df.head()


# ### Data



train_img_names = get_image_files(train_img)
len(train_img_names)




train_img_names[:3]


# # TODO read in file names
# 
# Where n in bw 0 and 6 inclusive
# 
# w is 0-x incl for train, x+1-end for test
# 
# well_{w}_crop_{n}.png')



for im in train_img_names:
    assert 'crop' in str(im)




train_mask_names = get_image_files(train_mask)
train_mask_names[:3]




for im in train_mask_names:
    assert 'crop' in str(im)


# ### Data QC



inames=[]
mnames=[]
for im in train_img_names:
    inames.append(im.name)
for im in train_mask_names:
    mnames.append(im.name)




train_img_names[0]




missing_i_m=set(inames) - set(mnames)




missing_m_i= set(mnames)-set(inames) 




assert len(missing_i_m)==len(missing_m_i)==0




len(inames)




len(mnames)




inames.sort()
mnames.sort()




len(inames)




for im,mm in zip(inames, mnames):
    assert im==mm
    img=train_img/f'{im}'
    mmg=train_mask/f'{mm}'
    img_ =pil_image.open(img)
    mmg_ =pil_image.open(mmg)
    if img_.size != mmg_.size:
        print(f'img_.size: {img_.size} != mmg_.size: {mmg_.size}')




img_f = train_img_names[5]
print(img_f)
img_gr = open_image(img_f)
img_gr.show(figsize=(18,4))




mask_f = train_mask_names[5]
mask_gr = open_image(mask_f)
mask_gr.show(figsize=(18,4))


# ### Link Masks with Images



get_y_fn = lambda x: train_mask/f'{x.stem}{x.suffix}' # converts from image to mask file name




get_y_fn(img_f)




mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(18,4), alpha=1)




u_values=[]
def get_all_uniques(images, subset=None):
    for i, im in enumerate(images):
        if subset and (i > subset):
            return u_values
        mask = np.asarray(pil_image.open(im))
        u_values.extend(np.unique(mask.tolist()))
    return u_values




#u_values=list(set(get_all_uniques(train_mask_names, 100)))
#u_values.sort()
#print(u_values)


# Note that not all colours are mapped by fastai - we need to do this manually

# ### Log Facies

# <pre>
#             ▪ 0 (None), 
#             ▪ 1 (Symmetrical), Hour glass (Prograding and retrograding)
#             ▪ 2 (Cylindrical) Blocky sst (Aggrading)
#             ▪ 3 (Funnel) Coarsening up (Prograding)
#             ▪ 4 (Bell) Fining up (Retrograding)
# </pre>
# 
# 



#codes = array(['Background', 'None', 'Symmetrical', 'Cylindrical', 'Funnel', 'Bell'])
codes = array(['Background', 'Funnel', 'None', 'Cylindrical', 'Symmetrical', 'Bell'])




src_size = np.array(mask.shape[1:])
src_size,mask.data




gr_size = np.array(img_gr.shape[1:])
gr_size,img_gr.data


# ## Datasets



bs = 8
#size=src_size//2
#squish to square

size=(tgt_height, tgt_height)




tfms=get_transforms(do_flip=False, flip_vert=False, max_rotate=0., max_zoom=1.1, max_lighting=0.0,                     max_warp=0., p_affine=0., p_lighting=0.0)




src = (SegmentationItemList.from_folder(path=train_img)
    .split_by_fname_file('../val_20pct_4.csv', path=train_img)
    .label_from_func(get_y_fn, classes=codes))




data = (src.transform(tfms, size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))




len(src.train.x)




#src_test = (SegmentationItemList.from_folder(path=train_img)
#    .label_empty()
#    .split_none()
#    .add_test_folder(test_folder='../../test', tfms=None, tfm_y=False))




#data_test = (src_test.transform(size=size, tfms=None, tfm_y=False)
#        .databunch(bs=bs)
#        .normalize(imagenet_stats))




data.train_ds.x[0].shape




#data_test.valid_ds.x[0].shape




data.train_ds.y[0].shape




uy=[]
def get_unique_y_vals(data_list):
    for yt in data_list:
        y=yt.data.numpy()
        uy.extend(np.unique(y.tolist()))
    return uy




#u_values=list(set(get_unique_y_vals(data.train_ds.y)))
#u_values.sort()
#print(u_values)




#y=data.train_ds.y[0].data.numpy()




#np.unique(y)




data.label_list




data.show_batch(4, figsize=(10,7))




data.show_batch(2, figsize=(10,7), ds_type=DatasetType.Valid)


# ### Model

# Eval criterion:
#     
# Classification Accuracy (percentage of correctly predicted rows)



def dice(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def iou(input:Tensor, targs:Tensor) -> Rank0Tensor:
    "IoU coefficient metric for binary target."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect = (input*targs).sum().float()
    union = (input+targs).sum().float()
    return intersect / (union-intersect+1.0)




name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Background']
print(void_code)

def acc_camvid(input, target):
    #print(f'in: {input.shape}, tgt: {target.shape}')
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()




data




wd=1e-2




# Create U-Net with a pretrained resnet34 as encoder
learn = unet_learner(data, models.resnet34, metrics=acc_camvid, wd=wd).to_fp16()




#learn.model = torch.nn.DataParallel(learn.model, device_ids=[0, 1, 2])




learn.callback_fns




#learn.loss_func=dice
learn.loss_func


#learn.pred_batch()



lr=5e-4




learn.fit_one_cycle(1, slice(lr))
learn.save(f'{NB}-{UID}_unet_squish-s1-r0-{DATE}')

learn.fit_one_cycle(1, slice(lr))
learn.save(f'{NB}-{UID}_unet_squish-s1_1-r0-{DATE}')

learn.fit_one_cycle(1, slice(lr))
learn.save(f'{NB}-{UID}_unet_squish-s1_2-r0-{DATE}')

learn.fit_one_cycle(1, slice(lr))
learn.save(f'{NB}-{UID}_unet_squish-s1_3-r0-{DATE}')

learn.fit_one_cycle(1, slice(lr))
learn.save(f'{NB}-{UID}_unet_squish-s1_4-r0-{DATE}')

learn.recorder.plot_losses()

# ### Re-train

learn.unfreeze()

learn.fit_one_cycle(2, slice(5e-6, lr/5))
learn.save(f'{NB}-{UID}_unet_squish-s2-r0-{DATE}')
learn.recorder.plot_losses()


# ### Loss QC



learn.load(f'{NB}-{UID}_unet_squish-s1-r0-{DATE}')




learn.data.classes




interp=SegmentationInterpretation.from_learner(learn)




top_losses, top_idxs=interp.top_losses((275,275))




top_losses, top_idxs




top_losses.shape




top_idxs.shape




top_idxs[:20]




tnp=top_idxs.numpy()
idxs=tnp[tnp<800]




# show top loss
print(top_losses[idxs[0]])
interp.show_xyz(idxs[0], codes, sz=15)




interp.show_xyz(idxs[3], codes, sz=15)


# ### Loss Distribution



# plot loss distribution
plt.hist(to_np(top_losses), bins=20)




# top loss idxs of images
top_idxs[:5]




mean_cm, single_img_cm = interp._generate_confusion()




mean_cm.shape, single_img_cm.shape




# global class performance
df = interp._plot_intersect_cm(mean_cm, "Mean of Ratio of Intersection given True Label")
df.to_csv('02_07_3_mean_of_ratio')



# single image class performance
i = 457
df = interp._plot_intersect_cm(single_img_cm[i], f"Ratio of Intersection given True Label, Image:{i}")




# show xyz
interp.show_xyz(i)












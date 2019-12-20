#!/usr/bin/env python
# coding: utf-8

# # Image segmentaion with masked facies
# 
# Squish rectangular images to square



# In[2]:


import sys
from numbers import Integral
from random import uniform
from PIL import Image as pil_image
import fastai
from fastai.vision import *
from fastai.vision import Image
from fastai.vision.transform import _minus_epsilon
from fastai.vision.data import SegmentationProcessor
from mask_functions import *
from collections import defaultdict
import cv2
from IPython.display import display 


# In[3]:


fastai.__version__


# In[4]:


torch.cuda.set_device(0)


# In[5]:


SUBSET_DATA=True
SUBSET_LEN=170


# In[6]:


tgt_height = 256
data_dir = Path('data')
train_images = data_dir/'train_images'
test_img = train_images/'mask_fill/test'

train_path = train_images/'mask_fill/train'
train_mask = train_path/'masks'
train_img = train_path/'images'


#subset
train_sub_path = train_images/'mask_fill/subset/train'
train_sub_mask = train_sub_path/'masks'
train_sub_img = train_sub_path/'images'

train_path = train_sub_path
train_mask = train_sub_mask
train_img = train_sub_img


# In[7]:


filename = 'data/CAX_LogFacies_Train_File.csv'


# #### all data

# In[8]:


training_data = pd.read_csv(filename)
training_data.head()
training_data['well_file']='well_'+training_data['well_id'].astype(str)+'.png'
wells=training_data['well_file'].unique()
all_wells_df=pd.DataFrame(wells)
all_wells_df.head()
    
df_val = all_wells_df.sample(frac=0.2)
idx=df_val.index
df_trn=all_wells_df[~all_wells_df.index.isin(idx)]
assert len(df_val)+len(df_trn)==len(all_wells_df)
#df_val.to_csv(train_path/'val_20pct.csv', index=False, header=False)


# In[9]:


df_val.tail()


# #### Subset

# In[10]:


df_val = pd.read_csv(train_images/'mask_fill/train/val_20pct.csv', names=["well_id"])
ids=range(0,SUBSET_LEN)
well_names=[]
for i in ids:
    well_names.append('well_'+str(i)+'.png')
df_sub_val=df_val.loc[df_val['well_id'].isin(well_names)]
df_sub_val.to_csv(train_sub_path/'val_sub_20pct.csv', index=False, header=False)


# ### Data

# In[11]:


train_img_names = get_image_files(train_img)
len(train_img_names)


# In[12]:


train_img_names[:3]


# In[13]:


for im in train_img_names:
    assert 'crop' not in str(im)


# In[14]:


train_mask_names = get_image_files(train_mask)
train_mask_names[:3]


# In[15]:


for im in train_mask_names:
    assert 'crop' not in str(im)


# #### Resize
# 
# If required

# In[16]:


def resize_to(to_w, to_h, from_w, from_h, im_folder, postfix='_resized'):
    w_ratio = to_w/from_w
    h_ratio = to_h/from_h

    for infile in im_folder:
        name=str(infile).split('/')[-1].split('.png')[0]
        img = cv2.imread(str(infile), cv2.IMREAD_UNCHANGED) 
        width = int(img.shape[1] * w_ratio )
        height = int(img.shape[0] * h_ratio) 
        dim = (width, height) 
        #INTER_AREA introduces extra colours
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST) 
        pim = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        im_pil = pil_image.fromarray(pim)
        im_pil.save(train_mask/f'{name}{postfix}.png')


# In[17]:


#resize_to(1100, 275, 1528, 390, train_mask, postfix='_resized')


# ### Data QC

# In[18]:


inames=[]
mnames=[]
for im in train_img_names:
    inames.append(im.name)
for im in train_mask_names:
    mnames.append(im.name)


# In[19]:


missing_i_m=set(inames) - set(mnames)


# In[20]:


missing_m_i= set(mnames)-set(inames) 


# In[21]:


assert len(missing_i_m)==len(missing_m_i)==0


# In[22]:


len(inames)


# In[23]:


len(mnames)


# In[24]:


inames.sort()
mnames.sort()


# In[25]:


len(inames)


# In[26]:


for im,mm in zip(inames, mnames):
    assert im==mm
    img=train_img/f'{im}'
    mmg=train_mask/f'{mm}'
    img_ =pil_image.open(img)
    mmg_ =pil_image.open(mmg)
    if img_.size != mmg_.size:
        print(f'img_.size: {img_.size} != mmg_.size: {mmg_.size}')


# In[27]:


img_f = train_img_names[5]
print(img_f)
img_gr = open_image(img_f)
img_gr.show(figsize=(18,4))


# In[28]:


img_h, img_w=img_gr.size


# In[29]:


img_w


# In[30]:


img_h


# ### Link Masks with Images

# In[31]:


get_y_fn = lambda x: train_mask/f'{x.stem}{x.suffix}' # converts from image to mask file name


# In[32]:


get_y_fn(img_f)


# We need a custom mask function as fastai merges colours
# 
# see https://forums.fast.ai/t/u-net-rgb-masks-values-convertion-in-fast-ai/50672/3

# In[33]:


'''

def replace_bad_colors(infile, outpath):
    #Needs work
    name=str(infile).split('/')[-1].split('.png')[0]
    im = pil_image.open(infile)   
    out = pil_image.new('RGB', im.size)
    facies_tupes=[tuple(i) for i in facies_rgb_list]

    width, height = im.size
    for x in range(width):
        for y in range(height):
            r,g,b = im.getpixel((x,y))
            p=(r,g,b)
            if p not in facies_tupes:
                print(p)
                p = (255,255,255)
                out.putpixel((x,y), p)

    out.save(train_mask/f'{name}_fixed.png')
    
'''


# In[34]:


#for im in train_mask_names:
#    replace_bad_colors(im, train_mask)


# In[35]:


def get_unique_colours(img_name):
    im = pil_image.open(img_name)
    by_color = defaultdict(int)
    for pixel in im.getdata():
        by_color[pixel] += 1
    return by_color


# In[36]:


rain_lbl_3=train_mask/'well_3.png'


# In[37]:


get_unique_colours(rain_lbl_3)


# In[38]:


train_mask_names[4]


# In[39]:


get_unique_colours(train_mask_names[4])


# Create a list of RGB values in order of idx value to replace with, i.e. 0: [0,0,0], 1: [255,0,0]
# 
# Note we have an extra colour for background (where GR is > log value but less than absolute GR cutoff)
# 
# We use RGB not RGBA

# In[40]:


facies_rgb_list = [
    [255, 255, 255],
    [153, 102, 51],
    [0, 128, 0],
    [255, 0, 0],
    [0, 0, 255],
    [255, 255, 0]]


# In[41]:


def div_rgbs(rgb_list):
    rgb_zero_one=[]
    for l in rgb_list:
        rgb_zero_one.append([i /255 for i in l])
    return rgb_zero_one


# In[42]:


def convert_mask(old_mask, rgb_list):
    '''
    create a bytemask for pixels = rgb value to be replaced, 
    sum over all columns
    fill in pixels with new idx value
    '''
    new_mask = torch.zeros((old_mask.shape[-2],old_mask.shape[-1]))
    for idx, rgb in enumerate(rgb_list):
        rgb_mask = torch.sum(old_mask.data.view((3,-1)).permute(1,0) == tensor(rgb),dim=1)==3 
        new_mask.masked_fill_(rgb_mask.view(new_mask.shape), tensor(idx)) 
    return ImageSegment(new_mask.unsqueeze(0))

def open_mask_converted(fn:PathOrStr, div=False, convert_mode='RGB', after_open:Callable=None, rgb_list=facies_rgb_list)->ImageSegment:
    '''Return ImageSegment object create from mask in file fn. 
    If div, divides pixel values by 255.'''
    return convert_mask(open_image(fn, div=div, convert_mode=convert_mode, cls=ImageSegment, after_open=after_open), rgb_list)


# need to override SegmentationLabelList to use unique color masks

# In[43]:


class UniqueSegmentationLabelList(ImageList):
    "ItemList for segmentation masks"
    _processor=SegmentationProcessor
    def __init__(self, items:Iterator, classes:Collection=None, **kwargs):
        super().__init__(items, **kwargs)
        self.copy_new.append('classes')
        self.classes,self.loss_func = classes,CrossEntropyFlat(axis=1)
    def open(self, fn): 
        #need to use div=troe so is bw 0 and 1
        m=open_mask_converted(fn)
        return div_rgbs(m)
    def analyze_pred(self, pred, thresh:float=0.5): 
        return pred.argmax(dim=0)[None]
    def reconstruct(self, t:Tensor): 
        return ImageSegment(t)

class UniqueSegmentationItemList(ImageList):
    _label_cls,_square_show_res = UniqueSegmentationLabelList,False


# In[44]:


mask = open_mask_converted(get_y_fn(img_f), convert_mode='RGB', div=False)


# In[45]:


#mask = open_mask(get_y_fn(img_f)) # fastai shows masks with distinct colors 
# or open_mask(get_y_fn(img_f))  # mask reads file as (bs,x,y) so bs=1 here
# mask also assigns ints to floats >

#Return ImageSegment object create from mask in file fn. If div, divides pixel values by 255.
mask.show(figsize=(18,4), alpha=1)


# In[46]:


type(mask)


# Compare to facies generated in matplotlib

# In[47]:


img_m = get_y_fn(img_f)
img = open_image(img_m)
img.show(figsize=(18,4))


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

# In[48]:


codes = array(['Background', 'None', 'Symmetrical', 'Cylindrical', 'Funnel', 'Bell'])


# In[49]:


src_size = np.array(mask.shape[1:])
src_size,mask.data


# In[50]:


gr_size = np.array(img_gr.shape[1:])
gr_size,img_gr.data


# <pre>
# Need data layout:
#     data/
#         images/
#         labels/ #masks - for camvid same name as image but with _P postfix (before .png)
#         codes.txt #text with each label on sep line
# </pre>

# In[51]:


train_img


# In[52]:


df_val=pd.read_csv(train_path/'val_20pct.csv', header=None)


# In[53]:


df_val.head()


# In[54]:


len(df_val)


# ## Datasets

# In[55]:


bs = 16
#size=src_size//2
#squish to square

size=(tgt_height, tgt_height)


# In[56]:


tfms=get_transforms(do_flip=False, flip_vert=False, max_rotate=0., max_zoom=1.1, max_lighting=0.0,                     max_warp=0., p_affine=0., p_lighting=0.0)


# In[58]:


if SUBSET_DATA:
    src = (SegmentationLabelList.from_folder(train_sub_img)
       .split_by_fname_file('../val_sub_20pct.csv', path=train_sub_img)
       #.split_by_rand_pct()
       .label_from_func(get_y_fn, classes=codes))
else:
    src = (UniqueSegmentationLabelList.from_folder(train_img)
       .split_by_fname_file('../val_20pct.csv', path=train_img)
       #.split_by_rand_pct()
       .label_from_func(get_y_fn, classes=codes))


# In[ ]:


train_img


# In[ ]:


len(src.train.x)


# <pre>
# SegmentationItemList->ImageList->ItemList
# ItemList.__getitem__
# 
# ImageDataBunch->DataBunch
# DataBunch.dl = DeviceDataLoader
# </pre>

# In[ ]:


data = (src.transform(tfms, size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))


# In[ ]:


data.train_ds.x[0].shape


# In[ ]:


data.train_ds.y[0].shape


# In[ ]:


data.label_list


# In[ ]:


data.show_batch(2, figsize=(10,7))


# In[ ]:


data.show_batch(2, figsize=(10,7))


# ### Model

# Eval criterion:
#     
# Classification Accuracy (percentage of correctly predicted rows)

# In[64]:


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


# In[65]:


name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Background']
print(void_code)

def acc_camvid(input, target):
    print(f'in: {input.shape}, tgt: {target.shape}')
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


# In[66]:


data


# In[67]:


wd=1e-2


# In[68]:


# Create U-Net with a pretrained resnet34 as encoder
learn = unet_learner(data, models.resnet34, metrics=acc_camvid, wd=wd).to_fp16()


# In[69]:


learn.callback_fns


# In[70]:


#learn.loss_func=dice
learn.loss_func


# In[71]:


#learn.pred_batch()


# In[72]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr=1e-4


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


# Unfreeze the encoder (resnet34)
learn.unfreeze()


# In[ ]:


# Fit one cycle of 12 epochs
lr = 1e-3
learn.fit_one_cycle(12, slice(lr/30, lr))


# In[ ]:


# Predictions for the validation set
preds, ys = learn.get_preds()
preds = preds[:,1,...]
ys = ys.squeeze()


# In[ ]:


def dice_overall(preds, targs):
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds+targs).sum(-1).float()
    u0 = union==0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union)


# In[ ]:


# Find optimal threshold
dices = []
thrs = np.arange(0.01, 1, 0.01)
for i in progress_bar(thrs):
    preds_m = (preds>i).long()
    dices.append(dice_overall(preds_m, ys).mean())
dices = np.array(dices)


# In[ ]:


best_dice = dices.max()
best_thr = thrs[dices.argmax()]

plt.figure(figsize=(8,4))
plt.plot(thrs, dices)
plt.vlines(x=best_thr, ymin=dices.min(), ymax=dices.max())
plt.text(best_thr+0.03, best_dice-0.01, f'DICE = {best_dice:.3f}', fontsize=14);
plt.show()


# In[ ]:


# Plot some samples
rows = 10
plot_idx = ys.sum((1,2)).sort(descending=True).indices[:rows]
for idx in plot_idx:
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
    ax0.imshow(data.valid_ds[idx][0].data.numpy().transpose(1,2,0))
    ax1.imshow(ys[idx], vmin=0, vmax=1)
    ax2.imshow(preds[idx], vmin=0, vmax=1)
    ax1.set_title('Targets')
    ax2.set_title('Predictions')


# In[ ]:


# Predictions for test set
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
preds = (preds[:,1,...]>best_thr).long().numpy()
print(preds.sum())


# In[ ]:


# Generate rle encodings (images are first converted to the original size)
rles = []
for p in progress_bar(preds):
    im = PIL.Image.fromarray((p.T*255).astype(np.uint8)).resize((1024,1024))
    im = np.asarray(im)
    rles.append(mask2rle(im, 1024, 1024))


# In[ ]:


ids = [o.stem for o in data.test_ds.items]
sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
sub_df.head()


# In[ ]:


sub_df.to_csv('submission.csv', index=False)


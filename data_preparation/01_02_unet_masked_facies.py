#!/usr/bin/env python
# coding: utf-8

# # Image segmentaion with masked facies



import sys
from numbers import Integral
from random import uniform
from PIL import Image as pil_image
import fastai
from fastai.vision import *
from fastai.vision.transform import _minus_epsilon
from fastai.vision.data import SegmentationProcessor
from mask_functions import *
from collections import defaultdict
import cv2




fastai.__version__




torch.cuda.set_device(0)




SUBSET_DATA=True
SUBSET_LEN=100




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




filename = 'data/CAX_LogFacies_Train_File.csv'







# #### all data



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
df_val.to_csv(train_path/'val_20pct.csv', index=False, header=False)




df_val.head()


# #### Subset



df_val = pd.read_csv(train_images/'mask_fill/train/val_20pct.csv', names=["well_id"])
ids=range(0,SUBSET_LEN)
well_names=[]
for i in ids:
    well_names.append('well_'+str(i)+'.png')
df_sub_val=df_val.loc[df_val['well_id'].isin(well_names)]
df_sub_val.to_csv(train_sub_path/'val_sub_20pct.csv', index=False, header=False)


# ### Data



fnames = get_image_files(train_img)
len(fnames)




fnames[:3]




lbl_names = get_image_files(train_mask)
lbl_names[:3]


# #### Resize all images to square



def resize_to(f, outpath):
    '''using open CV to resize as this allows exact dims to be used'''
    pil_im=pil_image.open(f)
    img=np.array(pil_im)
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img, size, interpolation = cv2.INTER_LANCZOS4) 
    outimg = pil_image.fromarray(resized, "RGB" )
    out_file=outpath/f'{f.name}'
    outimg.save(out_file)
    outimg.close()




def resize_all(size, fnames, mnames):
    pma=path_lbl/f'{size[0]}'
    os.makedirs(pma, exist_ok=True)
    pim=path_img/f'{size[0]}'
    os.makedirs(pim, exist_ok=True)
    for f in fnames:
        resize_to(f, pim)
    for f in mnames:
        resize_to(f, pma)




#resize_all((tgt_height,tgt_height), fnames, lbl_names)




def resize_masks():
    img = img.resize((new_width, new_height), Image.ANTIALIAS)


# ### Data QC



ims = [open_image(train_img/f'well_{i}.png') for i in range(6)]
im_masks = [open_image(train_mask/f'well_{i}.png') for i in range(6)]




img_f = fnames[5]
print(img_f)
img_gr = open_image(img_f)
img_gr.show(figsize=(18,4))


# ### Link Masks with Images



get_y_fn = lambda x: train_mask/f'{x.stem}{x.suffix}' # converts from image to mask file name




get_y_fn(img_f)


# We need a custom mask function as fastai merges colours
# 
# see https://forums.fast.ai/t/u-net-rgb-masks-values-convertion-in-fast-ai/50672/3



def get_unique_colours(img_name):
    im = pil_image.open(img_name)
    by_color = defaultdict(int)
    for pixel in im.getdata():
        by_color[pixel] += 1
    return by_color




rain_lbl_3=train_mask/'well_3.png'




get_unique_colours(rain_lbl_3)




lbl_names[4]




get_unique_colours(lbl_names[4])


# Create a list of RGB values in order of idx value to replace with, i.e. 0: [0,0,0], 1: [255,0,0]
# 
# Note we have an extra colour for background (where GR is > log value but less than absolute GR cutoff)
# 
# We use RGB not RGBA



facies_rgb_list = [
    [255, 255, 255],
    [153, 102, 51],
    [0, 128, 0],
    [255, 0, 0],
    [0, 0, 255],
    [255, 255, 0]]




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



class UniqueSegmentationLabelList(ImageList):
    "ItemList for segmentation masks"
    _processor=SegmentationProcessor
    def __init__(self, items:Iterator, classes:Collection=None, **kwargs):
        super().__init__(items, **kwargs)
        self.copy_new.append('classes')
        self.classes,self.loss_func = classes,CrossEntropyFlat(axis=1)

    def open(self, fn): 
        return open_mask_converted(fn)
    def analyze_pred(self, pred, thresh:float=0.5): 
        return pred.argmax(dim=0)[None]
    def reconstruct(self, t:Tensor): 
        return ImageSegment(t)

class UniqueSegmentationItemList(ImageList):
    _label_cls,_square_show_res = UniqueSegmentationLabelList,False




class AdvSegmentationItemList(UniqueSegmentationItemList):
    def __getitem__(self,idxs:int)->Any:
        "returns a single item based if `idxs` is an integer or a new `ItemList` object if `idxs` is a range."
        print(f'>>ItemList.__getItem__ items @ idxs: {self.items[idxs]}')
        idxs = try_int(idxs)
        if isinstance(idxs, Integral): return self.get(idxs)
        else: return self.new(self.items[idxs], inner_df=index_row(self.inner_df, idxs))




mask = open_mask_converted(get_y_fn(img_f), convert_mode='RGB', div=False)




#mask = open_mask(get_y_fn(img_f)) # fastai shows masks with distinct colors 
# or open_mask(get_y_fn(img_f))  # mask reads file as (bs,x,y) so bs=1 here
# mask also assigns ints to floats >

#Return ImageSegment object create from mask in file fn. If div, divides pixel values by 255.
mask.show(figsize=(18,4), alpha=1)


# Compare to facies generated in matplotlib



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



codes = array(['Background', 'None', 'Symmetrical', 'Cylindrical', 'Funnel', 'Bell'])




src_size = np.array(mask.shape[1:])
src_size,mask.data




gr_size = np.array(img_gr.shape[1:])
gr_size,img_gr.data


# <pre>
# Need data layout:
#     data/
#         images/
#         labels/ #masks - for camvid same name as image but with _P postfix (before .png)
#         codes.txt #text with each label on sep line
# </pre>



train_img




df_val=pd.read_csv(train_path/'val_20pct.csv', header=None)




df_val.head()




len(df_val)


# ## Datasets



bs = 16
#size=src_size//2
#squish to square
size=(tgt_height, tgt_height)




def get_n_rand_crops(n):
    rc=[]
    for i in range(n):
        rc.append(round(uniform( 0.0, 1.0), 2))
    return rc




n_crops=[0.0, 0.166, 0.333, 0.5, 0.666, 1.0]




def _sliding_crop(x, size, row_pct:uniform=0.5, col_pct:uniform=0.5):
    "Crop `x` to `size` pixels. `row_pct`,`col_pct` select focal point of crop."
    rows,cols = tis2hw(size)
    print(f'rows: {rows}, cols: {cols}')
    row_pct,col_pct = _minus_epsilon(row_pct,col_pct)
    print(f'row_pct: {row_pct}, col_pct: {col_pct}')
    row = int((x.size(1)-rows+1) * row_pct)
    col = int((x.size(2)-cols+1) * col_pct)
    print(f'row: {row}, col: {col}')
    img= x[:, row:row+rows, col:col+cols].contiguous()
    return img




sliding_crop=TfmPixel(_sliding_crop)




print(1+2)




crops = [sliding_crop(size=(tgt_height, tgt_height))]




tfms=get_transforms(do_flip=False, flip_vert=False, max_rotate=0., max_zoom=1.0, max_lighting=0.0,                     max_warp=0., p_affine=0., p_lighting=0.0, xtra_tfms=crops)




if SUBSET_DATA:
    src = (SegmentationItemList.from_folder(train_sub_img)
       .split_by_fname_file('../val_sub_20pct.csv', path=train_sub_img)
       #.split_by_rand_pct()
       .label_from_func(get_y_fn, classes=codes))
else:
    src = (SegmentationItemList.from_folder(train_img)
       .split_by_fname_file('../val_20pct.csv', path=train_img)
       #.split_by_rand_pct()
       .label_from_func(get_y_fn, classes=codes))




len(src.train.x)


# <pre>
# SegmentationItemList->ImageList->ItemList
# ItemList.__getitem__
# 
# ImageDataBunch->DataBunch
# DataBunch.dl = DeviceDataLoader
# </pre>



data = (src.transform(tfms, size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))




data.train_ds.x[0].shape




data.train_ds.y[0].shape




data.show_batch(2, figsize=(10,7))




data.show_batch(2, figsize=(10,7))


# ### Model

# Eval criterion:
#     
# Classification Accuracy (percentage of correctly predicted rows)



def dice_loss(input, target):
    # pdb.set_trace()
    smooth = 1.
    
    input = input.sigmoid()
    input = input[:,1,None]
    iflat = input.contiguous().view(-1).float()
    tflat = target.view(-1).float()
    intersection = (iflat * tflat).sum()    

    return (1 - ((2. * intersection + smooth).float() / 
                 (iflat.sum() + tflat.sum() +smooth)).float())

def combo_loss(pred, targ):
    bce_loss = CrossEntropyFlat()
    return bce_loss(pred,targ) + dice_loss(pred,targ)




def dice(input, target):
    smooth=.001
    input=input[:,1,:,:].contiguous().view(-1).float().cuda()
    target=target.view(-1).float().cuda()
    return(1-2*(input*target).sum()/(input.sum()+target.sum()+smooth))




#see github/daveluo/zanzibar-aerial-mapping
def acc_fixed(input, targs):
    n = targs.shape[0]
    targs = targs.squeeze(1)
    targs = targs.view(n,-1)
    input = input.argmax(dim=1).view(n,-1)
    return (input==targs).float().mean()

def acc_thresh(input:Tensor, target:Tensor, thresh:float=0.5, sigmoid:bool=True)->Rank0Tensor:
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    
    # pdb.set_trace()
    if sigmoid: input = input.sigmoid()
    n = input.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    target = target.view(n,-1)
    return ((input>thresh)==target.byte()).float().mean()




metrics = [dice_loss, acc_thresh, dice]




name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Background']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()




@dataclass
class DebugCB(LearnerCallback):
    def __init__(self, learn:Learner, clip:float = 0.):
        super().__init__(learn)
        print('DebugCB.init')

    def on_batch_begin(self, **kwargs):
        print('on_batch_begin')
    '''
    def on_backward_end(self, **kwargs):
        print('on_backward_end')
        #if self.clip: nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)
    def on_step_end(self, **kwargs):
        print('on_step_end')
    def on_batch_end(self, **kwargs):
        print('on_batch_end')
    '''
    def on_epoch_end(self, **kwargs):
        print('on_epoch_end')
    def on_train_end(self, **kwargs):
        print('on_train_end')




# Create U-Net with a pretrained resnet34 as encoder
learn = unet_learner(data, models.resnet34, metrics=acc_camvid, callback_fns=[partial(DebugCB)]).to_fp16()




#function: batch wrapped by transfroms
b=learn.data.train_dl.__iter__




type(learn.data.train_ds)




learn.callback_fns




#learn.loss_func=dice
learn.loss_func




learn.lr_find()




learn.recorder.plot()




lr=1e-4




learn.fit_one_cycle(5, slice(lr))




# Unfreeze the encoder (resnet34)
learn.unfreeze()




# Fit one cycle of 12 epochs
lr = 1e-3
learn.fit_one_cycle(12, slice(lr/30, lr))




# Predictions for the validation set
preds, ys = learn.get_preds()
preds = preds[:,1,...]
ys = ys.squeeze()




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




# Find optimal threshold
dices = []
thrs = np.arange(0.01, 1, 0.01)
for i in progress_bar(thrs):
    preds_m = (preds>i).long()
    dices.append(dice_overall(preds_m, ys).mean())
dices = np.array(dices)




best_dice = dices.max()
best_thr = thrs[dices.argmax()]

plt.figure(figsize=(8,4))
plt.plot(thrs, dices)
plt.vlines(x=best_thr, ymin=dices.min(), ymax=dices.max())
plt.text(best_thr+0.03, best_dice-0.01, f'DICE = {best_dice:.3f}', fontsize=14);
plt.show()




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




# Predictions for test set
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
preds = (preds[:,1,...]>best_thr).long().numpy()
print(preds.sum())




# Generate rle encodings (images are first converted to the original size)
rles = []
for p in progress_bar(preds):
    im = PIL.Image.fromarray((p.T*255).astype(np.uint8)).resize((1024,1024))
    im = np.asarray(im)
    rles.append(mask2rle(im, 1024, 1024))




ids = [o.stem for o in data.test_ds.items]
sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
sub_df.head()




sub_df.to_csv('submission.csv', index=False)


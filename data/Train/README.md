Download datasets from the following links and place them in this directory. Your directory tree should look like this

`Derain` <br/>
  `├──`[rainy](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)  <br/>
  `└──`gt <br/>

`Dehaze` <br/>
  `├──`[synthetic](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)  <br/>
  `└──`original <br/>

For denoise dir, you could just put the clean images (e.g,  [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing) and [WED](https://ece.uwaterloo.ca/~k29ma/exploration/)) into the directory. In the paper, we use the combination of BSD400 and WED as the training set.


### 关于 Deblur 的定义
- 图像对路径由 _get_clean_name() 定义。
- _init_db_ids() 方法获取模糊图片
- de_dict 5

### 关于 SR 的定义
- 图像对路径由 _get_sr_name() 定义
- _init_sr_ids() 方法获取模糊图片，注意 lq 图片可能带有倍率
- DGRN 中改了 DGRN_U 加了 Upsample
- option 中添加了 scale 与 num_feats
- de_dict 6
- lq 与 gt 文件名字相同
- 图片截取 crop 添加了倍率以适应超分辨率任务
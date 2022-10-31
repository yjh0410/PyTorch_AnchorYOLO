# AnchorYOLO
这个AnchorYOLO项目是配合我在知乎专栏上连载的《YOLO入门教程》而创建的：

https://zhuanlan.zhihu.com/c_1364967262269693952

感兴趣的小伙伴可以配合着上面的专栏来一起学习，入门目标检测。

# 配置环境
- 我们建议使用anaconda来创建虚拟环境:
```Shell
conda create -n yolo python=3.6
```

- 然后，激活虚拟环境:
```Shell
conda activate yolo
```

- 配置环境:
运行下方的命令即可一键配置相关的深度学习环境：
```Shell
pip install -r requirements.txt 
```

# 训练技巧
- [x] [Mosaic Augmentation](https://github.com/yjh0410/FreeYOLO/blob/master/dataset/transforms.py)
- [x] [Mixup Augmentation](https://github.com/yjh0410/FreeYOLO/blob/master/dataset/transforms.py)
- [x] Multi scale training
- [x] Cosine Annealing Schedule

# 训练配置
|   Configuration         |                      |
|-------------------------|----------------------|
| Batch Size (bs)         | 16                   |
| Init Lr                 | 0.01/64 × bs         |
| Lr Scheduler            | Cos                  |
| Optimizer               | SGD                  |
| ImageNet Predtrained    | True                 |
| Multi Scale Train       | True                 |
| Mosaic                  | True                 |
| Mixup                   | True                 |


# 实验结果
## COCO

Main results on COCO-val:

| Model         |  Scale  | FPS<sup><br>2080ti |  GFLOPs | Params(M) |    AP    |    AP50    |  Weight  |
|---------------|---------|--------------------|---------|-----------|----------|------------|----------|
| AnchorYOLO    |  320    |  --                |  42.3   |   62.0    |  39.8    |   58.6     | - |
| AnchorYOLO    |  416    |  --                |  71.5   |   62.0    |  42.8    |   62.1     | - |
| AnchorYOLO    |  512    |  --                |  108.3  |   62.0    |  44.6    |   64.5     | - |
| AnchorYOLO    |  608    |  --                |  152.7  |   62.0    |  45.5    |   66.0     | - |
| AnchorYOLO    |  640    |  45                |  168.8  |   62.0    |  45.6    |   63.6     | [github](https://github.com/yjh0410/PyTorch_AnchorYOLO/releases/download/yolo_anchor_weight/yolo_anchor_45.6.pth) |

# 训练
## 单GPU训练
一键运行本项目提供的bash文件：

```Shell
sh train.sh
```

请根据自己的情况去修改```train.sh```文件中的配置。

## 多PGU训练
一键运行本项目提供的bash文件：

```Shell
sh train_ddp.sh
```

请根据自己的情况去修改```train.sh```文件中的配置。

**当遇到训练被终端的情况时**, 你可以给 `--resume` 传入最新的权重（默认为None），如下:

```Shell
python train.py \
        --cuda \
        -d coco \
        -v yolo_anchor \
        --ema \
        --fp16 \
        --eval_epoch 10 \
        --resume weights/coco/yolo_anchor/yolo_anchor_epoch_151_39.24.pth
```

然后，训练会从151 epoch继续。

# 测试
你可以参考下方的命令来测试模型在数据集上的检测性能，将会看到检测结果的可视化图像。

```Shell
python test.py -d coco \
               --cuda \
               -v yolo_anchor \
               --img_size 640 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
```

# 验证
你可以参考下方的命令来测试模型在数据集上的AP指标。

```Shell
python eval.py -d coco-val \
               --cuda \
               -v yolo_anchor \
               --img_size 640 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
```

# Demo
本项目提供了一些图片，在 `data/demo/images/`文件夹中, 你可以运行下面的命令来检测本地的图片:

```Shell
python demo.py --mode image \
               --path_to_img data/demo/images/ \
               -v yolo_anchor \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

你可以通过修改`--path_to_img`的参数来指向你自己的图片所在文件夹。

如果你想检测本地的视频，可以给参数`--mode`传入`video`，并给参数`--path_to_vid`传入视频的路径：

```Shell
python demo.py --mode video \
               --path_to_img data/demo/videos/your_video \
               -v yolo_anchor \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

如果你想检测用外部的摄像头来实时运行模型，可以给参数`--mode`传入`camera`：

```Shell
python demo.py --mode camera \
               -v yolo_anchor \
               --img_size 640 \
               --cuda \
               --weight path/to/weight
```

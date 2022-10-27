# Train AnchorYOLO
python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolo_anchor \
        --ema \
        --fp16 \
        --eval_epoch 10 \
        # --resume weights/coco/yolo_anchor/yolo_anchor_epoch_201_46.98.pth

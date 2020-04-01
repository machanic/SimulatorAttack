# export PYTHONPATH=./:$PYTHONPATH
python adversarial_defense/feature_scatter/fs_main.py --adv_mode 'feature_scatter'  --lr 0.1  \
    --init_model_pass latest \
    --max_epoch 200 \
    --save_epochs 100 \
    --decay_epoch1 60 \
    --decay_epoch2 90 \
    --batch_size_train 60 \
    --dataset CIFAR-100 \
    --arch gdas --gpu 2


python train.py config_file=conf/resisc45.yaml experiment.name=resisc45-vit experiment.module.classification_model=vit
python train.py config_file=conf/resisc45.yaml experiment.name=resisc45-convnext experiment.module.classification_model=convnext experiment.datamodule.batch_size=64
python train.py config_file=conf/resisc45.yaml experiment.name=resisc45-resnet experiment.module.classification_model=resnet

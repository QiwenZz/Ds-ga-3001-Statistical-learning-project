# Ds-ga-3001-Statistical-learning-project

To Do
- Fix name defintion (name is the eval_accuracy and the state dict contains the arguments of the experiment) 
- Add .py file for submission csv generation- Qiwen
- Re write the train_model_se for clarity
- Add majority voting for the ensemble (so far done averaging)
- Fix snapshot ensembling for Adam (same error on both my implementation and https://ensemble-pytorch.readthedocs.io/en/latest/index.html)
- Perhaps add more sophisticated augmentations (segmentations, mixup/cutmix, random augmentation, auto augmentation)
- Play with the training loop, initially finetune on small images, and repeat by using the finetuned as a base model and further finetuning by gradually increasing size - Qiwen
- Hyperparameter tuning(Vision transformer - Qiwen)

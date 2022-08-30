export CUDA_VISIBLE_DEVICES=0
python train_bert.py --exp_name bert_cls_medis
python train_bert.py —exp_name bert_cls_medis_aug --data_aug

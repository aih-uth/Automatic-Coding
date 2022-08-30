# MEDIS
python eval_bert.py --exp_name bert_cls_medis --save_path bert_cls_medis
python eval_bert.py --exp_name bert_cls_medis_aug --save_path bert_cls_medis_aug

python eval_bert_knn.py --exp_name bert_cls_medis --save_path bert_cls_medis_knn_1 --k 1
python eval_bert_knn.py --exp_name bert_cls_medis --save_path bert_cls_medis_knn_3 --k 3

python eval_bert_knn.py --exp_name bert_cls_medis_aug --save_path bert_cls_medis_aug_knn_1 --k 1
python eval_bert_knn.py --exp_name bert_cls_medis_aug --save_path bert_cls_medis_aug_knn_3 --k 3

python eval_bert_knn.py --exp_name bert_adacos_medis --save_path bert_adacos_medis_knn_1 --k 1
python eval_bert_knn.py --exp_name bert_adacos_medis --save_path bert_adacos_medis_knn_3 --k 3

python eval_bert_knn.py --exp_name bert_adacos_medis_aug --save_path bert_adacos_medis_aug_knn_1 --k 1
python eval_bert_knn.py --exp_name bert_adacos_medis_aug --save_path bert_adacos_medis_aug_knn_3 --k 3
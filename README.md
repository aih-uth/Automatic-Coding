# 医療用語の病名交換コード/ICD-10へのマッピング

本リポジトリは第42回医療情報学連合大会で発表したYYYの実験コードです。
詳細抄録に掲載されているデータ拡張を用いた実験の再現にはティ辞書企画さまより販売されているシソーラス辞書が必要となります。

## 自前準備

1. [ICD10対応標準病名マスター](http://www2.medis.or.jp/stdcd/byomei/index.html)をダウンロードし、./data/medisにindex504.txtとnmain504.csvを置く。
2. [シソーラス辞書](https://www.tdic.co.jp/)がある場合はtdic_tenkai202111_UTF8.csvを./data/tに置く。
3. [万病辞書](http://sociocom.jp/~data/2018-manbyo/index.html)をダウンロードし、./data/manbyoにMANBYO_20210602.csvを置く。
4. python preprocess.py --data_augを実行する（シソーラス辞書がない場合は--data_augをつけない）
5. [UTH-BERT](https://ai-health.m.u-tokyo.ac.jp/home/research/uth-bert)をダウンロードし、./BERTに置く。

## 学習

MEDISは病名交換コードの分類、ICDはICD-10の分類を意味します。

### 機械学習モデルの訓練 (MEDIS)
```
bash train_medsi.sh
```

### 機械学習モデルの訓練 (ICD)
```
bash train_icd.sh
```

## 評価

### MEDISで評価を行い、UNKを考慮しない場合

```
bash eval_medis_wo_unk.sh
```

### ICDで評価を行い、UNKを考慮しない場合

```
bash eval_icd_wo_unk.sh
```

### MEDISで評価を行い、UNKを考慮する場合

```
bash eval_medis_w_unk.sh
```

### ICDで評価を行い、UNKを考慮する場合

```
bash eval_icd_w_unk.sh
```

### ベースライン (MEDIS)
ベースラインを実行する前に必ず機械学習モデルの訓練を実行してください。

```
# Simstring
# データ拡張なし
python baseline.py --data_path ./results/bert_cls_medis
# データ拡張あり
python baseline.py --data_path ./results/bert_cls_medis_aug

# TF-IDF + ロジスティック回帰
# データ拡張なし
python baseline_tfidf.py --data_path ./results/bert_cls_medis --exp_name tfidf_cls_medis
# データ拡張あり
python baseline_tfidf.py --data_path ./results/bert_cls_medis_aug --exp_name tfidf_cls_medis

# TF-IDF + kNN
# データ拡張なし
python baseline_tfidf.py --data_path ./results/bert_cls_medis --exp_name tfidf_knn_medis -knn
# データ拡張あり
python baseline_tfidf.py --data_path ./results/bert_cls_medis_aug --exp_name tfidf_knn_medis -knn
```

### ベースライン (ICD)
```
# Simstring
# データ拡張なし
python baseline.py --data_path ./results/bert_cls_icd --icd
# データ拡張あり
python baseline.py --data_path ./results/bert_cls_icd_aug --icd

# TF-IDF + ロジスティック回帰
# データ拡張なし
python baseline_tfidf.py --data_path ./results/bert_cls_icd --exp_name tfidf_cls_icd --icd
# データ拡張あり
python baseline_tfidf.py --data_path ./results/bert_cls_icd_aug --exp_name tfidf_cls_icd --icd

# TF-IDF + kNN
# データ拡張なし
python baseline_tfidf.py --data_path ./results/bert_cls_medis --exp_name tfidf_knn_icd -knn
# データ拡張あり
python baseline_tfidf.py --data_path ./results/bert_cls_medis_aug --exp_name tfidf_knn_icd -knn
```

## 参考

- pytorch-adacos Software available from https://github.com/4uiiurz1/pytorch-adacos.


## 引用
本リポジトリを参照する場合は以下の文献を引用してください。


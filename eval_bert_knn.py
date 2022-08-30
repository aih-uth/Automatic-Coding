import os
import sys
import random
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from sklearn.metrics import classification_report
import transformers
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import copy, argparse
import json, MeCab
from transformers import BertTokenizer, BertModel
import jaconv, unicodedata, regex
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lib.util import load_data, data_augmentation, ExperimentDataset, my_collate_fn, cgi_medis2icd
from lib.model import BERT_POOLING, AdaCos
from sklearn.neighbors import KNeighborsClassifier


def main():

    for fold in range(0, 5, 1):
        # 訓練データとテストデータを読み込む
        X_train = pd.read_csv("./results/{1}/X_train_{0}.csv".format(fold, args.exp_name))
        X_test = pd.read_csv("./results/{1}/X_test_{0}.csv".format(fold, args.exp_name))

        # 訓練データでkNNを学習します
        bert, knn, idx2medis = build_knn(X_train, fold)

        # UNKを考慮する場合の評価
        
        if args.w_unk:
            # CGIでの評価 /w UNK
            cgi_res_w_unk_dct, cgi_res_dct_wo_unk_consider = extrapolation_evaluration_w_unk(bert, knn, idx2medis, fold, X_train)
            with open('./cgi_results/{1}/cgi_w_unk_{0}.json'.format(fold, args.save_path), 'w') as f:
                json.dump(cgi_res_w_unk_dct, f, indent=4)
            with open('./cgi_results/{1}/cgi_wo_unk_consider_{0}.json'.format(fold, args.save_path), 'w') as f:
                json.dump(cgi_res_dct_wo_unk_consider, f, indent=4)
        # しない場合
        else:
            # テストデータでの評価
            test_res_dct = interpotation_evaluation(X_test, bert, knn, idx2medis, fold, X_train)
            # CGIでの評価 w/o UNK
            cgi_res_wo_unk_dct = extrapolation_evaluration_wo_unk(bert, knn, idx2medis, fold, X_train)
            # 結果の保存
            with open('./test_results/{1}/test_{0}.json'.format(fold, args.save_path), 'w') as f:
                json.dump(test_res_dct, f, indent=4)
            with open('./cgi_results/{1}/cgi_wo_unk_{0}.json'.format(fold, args.save_path), 'w') as f:
                json.dump(cgi_res_wo_unk_dct, f, indent=4)



def build_knn(X_train, fold):
    with torch.no_grad():
        medis2idx = json.load(open("./results/{0}/medis2idx_{1}.json".format(args.exp_name, fold)))
        idx2medis = {v: k for k, v in medis2idx.items()}
        # DataLoer
        train_dataset = ExperimentDataset(X_train["token"].to_list(), X_train["medis"].to_list(), tokenizer, medis2idx)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
        # モデルを読み込む
        model = BERT_POOLING(args, medis2idx).to(device)
        #検証データでの損失が最良となったベストモデルを読み込む
        model.load_state_dict(torch.load('./models/{1}/model_{0}.pt'.format(fold, args.exp_name)))
        model = torch.nn.DataParallel(model)
        # 勾配を更新しない
        model.eval()
        # 訓練データの埋め込み表現を得る
        train_embed = []
        pbar_train = tqdm(train_loader)
        for batch in pbar_train:
            # to tensor
            train_vecs = pad_sequence(batch[0], padding_value=0, batch_first=True).to(device)
            train_labels = torch.tensor(batch[1]).to(device)
            # 予測
            _, embed = model(train_vecs)
            train_embed.append(embed.detach().cpu().numpy())
        # 結合
        train_embeds = np.concatenate(train_embed)
    # 正解ラベル
    #tag2idx = {x: i for i, x in enumerate(X_train["medis"].unique())}
    #idx2tag = {v: k for k, v in tag2idx.items()}
    labels = np.array([medis2idx[x] for x in X_train["medis"]])
    #KNNを学習
    knn = KNeighborsClassifier(n_neighbors=args.k)
    knn.fit(train_embeds, labels)
    return model, knn, idx2medis


def interpotation_evaluation(X_test, bert, knn, idx2medis, fold, X_train):
    # DataLoer
    medis2idx = json.load(open("./results/{0}/medis2idx_{1}.json".format(args.exp_name, fold)))
    test_dataset = ExperimentDataset(X_test["token"].to_list(), X_test["medis"].to_list(), tokenizer, medis2idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    bert.eval()
    # テストデータの埋め込み表現を得る
    test_preds, test_labels = [], list(X_test["medis"])
    pbar_test = tqdm(test_loader)
    for batch in pbar_test:
        # to tensor
        test_vecs = pad_sequence(batch[0], padding_value=0, batch_first=True).to(device)
        # 予測
        _, embed = bert(test_vecs)
        embed = embed.detach().cpu().numpy()
        # kNNで予測
        pred_medis = knn.predict(embed)
        test_preds.extend([idx2medis[x] for x in pred_medis])
    # 評価
    test_res_dct = classification_report(test_labels, test_preds, output_dict=True)
    return test_res_dct


def extrapolation_evaluration_wo_unk(bert, knn, idx2medis, fold, X_train):
    # テストデータ
    X_test = pd.read_csv("./data/cgi/cgi_raw_data.csv")
    X_test= X_test.drop_duplicates(subset=["input", "gold", "medis"])
    # UNKを弾く
    X_test = X_test[X_test["gold"] != "unknown"]
    # ID
    medis2idx = json.load(open("./results/{0}/medis2idx_{1}.json".format(args.exp_name, fold)))
    idx2medis = {v: k for k, v in medis2idx.items()}

    if args.icd:
        X_test = cgi_medis2icd(X_test)

    # 学習データに存在しない病名交換コードは弾く
    codes = list(X_train["medis"].unique())
    X_test = X_test.query('medis in @codes')
    
    # DataLoader
    test_dataset = ExperimentDataset(X_test["token"].to_list(), X_test["medis"].to_list(), tokenizer, medis2idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    bert.eval()
    # テストデータの埋め込み表現を得る
    test_preds, test_labels = [], list(X_test["medis"])
    pbar_train = tqdm(test_loader)
    for batch in pbar_train:
        # to tensor
        test_vecs = pad_sequence(batch[0], padding_value=0, batch_first=True).to(device)
        # test_labels = torch.tensor(batch[1]).to(device)
        # 予測
        _, embed = bert(test_vecs)
        embed = embed.detach().cpu().numpy()
        # kNNで予測
        pred_medis = knn.predict(embed)
        test_preds.extend([idx2medis[x] for x in pred_medis])
    # 評価
    cgi_res_dct = classification_report(test_labels, test_preds, output_dict=True)
    X_test["pred"] = test_preds
    X_test.to_csv("./cgi_results/{0}/pred_cgi_w_unk_{1}.csv".format(args.save_path, fold))
    return cgi_res_dct


def extrapolation_evaluration_w_unk(bert, knn, idx2medis, fold, X_train):
    # テストデータ
    X_test = pd.read_csv("./data/cgi/cgi_raw_data.csv")
    X_test= X_test.drop_duplicates(subset=["input", "gold", "medis"])
    # ID
    medis2idx = json.load(open("./results/{0}/medis2idx_{1}.json".format(args.exp_name, fold)))
    # UNKを追加
    medis2idx["----"] = len(medis2idx)
    idx2medis = {v: k for k, v in medis2idx.items()}

    if args.icd:
        X_test = cgi_medis2icd(X_test)

    # 学習データに存在しない病名交換コードは弾く
    codes = list(X_train["medis"].unique()) + ["----"]
    X_test = X_test.query('medis in @codes')

    # DataLoader
    test_dataset = ExperimentDataset(X_test["token"].to_list(), X_test["medis"].to_list(), tokenizer, medis2idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    bert.eval()
    # テストデータの埋め込み表現を得る
    test_preds, cgi_dist = [], []
    pbar_train = tqdm(test_loader)
    for batch in pbar_train:
        # to tensor
        test_vecs = pad_sequence(batch[0], padding_value=0, batch_first=True).to(device)
        # 予測
        _, embed = bert(test_vecs)
        embed = embed.detach().cpu().numpy()
        # kNNで予測
        pred_medis = knn.predict(embed)
        test_preds.extend([idx2medis[x] for x in pred_medis])
        neigh_dist, _ = knn.kneighbors(embed)
        cgi_dist.append(neigh_dist)
    # 近傍距離で弾く
    cgi_preds_update = []
    for i, x in enumerate(test_preds):
        dist = np.concatenate(cgi_dist)[i, :].mean()
        if dist >= args.mean_distance:
            cgi_preds_update.append("----")
        else:
            cgi_preds_update.append(x)
    X_test["pred"] = cgi_preds_update
    X_test.to_csv("./cgi_results/{0}/pred_cgi_w_unk_{1}.csv".format(args.save_path, fold))
    # 評価
    cgi_res_dct = classification_report(X_test["medis"].to_list(), cgi_preds_update, output_dict=True, labels=X_test["medis"].unique())
    # UNK (----)を除外して評価する
    cgi_res_dct_wo_unk = classification_report(X_test["medis"].to_list(), cgi_preds_update, output_dict=True, labels=[x for x in X_test["medis"].unique() if x != "----"])
    return cgi_res_dct, cgi_res_dct_wo_unk


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('use cuda device')
        seed=777
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        print('use cpu')
        device = torch.device('cpu')
        torch.manual_seed(999)
        np.random.seed(999)

    # 引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str, default='../BERT/UTH_BERT_BASE_512_MC_BPE_WWM_V25000_352K')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--exp_name', type=str, default="bert_cls_aug")
    parser.add_argument('--save_path', type=str, default="bert_cls_knn_1_md_9")
    
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--mean_distance', type=int, default=9)
    parser.add_argument('--w_unk', action='store_true')
    parser.add_argument('--icd', action='store_true')
    args = parser.parse_args()
    
    # フォルダ作成
    # for SAMPLE_DIR in ["./logs/{0}".format(args.exp_name), "./cgi_results/{0}".format(args.exp_name), "./test_results/{0}".format(args.exp_name)]:
    # UNKを考慮する場合はCGIの評価のみ、cgi_results内だけにフォルダを作る
    if args.w_unk:
        for SAMPLE_DIR in ["./cgi_results/{0}".format(args.save_path)]:
            if not os.path.exists(SAMPLE_DIR):
                os.makedirs(SAMPLE_DIR)
    else:
        for SAMPLE_DIR in ["./cgi_results/{0}".format(args.save_path), "./test_results/{0}".format(args.save_path)]:
            if not os.path.exists(SAMPLE_DIR):
                os.makedirs(SAMPLE_DIR)

    tokenizer = BertTokenizer(Path(args.bert_path) / "vocab.txt", do_lower_case=False, do_basic_tokenize=False)

    main()
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


def main():
    interpotation_evaluation()
    extrapolation_evaluration()


def interpotation_evaluation():
    # 内挿性能の評価
    for fold in range(0, 5, 1):
        # テストデータ
        X_test = pd.read_csv("./results/{0}/X_test_{1}.csv".format(args.exp_name, fold))
        medis2idx = json.load(open("./results/{0}/medis2idx_{1}.json".format(args.exp_name, fold)))
        idx2medis = {v: k for k, v in medis2idx.items()}
        # DataLoer
        test_dataset = ExperimentDataset(X_test["token"].to_list(), X_test["medis"].to_list(), tokenizer, medis2idx)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
        # モデルを読み込む
        model = BERT_POOLING(args, medis2idx).to(device)
        #検証データでの損失が最良となったベストモデルを読み込む
        model.load_state_dict(torch.load('./models/{1}/model_{0}.pt'.format(fold, args.exp_name)))
        model = torch.nn.DataParallel(model)
        # 勾配を更新しない
        model.eval()
        # 検証
        preds = []
        with torch.inference_mode():
            # 検証モード
            model.eval()
            val_running_loss = 0
            pbar_test = tqdm(test_loader)
            for batch in pbar_test:
                # to tensor
                test_vecs = pad_sequence(batch[0], padding_value=0, batch_first=True).to(device)
                test_labels = torch.tensor(batch[1]).to(device)
                # 予測
                test_logits, _ = model(test_vecs)
                # 変換
                test_indexs = torch.argmax(test_logits, dim=1).detach().cpu().numpy().tolist()
                preds.extend([idx2medis[x] for x in test_indexs])
            # 結果
            res_dct = classification_report(X_test["medis"].to_list(), preds, output_dict=True)
            # 保存
            with open('./test_results/{1}/test_results_{0}.json'.format(fold, args.save_path), 'w') as f:
                json.dump(res_dct, f, indent=4)


def extrapolation_evaluration():
    # 外挿性能の評価
    for fold in range(0, 5, 1):
        # テストデータ
        X_train = pd.read_csv("./results/{0}/X_train_{1}.csv".format(args.exp_name, fold))
        X_test = pd.read_csv("./data/cgi/cgi_raw_data.csv")
        X_test= X_test.drop_duplicates(subset=["input", "gold", "medis"])
        # UNKを弾く
        X_test = X_test[X_test["gold"] != "unknown"]
        # ID
        medis2idx = json.load(open("./results/{0}/medis2idx_{1}.json".format(args.exp_name, fold)))
        idx2medis = {v: k for k, v in medis2idx.items()}

        print(X_test.shape)

        if args.icd:
            X_test = cgi_medis2icd(X_test)

        # 学習データに存在しない病名交換コードは弾く
        # codes = list(medis2idx.keys())
        codes = list(X_train["medis"].unique())
        X_test = X_test.query('medis in @codes')

        print(X_test.shape)


        # DataLoer
        test_dataset = ExperimentDataset(X_test["token"].to_list(), X_test["medis"].to_list(), tokenizer, medis2idx)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
        # モデルを読み込む
        model = BERT_POOLING(args, medis2idx).to(device)
        #検証データでの損失が最良となったベストモデルを読み込む
        model.load_state_dict(torch.load('./models/{1}/model_{0}.pt'.format(fold, args.exp_name)))
        model = torch.nn.DataParallel(model)
        # 勾配を更新しない
        model.eval()
        # 検証
        preds = []
        with torch.inference_mode():
            # 検証モード
            model.eval()
            pbar_test = tqdm(test_loader)
            for batch in pbar_test:
                # to tensor
                test_vecs = pad_sequence(batch[0], padding_value=0, batch_first=True).to(device)
                test_labels = torch.tensor(batch[1]).to(device)
                # 予測
                test_logits, _ = model(test_vecs)
                # 変換
                test_indexs = torch.argmax(test_logits, dim=1).detach().cpu().numpy().tolist()
                preds.extend([idx2medis[x] for x in test_indexs])
            # 結果
            print(X_test.shape, len(preds))
            res_dct = classification_report(X_test["medis"].to_list(), preds, output_dict=True)
            X_test["pred"] = preds
            # 保存
            with open('./cgi_results/{1}/cgi_results_{0}.json'.format(fold, args.save_path), 'w') as f:
                json.dump(res_dct, f, indent=4)
            X_test.to_csv('./cgi_results/{1}/cgi_preds_{0}.csv'.format(fold, args.save_path))


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
    parser.add_argument('--exp_name', type=str, default="bert_cls")
    parser.add_argument('--save_path', type=str, default="bert_cls")
    parser.add_argument('--icd', action='store_true')
    args = parser.parse_args()
    
    # フォルダ作成
    # for SAMPLE_DIR in ["./logs/{0}".format(args.exp_name), "./cgi_results/{0}".format(args.exp_name), "./test_results/{0}".format(args.exp_name)]:
    for SAMPLE_DIR in ["./logs/{0}".format(args.save_path), "./cgi_results/{0}".format(args.save_path), "./test_results/{0}".format(args.save_path)]:
        if not os.path.exists(SAMPLE_DIR):
            os.makedirs(SAMPLE_DIR)

    tokenizer = BertTokenizer(Path(args.bert_path) / "vocab.txt", do_lower_case=False, do_basic_tokenize=False)

    main()
import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

from config import get_config

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
class CustomEvalCap:
    """
    res は dict 形式で、
    {
        'image_id0': { 'annotation': [str, str, ...], 'predicted': str },
        'image_id1': { 'annotation': [str, str, ...], 'predicted': str },
        ...
    }
    """
    def __init__(self, res):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}

        #self.coco = coco
        #self.cocoRes = cocoRes
        #self.params = {'image_id': coco.getImgIds()}
        self.res = res

        self.params = []
        for key, _ in self.res.items():
            self.params.append(key)
        print('tokenization...')
        self.tokenizer = PTBTokenizer()

    def evaluate(self):

        imgIds = self.params
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = [{'caption':self.res[imgId]['annotation'][0]}]
            res[imgId] = [{'caption':self.res[imgId]['predicted']}]

        # =================================================
        # Set up scorers
        # =================================================

        gts  = self.tokenizer.tokenize(gts)
        res  = self.tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            #(Spice(), "SPICE"),
            #(ClipScore(), ["CLIPScore", "RefCLIPScore"])
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    self.setEval(sc, m)
                    if 'Bleu' in m:
                        scores_bleu = scores[method.index(m)]
                        self.setImgToEvalImgs(scores_bleu, gts.keys(), m)
                    else:
                        self.setImgToEvalImgs(scores, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


def evaluation(json_files):
    results = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r') as fp:
                evaldata = json.load(fp)
            
            # create coco_eval object by taking coco and coco_result
            custom_eval = CustomEvalCap(evaldata)
            
            # SPICE will take a few minutes the first time, but speeds up due to caching
            custom_eval.evaluate()
            
            # print output evaluation scores
            key_name = os.path.dirname(json_file).split('/')[-1]
            print('\n\n ======================= \n')
            print(f'### {key_name} \n```')
            result_metrics = {}
            for metric, score in custom_eval.eval.items():
                print(f'{metric:7s}: {score:.3f}')
                result_metrics[metric] = score
            print(f'```\n')
            results[key_name] = result_metrics
        except:
            print('??? ', json_file)
    return results
    

def create_asciidoc_table(data: dict) -> str:
    """
    辞書からAsciidoc形式の表を作成します。
    エポックを列、メトリックを行に配置します。

    Args:
        data (dict): 入力データ。

    Returns:
        str: Asciidoc形式の表の文字列。
    """
    # 辞書をPandas DataFrameに変換
    # DataFrameを作成すると、キーが列名になるため、意図通りメトリックが行、エポックが列となる
    df = pd.DataFrame(data)

    # Asciidocのヘッダーを作成
    header = "| Metric | " + " | ".join(df.columns)
    
    # Asciidocの表を組み立てる
    table = ["[options=\"header\"]", "|===", header]
    for index, row in df.iterrows():
        # 各行のデータを文字列に変換し、結合
        formatted_scores = [f"{score:.3f}" for score in row.values]
        row_str = f"| {index} | " + " | ".join(map(str, formatted_scores))
        table.append(row_str)
    table.append("|===")
    
    return "\n".join(table)

def save_metric_plots(prefix: str, data: dict):
    """
    各メトリックのスコア推移を折れ線グラフにしてJPEGファイルに保存します。

    Args:
        data (dict): 入力データ。
    """
    df = pd.DataFrame(data)
    
    # 列名（'epoch_0', 'epoch_1'など）から数値（0, 1など）を抽出し、X軸データとする
    # 数値順にソートするために、数値キーと元の列名のタプルを作成
    epoch_keys = sorted(
        [(int(re.search(r'\d+', col).group()), col) for col in df.columns]
    )
    
    # ソートされた列名リストを作成
    sorted_columns = [col for _, col in epoch_keys]
    # X軸のラベル（数値）
    epochs = [num for num, _ in epoch_keys]

    # DataFrameをエポック順に並び替え
    df_sorted = df[sorted_columns]

    # 各メトリック（各行）に対してプロット
    for metric, scores in df_sorted.iterrows():
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, scores.values, marker='o', linestyle='-')
        
        plt.title(f'Metric: {metric}')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.grid(True)
        # X軸の目盛りを整数にする
        plt.xticks(epochs)
        
        # ファイル名を付けて保存
        filename = f"{prefix}/{metric}.jpeg"
        plt.savefig(filename)
        plt.close() # メモリを解放するためにプロットを閉じる
        print(f"グラフを '{filename}' として保存しました。")

def visualize(config):
    json_files = [f'{config.SAVE_DIR}/results.json']
    eval_results = evaluation(json_files)
    #prefix = config.res_path.split('/')[-1]
    # 1. Asciidoc形式の表を作成して表示
    print("--- Asciidoc Table ---")
    asciidoc_output = create_asciidoc_table(eval_results)
    print(asciidoc_output)
    print("\n" + "="*30 + "\n")
    with open(f"{config.SAVE_DIR}/metric.txt", "w", encoding="utf-8") as fp:
        print(asciidoc_output , file=fp)
    # 2. 各メトリックの折れ線グラフをJPEGで保存
    #print("--- Saving Plots ---")
    #save_metric_plots(config.SAVE_DIR, eval_results)

# --- メイン処理 ---
if __name__ == '__main__':
    config = get_config()
    visualize(config)

    """
    json_files = [f'{config.SAVE_DIR}/results.json']
    eval_results = evaluation(json_files)
    #prefix = config.res_path.split('/')[-1]
    # 1. Asciidoc形式の表を作成して表示
    print("--- Asciidoc Table ---")
    asciidoc_output = create_asciidoc_table(eval_results)
    print(asciidoc_output)
    print("\n" + "="*30 + "\n")
    with open(f"{config.SAVE_DIR}/metric.txt", "w", encoding="utf-8") as fp:
        print(asciidoc_output , file=fp)
    # 2. 各メトリックの折れ線グラフをJPEGで保存
    #print("--- Saving Plots ---")
    #save_metric_plots(config.SAVE_DIR, eval_results)
    """

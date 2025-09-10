import os
import json
from tqdm import tqdm
import tensorflow as tf
from config import get_config
from utility import  get_inference_model, generate_caption, generate_caption_fast, generate_captions_batch

def main(config):
    # Get tokenizer layer from disk
    tokenizer = tf.keras.models.load_model(config.tokernizer_path)
    tokenizer = tokenizer.layers[1]
    
    # Get model
    model = get_inference_model(config.get_model_config_path)
    
    # Load model weights
    #model.load_weights(config.get_model_weights_path)
    model.load_weights(config.get_best_model_path)
    
    with open(config.test_data_json_path, 'r') as json_file:
        test_data = json.load(json_file)
    
    vocab = tokenizer.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    results = {}
    for key, val in tqdm(test_data.items()):
        image_path = os.path.join(config.IMAGE_DIR, key)
        #text_caption = generate_caption(image_path, model, tokenizer, config.SEQ_LENGTH, config.IMAGE_SIZE, index_lookup)
        text_caption = generate_caption_fast(image_path, model, tokenizer, config.SEQ_LENGTH, config.IMAGE_SIZE, index_lookup)
        val =[ vs.replace("sos ", "").replace(" eos", "") for vs in val]
        results[os.path.basename(key)] ={
            'predicted': text_caption,
            'annotation' : val,
        }
    with open(os.path.join(config.SAVE_DIR, 'results.json'), 'w') as fp:
        json.dump(results, fp, indent=2, separators=(',', ': '))

def batchmain(config):
    # --- 1. モデルとデータの読み込み (変更なし) ---
    print("Loading tokenizer and model...")
    tokenizer = tf.keras.models.load_model(config.tokernizer_path).layers[1]
    model = get_inference_model(config.get_model_config_path)
    model.load_weights(config.get_best_model_path)
    
    with open(config.test_data_json_path, 'r') as json_file:
        test_data = json.load(json_file)
    
    vocab = tokenizer.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    
    # --- 2. バッチ処理の準備 ---
    results = {}
    
    # configからバッチサイズを取得 (例: config.BATCH_SIZE = 16)
    batch_size = config.EVAL_BATCH_SIZE
    
    # 処理するすべての画像キー(ファイル名)をリストとして取得
    all_image_keys = list(test_data.keys())
    
    print(f"Starting batch inference with batch size: {batch_size}")

    # --- 3. バッチ単位でのループ処理 ---
    # all_image_keysをbatch_sizeごとに分割してループ
    # tqdmで進捗状況を表示
    for i in tqdm(range(0, len(all_image_keys), batch_size)):
        # 現在のバッチに対応する画像キーのスライスを取得
        batch_keys = all_image_keys[i:i + batch_size]
        
        # 画像キーから完全な画像パスのリストを作成
        batch_image_paths = [os.path.join(config.IMAGE_DIR, key) for key in batch_keys]
        
        # バッチ推論関数を呼び出し
        batch_captions = generate_captions_batch(
            batch_image_paths,
            model,
            tokenizer,
            config.SEQ_LENGTH,
            config.IMAGE_SIZE,
            index_lookup
        )
        
        # --- 4. 結果の格納 (バッチごと) ---
        # バッチのキーと生成されたキャプションを組み合わせて結果を格納
        for key, caption in zip(batch_keys, batch_captions):
            # 元のアノテーションデータを取得
            val = test_data[key]
            # アノテーションの前処理 (sos, eosトークンの除去)
            val = [vs.replace("sos ", "").replace(" eos", "") for vs in val]
            
            results[os.path.basename(key)] = {
                'predicted': caption,
                'annotation': val,
            }
            
    # --- 5. 結果の保存 (変更なし) ---
    print("Saving results to JSON file...")
    with open(os.path.join(config.SAVE_DIR, 'results.json'), 'w') as fp:
        json.dump(results, fp, indent=2, separators=(',', ': '))
    
    print("Done.")

from visualize import visualize
if __name__ == "__main__":
    config = get_config()
    batchmain(config)
    visualize(config)

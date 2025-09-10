import tensorflow as tf
#from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from custom_schedule import custom_schedule
from tensorflow import keras
from model import get_cnn_model, TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel
from dataset import read_image_inf
import numpy as np
import json
import re
#from settings import *

def save_tokenizer(tokenizer, path_save):
    input = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    output = tokenizer(input)
    model = tf.keras.Model(input, output)
    model.save(path_save) #, save_format='tf')

def get_inference_model(model_config_path):
    with open(model_config_path) as json_file:
        model_config = json.load(json_file)

    EMBED_DIM = model_config["EMBED_DIM"]
    FF_DIM = model_config["FF_DIM"]
    NUM_HEADS = model_config["NUM_HEADS"]
    VOCAB_SIZE = model_config["VOCAB_SIZE"]
    IMAGE_SIZE = model_config["IMAGE_SIZE"]
    SEQ_LENGTH = model_config["SEQ_LENGTH"]
    CNN_MODEL = model_config["CNN_MODEL"]

    cnn_model = get_cnn_model(IMAGE_SIZE, CNN_MODEL)
    encoder = TransformerEncoderBlock(
        embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS
    )
    decoder = TransformerDecoderBlock(
        embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE, seq_length=SEQ_LENGTH
    )
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder
    )

    ##### It's necessary for init model -> without it, weights subclass model fails
    cnn_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    training = False
    decoder_input = tf.keras.layers.Input(shape=(None,))
    caption_model([cnn_input, decoder_input], training=training)
    #####

    return caption_model


def generate_caption(image_path, caption_model, tokenizer, SEQ_LENGTH, IMAGE_SIZE, index_lookup):
    #vocab = tokenizer.get_vocabulary()
    #index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1

    # Read the image from the disk
    img = read_image_inf(image_path, IMAGE_SIZE)

    # Pass the image to the CNN
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "sos "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = tokenizer([decoded_caption])[:, :-1]
        #tokenized_caption = tokenizer.predict([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "eos":
            break
        decoded_caption += " " + sampled_token

    return decoded_caption.replace("sos ", "")


import tensorflow as tf
import numpy as np

# 推論ループをグラフ化するためのヘルパー関数
# @tf.functionデコレータにより、この関数は高速な計算グラフにコンパイルされます
@tf.function
def run_inference(encoded_img, caption_model, tokenizer, max_len):
    """
    高速化された推論ループを実行します (tf.while_loopを使用)。
    """
    sos_id = tokenizer(["sos"])[0, 0]
    eos_id = tokenizer(["eos"])[0, 0]

    # ループで使う変数の初期値を定義
    initial_i = tf.constant(0)
    initial_tokens = tf.expand_dims([sos_id], axis=0)
    
    # ループの本体処理を定義する内部関数
    def body(i, decoded_tokens):
        mask = tf.math.not_equal(decoded_tokens, 0)
        predictions = caption_model.decoder(
            decoded_tokens, encoded_img, training=False, mask=mask
        )
        last_prediction = predictions[:, -1, :]
        sampled_token_id = tf.argmax(last_prediction, axis=1, output_type=tf.int64)
        
        # 新しいトークンを連結
        decoded_tokens = tf.concat([decoded_tokens, [sampled_token_id]], axis=1)
        
        # 次のループのために変数を返す (カウンタをインクリメント)
        return i + 1, decoded_tokens

    # ループの継続条件を定義する内部関数
    def cond(i, decoded_tokens):
        # 最後に生成されたトークンを取得
        last_token = decoded_tokens[0, -1]
        
        # カウンタがmax_len未満 かつ 最後のトークンがeosでないならループを継続
        continue_condition = tf.logical_and(
            i < max_len - 1, 
            tf.not_equal(last_token, eos_id)
        )
        return continue_condition

    # tf.while_loopを実行
    _, final_tokens = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=[initial_i, initial_tokens], # ループで使う変数の初期値
        # ▼▼▼【ここが最重要ポイント】▼▼▼
        # 形状が変化することをTensorFlowに伝える
        shape_invariants=[
            initial_i.get_shape(),  # iの形状は不変
            tf.TensorShape([1, None]) # decoded_tokensの形状は(バッチサイズ=1, 長さ=可変)
        ]
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    )

    return final_tokens

def generate_caption_fast(image_path, caption_model, tokenizer, SEQ_LENGTH, IMAGE_SIZE, index_lookup):
    """
    高速化された画像キャプション生成関数
    """
    max_decoded_sentence_length = SEQ_LENGTH - 1

    # 1. 画像の読み込みとエンコード (この部分は変更なし)
    img = read_image_inf(image_path, IMAGE_SIZE)
    img = caption_model.cnn_model(img)
    encoded_img = caption_model.encoder(img, training=False)

    # 2. 高速化された推論関数を呼び出し、トークンIDのシーケンスを取得
    result_tokens_tensor = run_inference(
        encoded_img, caption_model, tokenizer, max_decoded_sentence_length
    )

    # 3. テンソルをNumPy配列に変換
    result_tokens = result_tokens_tensor.numpy()[0]

    # 4. トークンIDのシーケンスを単語に変換し、最終的なキャプション文字列を生成
    # ループの外で一度だけ文字列化処理を行う
    decoded_caption_list = []
    for token_id in result_tokens:
        # 開始トークンはスキップ
        if token_id == tokenizer(["sos"])[0, 0].numpy():
            continue
        # 終了トークンが見つかったら終了
        if token_id == tokenizer(["eos"])[0, 0].numpy():
            break
        # ルックアップテーブルを使ってIDを単語に変換
        word = index_lookup.get(token_id, "")  # 不明なIDは空文字に
        decoded_caption_list.append(word)

    return " ".join(decoded_caption_list)

import tensorflow as tf
import numpy as np
import time

# --- ヘルパー関数 (バッチ処理に対応) ---

def read_images_batch(image_paths, image_size):
    """
    画像パスのリストを受け取り、前処理して1つのバッチテンソルにまとめる。
    """
    batch_images = []
    for path in image_paths:
        # この部分は元のread_image_inf関数と同様の処理を想定
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size, method="bicubic")
        img = tf.image.convert_image_dtype(img, tf.float32)
        batch_images.append(img)
    
    # 画像のリストを1つのテンソルにスタックし、(batch_size, height, width, 3)の形状にする
    return tf.stack(batch_images, axis=0)


# --- 推論関数 (バッチ処理に対応) ---

#@tf.function(input_signature=[
#    # バッチサイズをNoneにすることで、任意の枚数の入力に対応
#    tf.TensorSpec(shape=[None, 256, 768], dtype=tf.float32), 
#    tf.keras.Model,
#    tf.keras.layers.TextVectorization,
#    tf.TensorSpec(shape=[], dtype=tf.int32)
#])
@tf.function
def run_inference_batch(encoded_imgs, caption_model, tokenizer, max_len):
    """
    バッチ化されたエンコード済み画像から、トークンIDのバッチを生成する。
    """
    batch_size = tf.shape(encoded_imgs)[0]
    sos_id = tokenizer(["sos"])[0, 0]
    eos_id = tokenizer(["eos"])[0, 0]

    # バッチサイズ分の"sos"トークンで初期化
    initial_tokens = tf.fill([batch_size, 1], sos_id)

    def body(i, decoded_tokens):
        # バッチ内の全画像に対して一度にデコーダーを適用
        mask = tf.math.not_equal(decoded_tokens, 0)
        predictions = caption_model.decoder(
            decoded_tokens, encoded_imgs, training=False, mask=mask
        )
        last_prediction = predictions[:, -1, :]
        sampled_token_ids = tf.argmax(last_prediction, axis=1, output_type=tf.int64)
        
        # (batch_size, 1) の形状に変換して連結
        sampled_token_ids = tf.expand_dims(sampled_token_ids, axis=1)
        decoded_tokens = tf.concat([decoded_tokens, sampled_token_ids], axis=1)
        return i + 1, decoded_tokens

    def cond(i, decoded_tokens):
        # バッチ処理では、単純にmax_lenまでループを回すのが最もシンプル
        return i < max_len - 1

    _, final_tokens = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=[tf.constant(0), initial_tokens],
        shape_invariants=[
            tf.TensorSpec(shape=[], dtype=tf.int32),
            # バッチサイズもシーケンス長も可変
            tf.TensorSpec(shape=[None, None], dtype=tf.int64) 
        ]
    )

    return final_tokens


# --- メイン関数 (バッチ処理に対応) ---

def generate_captions_batch(image_paths, caption_model, tokenizer, seq_length, image_size, index_lookup):
    """
    画像のパスリストを受け取り、キャプションのリストを返すバッチ推論関数。
    """
    # 1. 画像をバッチで読み込み、前処理
    batch_img_tensors = read_images_batch(image_paths, image_size)

    # 2. CNNとEncoderで画像をエンコード (バッチごと)
    #    モデルが(None, H, W, C)の入力を受け付けるように作られている必要がある
    encoded_imgs = caption_model.cnn_model(batch_img_tensors)
    encoded_imgs = caption_model.encoder(encoded_imgs, training=False)

    # 3. バッチ推論を実行
    result_tokens_tensor = run_inference_batch(
        encoded_imgs, caption_model, tokenizer, seq_length
    )
    result_tokens_batch = result_tokens_tensor.numpy()

    # 4. 結果を後処理し、キャプションのリストを作成
    captions = []
    sos_id_val = tokenizer(["sos"])[0, 0].numpy()
    eos_id_val = tokenizer(["eos"])[0, 0].numpy()
    
    for result_tokens in result_tokens_batch:
        caption_list = []
        for token_id in result_tokens:
            if token_id == sos_id_val:
                continue
            if token_id == eos_id_val:
                break
            word = index_lookup.get(token_id, "")
            caption_list.append(word)
        captions.append(" ".join(caption_list))
        
    return captions
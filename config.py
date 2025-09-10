import yaml
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import configargparse

def get_config():
    """
    configargparseを使ってプロジェクトの設定を管理する。
    優先順位: コマンドライン引数 > 環境変数 > 設定ファイル > デフォルト値
    """
    parser = configargparse.ArgParser(
        auto_env_var_prefix="APP_", # 環境変数のプレフィックス (例: APP_BATCH_SIZE)
        default_config_files=['config.yaml'], # デフォルトで読み込む設定ファイル
        ignore_unknown_config_file_keys=True,
        config_file_parser_class = configargparse.YAMLConfigFileParser,
        description="Image Captioning Project Settings"
    )

    # --- configargparseの基本設定 ---
    parser.add_argument(
        '-c', '--config',
        is_config_file=True,
        help='設定ファイルのパス'
    )

    parser.add_argument('--cnnmodel', type=str, default="EfficientNetB0", help='Name of ImageEncoder')
    # --- モデルのハイパーパラメータ ---
    # 元のIMAGE_SIZEは2つの引数に分割し、より設定しやすくします
    parser.add_argument('--IMAGE_WIDTH', type=int, default=299, help='画像の幅')
    parser.add_argument('--IMAGE_HEIGHT', type=int, default=299, help='画像の高さ')
    parser.add_argument('--MAX_VOCAB_SIZE', type=int, default=2000000, help='ボキャブラリの最大サイズ')
    parser.add_argument('--SEQ_LENGTH', type=int, default=64, help='シーケンスの固定長')
    parser.add_argument('--EMBED_DIM', type=int, default=512, help='埋め込みの次元数')
    parser.add_argument('--NUM_HEADS', type=int, default=6, help='Self-Attentionのヘッド数')
    parser.add_argument('--FF_DIM', type=int, default=1024, help='Feed-Forwardネットワークのユニット数')
    parser.add_argument('--NUM_LAYERS_DECODER', type=int, default=4, help='デコーダーの層数')
    
    # --- 学習設定 ---
    parser.add_argument('--SHUFFLE_DIM', type=int, default=512, help='データセットシャッフル時のバッファサイズ')
    parser.add_argument('--BATCH_SIZE', type=int, default=64, help='バッチサイズ')
    parser.add_argument('--EPOCHS', type=int, default=30, help='エポック数')
    parser.add_argument('--EARLY_STOP', type=int, default=8, help='早期終了(Early Stopping)のpatience値')

    # --- データセット関連 ---
    parser.add_argument('--REDUCE_DATASET', action='store_true', help='データセットを削減して使用するか')
    parser.add_argument('--MOD_DATASET', action='store_true', help='データセットに変更を加えるか')
    parser.add_argument('--NUM_TRAIN_IMG', type=int, default=68363, help='訓練画像の数')
    parser.add_argument('--NUM_VALID_IMG', type=int, default=20000, help='検証画像の数')
    parser.add_argument('--TRAIN_SET_AUG', action='store_true', help='訓練セットでデータ拡張を行うか')
    parser.add_argument('--VALID_SET_AUG', action='store_true', help='検証セットでデータ拡張を行うか')
    parser.add_argument('--TEST_SET', action='store_true', help='テストセットを使用するか')
    parser.add_argument('--num_captions_per_image', type=int, default=1, help='画像あたりのキャプション数')

    # --- パス関連 ---
    parser.add_argument('--train_data_json_path', type=str, default="ImageScript_dataset/A2_train.json", help='訓練データJSONのパス')
    parser.add_argument('--valid_data_json_path', type=str, default="ImageScript_dataset/A2_val.json", help='検証データJSONのパス')
    parser.add_argument('--test_data_json_path', type=str, default="ImageScript_dataset/A2_test.json", help='テストデータJSONのパス')
    parser.add_argument('--text_data_json_path', type=str, default="ImageScript_dataset/text_data.json", help='テキストデータJSONのパス')
    parser.add_argument('--SAVE_DIR', type=str, default="save_train_dir/is_20250725/", help='学習結果の保存先ディレクトリ')
    parser.add_argument('--IMAGE_DIR', type=str, default="ImageScript_dataset/images", help='画像が格納されているディレクトリ')

    parser.add_argument('--warmup_steps', type=int, default=4000, help='Warm Up ステップ数')
    parser.add_argument('--EVAL_BATCH_SIZE', type=int, default=8, help='バッチサイズ')

    # --- 設定をパース ---
    args = parser.parse_args()

    # --- 派生パスの生成とディレクトリ作成 ---
    # 元のクラスで行っていた後処理をここに記述
    save_dir = Path(args.SAVE_DIR)
    args.LOG_DIR = save_dir / "log/"
    args.tokernizer_path = save_dir / "tokenizer.keras"
    args.get_model_config_path = save_dir / "config_train.json"
    args.get_model_weights_path = save_dir / "model.weights.h5"
    args.get_best_model_path = save_dir / "best_model.keras"
    
    # 派生したIMAGE_SIZEをタプルとして追加
    args.IMAGE_SIZE: Tuple[int, int] = (args.IMAGE_WIDTH, args.IMAGE_HEIGHT)

    # 必要に応じてディレクトリを作成
    save_dir.mkdir(parents=True, exist_ok=True)
    args.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    return args

# --- メインの実行部分 ---
if __name__ == '__main__':
    # 設定を取得
    stg = get_config()

    # 設定内容をきれいに表示して確認
    print("--- Loaded Settings ---")
    print(stg)
    print("-----------------------")
    
    # 例: 設定値へのアクセス
    print(f"Batch size: {stg.BATCH_SIZE}")
    print(f"Save directory: {stg.SAVE_DIR}")
    print(f"Image size: {stg.IMAGE_SIZE}")

    import pdb; pdb.set_trace()

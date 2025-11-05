import pickle
import numpy as np
import os
from PIL import Image

def unpickle(file):
    """
    CIFAR-10のバッチファイルを読み込む
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_cifar_images(batch_file, output_dir):
    """
    バッチファイルから画像を読み込み、PNGとして保存する
    """
    # 出力先フォルダがなければ作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"'{batch_file}' を読み込んでいます...")
    data_dict = unpickle(batch_file)
    
    # dataは (10000, 3072) のNumpy配列
    images = data_dict[b'data']
    # 3072 = 3 * 32 * 32 (Red, Green, Blue)
    
    images = images.reshape(10000, 3, 32, 32)
    # PIL (Pillow) で扱えるように (N, 32, 32, 3) の形式に軸を入れ替える
    images = images.transpose(0, 2, 3, 1) 

    print(f"'{output_dir}' フォルダに 10,000 枚の画像を保存しています...")
    
    for i in range(len(images)):
        img_array = images[i]
        img = Image.fromarray(img_array)
        
        # ファイル名を 0.png, 1.png ... とする
        filename = f"{i}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)

    print("完了しました。")

# --- メイン処理 ---
if __name__ == "__main__":
    # test_batchファイルのパス
    test_batch_path = 'test_batch' 
    
    # 保存先フォルダ名
    output_folder = 'images' 
    
    if os.path.exists(test_batch_path):
        save_cifar_images(test_batch_path, output_folder)
    else:
        print(f"エラー: '{test_batch_path}' が見つかりません。")
        print("cifar-10-batches-py フォルダから 'test_batch' ファイルを")
        print("このスクリプトと同じ場所にコピーしてください。")
import pickle
import os
import csv

def unpickle(file):
    """
    CIFAR-10のバッチファイルを読み込む
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def create_labels_csv(batch_file, output_csv_file):
    """
    test_batchからラベル情報を読み込み、CSVファイルを作成する
    """
    
    # CIFAR-10の公式クラス名（順番が重要）
    class_names = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]

    print(f"'{batch_file}' を読み込んでいます...")
    data_dict = unpickle(batch_file)
    
    # b'labels' キーでラベルのリスト（[3, 8, 8, 0, ...] のような数値）を取得
    labels = data_dict[b'labels']

    print(f"'{output_csv_file}' を作成しています...")
    
    try:
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # ヘッダー行を書き込む
            writer.writerow(["index", "label_id", "label_name"])
            
            # 0から9999までのデータを書き込む
            for i in range(len(labels)):
                label_id = labels[i]
                label_name = class_names[label_id]
                
                # [index, ラベルID, クラス名] の形式で書き込む
                writer.writerow([i, label_id, label_name])

        print(f"完了しました。'{output_csv_file}' が作成されました。")

    except Exception as e:
        print(f"エラーが発生しました: {e}")

# --- メイン処理 ---
if __name__ == "__main__":
    # test_batchファイルのパス
    test_batch_path = 'test_batch' 
    
    # 出力するCSVファイル名
    output_csv = 'cifar10_labels.csv' 
    
    if os.path.exists(test_batch_path):
        create_labels_csv(test_batch_path, output_csv)
    else:
        print(f"エラー: '{test_batch_path}' が見つかりません。")
        print("cifar-10-batches-py フォルダから 'test_batch' ファイルを")
        print("このスクリプトと同じ場所にコピーしてください。")
import os
import asyncio
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import google.generativeai as genai
import time
import pickle
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# ===============================
# ログ設定
# ===============================
os.makedirs("logs", exist_ok=True)
log_file = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# ===============================
# 初期設定
# ===============================
load_dotenv()

API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4")
]
API_KEYS = [k for k in API_KEYS if k]
MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

CHUNK_SIZE = int(os.getenv("INPUT1_CHUNK_SIZE", "500"))
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "match_results.csv")
INPUT2_EMBED_FILE = "input2_embeddings.pkl"

# 各キーの利用カウンタ（スレッドセーフ）
KEY_USAGE = {key[-6:]: 0 for key in API_KEYS}
USAGE_LOCK = asyncio.Lock()

# ===============================
# Embedding取得関数
# ===============================
def get_embedding(text, api_key, retries=3, delay=2):
    key_tail = api_key[-6:]
    for attempt in range(retries):
        try:
            genai.configure(api_key=api_key)
            result = genai.embed_content(model=MODEL, content=text)
            return np.array(result["embedding"], dtype=np.float32), key_tail
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))  # 指数バックオフ
            else:
                logging.warning(f"[{key_tail}] Failed after {retries} retries: {e}")
    return None, key_tail

# ===============================
# 入力CSV2データ読み込み or キャッシュ生成
# ===============================
def prepare_input2_embeddings(input2_csv_path):
    if os.path.exists(INPUT2_EMBED_FILE):
        logging.info("Loading cached 入力CSV2 embeddings...")
        with open(INPUT2_EMBED_FILE, "rb") as f:
            data = pickle.load(f)
        return data["items"], data["embeddings"], data["embedding_status"]

    logging.info("Generating 入力CSV2 embeddings...")
    input2_df = pd.read_csv(input2_csv_path, header=None)
    input2_items = input2_df[0].astype(str).tolist()
    input2_embeddings = []
    input2_embedding_status = [] # Track success/failure

    # 入力CSV2は件数が少ないので1つのキーで処理
    key = API_KEYS[0]
    for item in tqdm(input2_items, desc="Embedding 入力CSV2"):
        emb, _ = get_embedding(item, key)
        if emb is not None:
            input2_embeddings.append(emb)
            input2_embedding_status.append(True)
        else:
            input2_embeddings.append(np.zeros(768))
            input2_embedding_status.append(False)
    
    input2_embeddings = np.vstack(input2_embeddings)
    with open(INPUT2_EMBED_FILE, "wb") as f:
        pickle.dump({"items": input2_items, "embeddings": input2_embeddings, "embedding_status": input2_embedding_status}, f)
    return input2_items, input2_embeddings, input2_embedding_status

# ===============================
# 入力CSV1のチャンク処理（並列対応版）
# ===============================
async def process_chunk(chunk_id, input1_list, input2_items, input2_embeddings, input2_embedding_status, api_key):
    start = time.time()
    results = []
    key_tail = api_key[-6:]

    logging.info(f"Chunk {chunk_id} 開始 ({len(input1_list)} 件) [Key: {key_tail}]")

    # 並列でEmbedding取得
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=10) as executor:
        tasks = [
            loop.run_in_executor(executor, get_embedding, item, api_key)
            for item in input1_list
        ]
        embeddings_with_keys = await asyncio.gather(*tasks)

    # 結果を集計
    input1_embeddings = []
    input1_embedding_status = [] # Track success/failure
    for (emb, used_key), item in zip(embeddings_with_keys, input1_list):
        if emb is not None:
            input1_embeddings.append(emb)
            input1_embedding_status.append(True)
            async with USAGE_LOCK:
                KEY_USAGE[used_key] = KEY_USAGE.get(used_key, 0) + 1
        else:
            input1_embeddings.append(np.zeros(input2_embeddings.shape[1]))
            input1_embedding_status.append(False)

    input1_embeddings = np.vstack(input1_embeddings)
    sims = cosine_similarity(input1_embeddings, input2_embeddings)

    # 類似度の高い入力CSV2項目を特定
    for i, (input1_text, sim_row) in enumerate(zip(input1_list, sims)):
        top_idx = np.argmax(sim_row)
        top_score = sim_row[top_idx]

        # Handle cases where embedding failed for either input1 or input2
        if not input1_embedding_status[i] or not input2_embedding_status[top_idx]:
            top_score = 0.0 # Set similarity to 0 if either embedding failed
            
        results.append({
            "入力CSV1項目名": input1_text,
            "入力CSV2項目名": input2_items[top_idx],
            "類似度": round(float(top_score), 4)
        })

    elapsed = time.time() - start
    avg_time = elapsed / len(input1_list)
    logging.info(
        f"Chunk {chunk_id} 完了 ({len(input1_list)} items) | "
        f"Time: {elapsed:.1f}s | Avg/item: {avg_time:.2f}s [Key: {key_tail}]"
    )
    
    return chunk_id, results

# ===============================
# メイン処理
# ===============================
async def main(input1_csv_path, input2_csv_path):
    start_time = time.time()
    logging.info("===== CSV項目類似度比較ツール 開始 =====")
    logging.info(f"使用APIキー数: {len(API_KEYS)}")
    logging.info(f"チャンクサイズ: {CHUNK_SIZE}件")

    # --- 入力CSV2準備 ---
    input2_items, input2_embeddings, input2_embedding_status = prepare_input2_embeddings(input2_csv_path)

    # --- 入力CSV1読み込み ---
    input1_df = pd.read_csv(input1_csv_path, header=None)
    input1_items = input1_df[0].astype(str).tolist()

    # --- 途中再開対応 ---
    done_items = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            done_df = pd.read_csv(OUTPUT_CSV)
            # カラム名を確認（複数のパターンに対応）
            if "入力CSV1項目名" in done_df.columns:
                done_items = set(done_df["入力CSV1項目名"].tolist())
            elif done_df.columns[0]:  # 最初のカラムを使用
                done_items = set(done_df.iloc[:, 0].tolist())
            logging.info(f"{len(done_items)}件は既に処理済み。スキップします。")
        except Exception as e:
            logging.warning(f"既存CSVの読み込みに失敗: {e}")
            logging.info("既存CSVを削除するか、ファイル名を変更して再実行してください。")
            logging.info("最初から処理を開始します。")

    remaining = [t for t in input1_items if t not in done_items]
    total = len(remaining)
    logging.info(f"残り {total} 件を処理開始します。")

    # チャンクを作成
    chunks = []
    for i in range(0, total, CHUNK_SIZE):
        chunk = remaining[i:i+CHUNK_SIZE]
        chunk_id = i // CHUNK_SIZE + 1
        chunks.append((chunk_id, chunk))

    # ファイル書き込み用のロック
    write_lock = asyncio.Lock()
    processed_count = 0

    # 4つずつ並列処理
    num_workers = len(API_KEYS)
    for batch_start in range(0, len(chunks), num_workers):
        batch = chunks[batch_start:batch_start + num_workers]
        
        # 各チャンクに異なるAPIキーを割り当て
        tasks = []
        for idx, (chunk_id, chunk_data) in enumerate(batch):
            api_key = API_KEYS[idx % len(API_KEYS)]
            tasks.append(process_chunk(chunk_id, chunk_data, input2_items, input2_embeddings, input2_embedding_status, api_key))
        
        # 並列実行
        results = await asyncio.gather(*tasks)
        
        # 結果を順次書き込み（チャンクIDでソート）
        results.sort(key=lambda x: x[0])
        for chunk_id, chunk_results in results:
            async with write_lock:
                header_flag = not os.path.exists(OUTPUT_CSV)
                pd.DataFrame(chunk_results).to_csv(
                    OUTPUT_CSV, mode="a", header=header_flag, index=False
                )
                processed_count += len(chunk_results)

        # 進捗表示
        elapsed_total = time.time() - start_time
        speed = processed_count / elapsed_total if elapsed_total > 0 else 0
        eta = (total - processed_count) / speed if speed > 0 else 0

        logging.info(
            f"進捗 {processed_count}/{total} "
            f"({processed_count/total*100:.1f}%) | "
            f"速度: {speed:.2f}件/秒 | 残り推定: {eta/60:.1f}分"
        )
        async with USAGE_LOCK:
            logging.info(
                "API使用状況: " + ", ".join([f"{k}:{v}" for k, v in KEY_USAGE.items()])
            )

    total_time = time.time() - start_time
    logging.info(f"全処理完了！総時間: {total_time/60:.1f}分")
    logging.info(f"結果ファイル: {OUTPUT_CSV}")
    logging.info(f"ログファイル: {log_file}")

# ===============================
# 実行
# ===============================
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python csv_content_matcher.py input1.csv input2.csv")
        exit(1)

    asyncio.run(main(sys.argv[1], sys.argv[2]))

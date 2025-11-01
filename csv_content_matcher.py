import os
import asyncio
import pandas as pd
import numpy as np
from itertools import cycle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import google.generativeai as genai
import time
import pickle
import logging
from datetime import datetime

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
API_CYCLE = cycle(API_KEYS)
MODEL = "text-embedding-004"

CHUNK_SIZE = 500
OUTPUT_CSV = "match_results.csv"
B_EMBED_FILE = "b_embeddings.pkl"

# 各キーの利用カウンタ
KEY_USAGE = {key[-6:]: 0 for key in API_KEYS}

# ===============================
# Embedding取得関数
# ===============================
def get_embedding(text, api_key, retries=3, delay=2):
    key_tail = api_key[-6:]
    for attempt in range(retries):
        try:
            genai.configure(api_key=api_key)
            result = genai.embed_content(model=MODEL, content=text)
            KEY_USAGE[key_tail] += 1
            return np.array(result["embedding"], dtype=np.float32)
        except Exception as e:
            logging.warning(f"[{key_tail}] Retry {attempt+1}/{retries} - {e}")
            time.sleep(delay)
    logging.error(f"[{key_tail}] Failed to embed after {retries} retries.")
    return None

# ===============================
# B社データ読み込み or キャッシュ生成
# ===============================
def prepare_b_embeddings(b_csv_path):
    if os.path.exists(B_EMBED_FILE):
        logging.info("📦 Loading cached B社 embeddings...")
        with open(B_EMBED_FILE, "rb") as f:
            data = pickle.load(f)
        return data["titles"], data["embeddings"]

    logging.info("⚙️ Generating B社 embeddings...")
    b_df = pd.read_csv(b_csv_path, header=None)
    b_titles = b_df[0].astype(str).tolist()
    b_embeddings = []
    for i, title in enumerate(tqdm(b_titles, desc="Embedding B社")):
        key = next(API_CYCLE)
        emb = get_embedding(title, key)
        if emb is not None:
            b_embeddings.append(emb)
        else:
            b_embeddings.append(np.zeros(768))
    b_embeddings = np.vstack(b_embeddings)
    with open(B_EMBED_FILE, "wb") as f:
        pickle.dump({"titles": b_titles, "embeddings": b_embeddings}, f)
    return b_titles, b_embeddings

# ===============================
# A社のチャンク処理
# ===============================
async def process_chunk(a_list, b_titles, b_embeddings):
    loop = asyncio.get_event_loop()
    start = time.time()
    results = []

    a_embeddings = []
    for title in tqdm(a_list, desc="Embedding A社", leave=False):
        key = next(API_CYCLE)
        emb = await loop.run_in_executor(None, get_embedding, title, key)
        if emb is not None:
            a_embeddings.append(emb)
        else:
            a_embeddings.append(np.zeros(b_embeddings.shape[1]))

    a_embeddings = np.vstack(a_embeddings)
    sims = cosine_similarity(a_embeddings, b_embeddings)

    for a_text, sim_row in zip(a_list, sims):
        top_idx = np.argmax(sim_row)
        top_score = sim_row[top_idx]
        results.append({
            "A社コンテンツ名": a_text,
            "B社コンテンツ名": b_titles[top_idx],
            "類似度": round(float(top_score), 4)
        })

    elapsed = time.time() - start
    avg_time = elapsed / len(a_list)
    logging.info(
        f"🧩 Chunk done ({len(a_list)} items) | "
        f"Time: {elapsed:.1f}s | Avg/item: {avg_time:.2f}s"
    )
    return results

# ===============================
# メイン処理
# ===============================
async def main(a_csv_path, b_csv_path):
    start_time = time.time()
    logging.info("===== E-Learning コンテンツ比較開始 =====")

    # --- B社準備 ---
    b_titles, b_embeddings = prepare_b_embeddings(b_csv_path)

    # --- A社読み込み ---
    a_df = pd.read_csv(a_csv_path, header=None)
    a_titles = a_df[0].astype(str).tolist()

    # --- 途中再開対応 ---
    done_titles = set()
    if os.path.exists(OUTPUT_CSV):
        done_df = pd.read_csv(OUTPUT_CSV)
        done_titles = set(done_df["A社コンテンツ名"].tolist())
        logging.info(f"🔄 {len(done_titles)}件は既に処理済み。スキップします。")

    remaining = [t for t in a_titles if t not in done_titles]
    total = len(remaining)
    logging.info(f"🚀 残り {total} 件を処理開始します。")

    for i in range(0, total, CHUNK_SIZE):
        chunk = remaining[i:i+CHUNK_SIZE]
        chunk_no = i // CHUNK_SIZE + 1
        logging.info(f"➡️ Chunk {chunk_no} 開始 ({len(chunk)} 件)")

        res = await process_chunk(chunk, b_titles, b_embeddings)

        # 書き込み
        header_flag = not os.path.exists(OUTPUT_CSV)
        pd.DataFrame(res).to_csv(OUTPUT_CSV, mode="a", header=header_flag, index=False)

        elapsed_total = time.time() - start_time
        processed = i + len(chunk)
        speed = processed / elapsed_total
        est_total_time = total / speed if speed > 0 else 0
        eta = est_total_time - elapsed_total

        logging.info(
            f"✅ Chunk {chunk_no} 完了。進捗 {processed}/{total} "
            f"({processed/total*100:.1f}%) | "
            f"速度: {speed:.2f}件/秒 | 残り推定: {eta/60:.1f}分"
        )
        logging.info(
            "🔑 API使用状況: " + ", ".join([f"{k}:{v}" for k, v in KEY_USAGE.items()])
        )

    total_time = time.time() - start_time
    logging.info(f"🎉 全処理完了！総時間: {total_time/60:.1f}分")
    logging.info("ログ出力先: " + log_file)

# ===============================
# 実行
# ===============================
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python compare_contents.py A_company.csv B_company.csv")
        exit(1)

    asyncio.run(main(sys.argv[1], sys.argv[2]))

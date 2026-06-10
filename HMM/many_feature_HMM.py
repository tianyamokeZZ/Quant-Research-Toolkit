import pandas as pd
from pathlib import Path

# === 1️⃣ 基本配置 ===
base_dir = Path(".")
files = [
    "eth_1h_history.csv",
    "eth_1h_buy_sell.csv",
    "eth_1h_coinmarketcap.csv",
    "eth_1h_Dvol.csv",
    "eth_1h_eth_trans_btc.csv",
    "eth_1h_Liquidation.csv",
    "eth_1h_marketoi.csv",
    "eth_1h_oi_weight_ohlc.csv",
    "eth_1h_stablecoin-margin.csv",
    "eth_1h_timeseries.csv",
    "eth_1h_vol_weight_ohlc.csv",
]

# === 2️⃣ 时间范围 ===
train_start = pd.Timestamp("2024-01-01", tz="UTC")
train_end   = pd.Timestamp("2025-02-01", tz="UTC")
test_end    = pd.Timestamp("2025-03-01", tz="UTC")

def parse_datetime(df):
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    elif "ts" in df.columns:
        ts = df["ts"].astype(float)
        unit = "ms" if (ts > 1e12).any() else "s"
        dt = pd.to_datetime(ts, unit=unit, errors="coerce", utc=True)
    else:
        raise ValueError("未找到 datetime 或 ts 列。")
    return dt.dt.floor("H")

# === 3️⃣ 循环读取文件并对齐时间 ===
aligned_data = {}

for fname in files:
    fpath = base_dir / fname
    if not fpath.exists():
        print(f"❌ 文件不存在: {fname}")
        continue

    df = pd.read_csv(fpath)
    df["datetime"] = parse_datetime(df)
    df = df.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"]).sort_values("datetime")

    aligned_data[fname] = df

    print(f"✅ 读取 {fname:<30} 行数: {len(df):>6} | 时间: {df['datetime'].min()} → {df['datetime'].max()}")

# === 4️⃣ 统一时间范围（取交集）
time_min = max(df["datetime"].min() for df in aligned_data.values())
time_max = min(df["datetime"].max() for df in aligned_data.values())
print(f"\n🕓 对齐后的统一时间范围: {time_min} → {time_max}")

# === 5️⃣ 数据集划分（使用行情文件为索引）
df_base = aligned_data["eth_1h_history.csv"]
df_base = df_base[(df_base["datetime"] >= train_start) & (df_base["datetime"] < test_end)].reset_index(drop=True)

train_mask = (df_base["datetime"] >= train_start) & (df_base["datetime"] < train_end)
test_mask  = (df_base["datetime"] >= train_end)   & (df_base["datetime"] < test_end)

print(f"\n📊 数据划分：")
print(f"  训练集时间: {train_start} → {train_end} | {train_mask.sum()} 条")
print(f"  测试集时间: {train_end} → {test_end} | {test_mask.sum()} 条")

# === 6️⃣ 输出结果说明 ===
print("\n✅ 所有文件读取完毕并时间对齐，可直接用于特征工程与 HMM 阶段。")

# === 7️⃣ 特征构造与选择 ===
import numpy as np

print("\n🚀 开始特征提取与对齐...")

# --- 取行情数据并计算收益与波动 ---
hist = aligned_data["eth_1h_history.csv"].copy()
hist["log_return"] = np.log(hist["c"] / hist["c"].shift(1))
hist["volatility_24h"] = hist["log_return"].rolling(24).std()

# --- 各文件核心列 ---
feat_map = {
    "eth_1h_Dvol.csv": ("c", "dvol"),
    "eth_1h_marketoi.csv": ("c", "open_interest"),
    "eth_1h_oi_weight_ohlc.csv": ("c", "funding_oiw"),
    "eth_1h_Liquidation.csv": None,  # 特殊处理
    "eth_1h_eth_trans_btc.csv": ("c", "eth_btc_ratio"),
    "eth_1h_stablecoin-margin.csv": ("c", "margin_ratio"),
    "eth_1h_coinmarketcap.csv": ("volume", "volume_cmc"),
    "eth_1h_timeseries.csv": ("posts_created", "sentiment"),
}

# --- 初始DF ---
features = hist[["datetime", "log_return", "volatility_24h"]].copy()

# --- 合并其他特征 ---
for fname, cols in feat_map.items():
    df = aligned_data[fname].copy()

    if fname == "eth_1h_Liquidation.csv":
        if all(c in df.columns for c in ["longLiquidationUsd", "shortLiquidationUsd"]):
            df["liq_ratio"] = (df["longLiquidationUsd"] - df["shortLiquidationUsd"]) / (
                df["longLiquidationUsd"] + df["shortLiquidationUsd"] + 1e-9
            )
            df = df[["datetime", "liq_ratio"]]
        else:
            continue
    else:
        if cols[0] not in df.columns:
            continue
        df = df[["datetime", cols[0]]].rename(columns={cols[0]: cols[1]})

    features = pd.merge(features, df, on="datetime", how="outer")

features = features.sort_values("datetime").dropna().reset_index(drop=True)
print(f"✅ 特征汇总完成，维度: {features.shape}")

# === 8️⃣ 聚类分析：看特征相似性 ===
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

train_feats = features[(features["datetime"] >= train_start) & (features["datetime"] < train_end)].copy()
feat_cols = [c for c in train_feats.columns if c not in ["datetime"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_feats[feat_cols])

corr = np.corrcoef(X_scaled.T)
corr_df = pd.DataFrame(corr, index=feat_cols, columns=feat_cols)

# --- 相关矩阵热力图 ---
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, cmap="coolwarm", center=0, square=True)
plt.title("特征相关矩阵 (训练集)")
plt.tight_layout()
plt.show()

# --- 层次聚类树状图 ---
linked = linkage(X_scaled.T, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linked, labels=feat_cols, leaf_rotation=45)
plt.title("特征聚类树状图 (Ward linkage)")
plt.tight_layout()
plt.show()

print("\n📈 已绘制特征相关性与聚类，可据此挑选最终输入 HMM 的核心特征。")

# ==============================================================
#   ETH 市场状态识别（使用 pomegranate.DenseHMM + 简化版特征 + 拒识状态）
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import RobustScaler
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal
import plotly.express as px

# ========== 1️⃣ 数据读取 ==========
df = pd.read_csv("eth_1h_history.csv")

if not all(col in df.columns for col in ['datetime', 'c', 'h', 'l', 'vol']):
    raise ValueError("❌ 数据文件必须包含列：datetime, c, h, l, vol")

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

train_start = pd.Timestamp("2024-01-01")
train_end   = pd.Timestamp("2025-02-01")
test_end    = pd.Timestamp("2025-03-01")

# ========== 2️⃣ 特征构建（简化版）==========
eps = 1e-9
df['Return'] = df['c'].pct_change()
df['Momentum_10h'] = df['c'].pct_change(10)
df['Volatility_10h'] = df['Return'].rolling(window=10).std()
df['Volume_Change'] = df['vol'].pct_change()

# 去除缺失值
df = df.dropna(subset=['Return', 'Momentum_10h', 'Volatility_10h', 'Volume_Change']).reset_index(drop=True)

# 选定特征
selected_feats = ['Return', 'Momentum_10h', 'Volatility_10h', 'Volume_Change']

# ========== 3️⃣ 标准化与数据划分 ==========
train_mask = (df['datetime'] >= train_start) & (df['datetime'] < train_end)
test_mask  = (df['datetime'] >= train_end)   & (df['datetime'] < test_end)

scaler = RobustScaler()
scaler.fit(df.loc[train_mask, selected_feats])
df[selected_feats] = scaler.transform(df[selected_feats])

# ========== 4️⃣ HMM 模型定义与训练 ==========
X_train = df.loc[train_mask, selected_feats].values.astype(np.float64)

mask = ~np.isfinite(X_train)
bad_cols = np.array(selected_feats)[mask.any(axis=0)]
if len(bad_cols) > 0:
    print("⚠️ 存在 NaN/Inf 的特征：", bad_cols)

assert np.isfinite(X_train).all(), "NaN 或 Inf 存在，请检查标准化阶段"

n_states = 3
n_features = X_train.shape[1]

# 初始化观测分布
distributions = [
    Normal(
        means=np.random.normal(0, 0.5, n_features),
        covs=np.ones(n_features),
        covariance_type='diag'
    )
    for _ in range(n_states)
]

# 初始化并训练 DenseHMM
model = DenseHMM(
    distributions=distributions,
    init='random',
    verbose=True,
    max_iter=50,
    tol=1e-8,
    random_state=42
)

print("🚀 开始训练 HMM 模型...")
model.fit([X_train])
print("✅ 训练完成！")

# ========== 5️⃣ 保存模型 ==========
torch.save(model, "trained_hmm_simple.pt")
print("💾 模型已保存为 trained_hmm_simple.pt")

# ========== 6️⃣ 预测状态 + 拒识处理 ==========

# --- 转换工具函数 ---
def to_numpy(x):
    """兼容 Tensor / numpy 输入，安全转为 numpy.ndarray"""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return np.asarray(x)

# === 1️⃣ 计算后验概率 ===
log_gamma = model.predict_proba([X_train])[0]
log_gamma = to_numpy(log_gamma)               # ✅ 关键修正！
gamma = np.exp(log_gamma)
max_p = gamma.max(axis=1)

# === 2️⃣ 置信度阈值设定 ===
tau = np.quantile(max_p, 0.3)
print(f"📉 置信度阈值 tau = {tau:.4f}")

# === 3️⃣ 标记低置信度样本为 “未知态 -1” ===
state_hard = gamma.argmax(axis=1)
state_with_unknown = np.where(max_p >= tau, state_hard, -1)

df.loc[train_mask, 'state'] = state_with_unknown

# 定义颜色映射
color_map = {
    0: 'royalblue',   # 低波动 / 上升
    1: 'gold',        # 中性 / 震荡
    2: 'red',         # 高波动 / 下行
    -1: 'silver'      # 不确定
}

# 保留状态映射
df_plot = df.loc[train_mask, ['datetime', 'c', 'state']].copy()
df_plot['state_name'] = df_plot['state'].map({
    0: 'Low Vol (Blue)',
    1: 'Neutral (Yellow)',
    2: 'High Vol (Red)',
    -1: 'Uncertain (Gray)'
})

# 创建主图（收盘价线 + 状态散点）
import plotly.graph_objects as go

# --- 创建主图 ---
fig = go.Figure()

# --- 收盘价主线 ---
fig.add_trace(go.Scatter(
    x=df_plot['datetime'],
    y=df_plot['c'],
    mode='lines',
    line=dict(color='black', width=1),
    name='Close Price',
    hovertemplate='时间: %{x}<br>收盘价: %{y:.2f}<extra></extra>'
))

# --- 各状态颜色与标签 ---
state_labels = {
    0: 'Low Vol (Blue)',
    1: 'Neutral (Yellow)',
    2: 'High Vol (Red)',
    -1: 'Uncertain (Gray)'
}

color_map = {
    0: 'royalblue',
    1: 'gold',
    2: 'red',
    -1: 'silver'
}

# --- 叠加状态散点 ---
for s in sorted(color_map.keys()):
    sub = df_plot[df_plot['state'] == s]
    if sub.empty:
        continue
    fig.add_trace(go.Scatter(
        x=sub['datetime'],
        y=sub['c'],
        mode='markers',
        marker=dict(color=color_map[s], size=5, opacity=0.8),
        name=state_labels[s],
        hovertemplate='时间: %{x}<br>收盘价: %{y:.2f}<br>状态: ' + state_labels[s] + '<extra></extra>'
    ))

# --- 图表样式 ---
fig.update_layout(
    title=dict(
        text="ETH 市场状态识别（含收盘价走势 + 拒识状态）",
        x=0.5,
        xanchor="center"
    ),
    xaxis_title="时间",
    yaxis_title="ETH 收盘价",
    template="plotly_white",
    hovermode='x unified',
    height=650,
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        bgcolor="rgba(255,255,255,0.8)"
    )
)

fig.show()
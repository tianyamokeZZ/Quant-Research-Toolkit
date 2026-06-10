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
import plotly.graph_objects as go
import pickle

# ========== 1️⃣ 数据读取 ==========
df = pd.read_csv("eth_1h_history.csv")

if not all(col in df.columns for col in ['datetime', 'c', 'h', 'l', 'vol']):
    raise ValueError("❌ 数据文件必须包含列：datetime, c, h, l, vol")

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

train_start = pd.Timestamp("2024-01-01")
train_end   = pd.Timestamp("2025-02-01")
test_end    = pd.Timestamp("2025-03-01")

# ========== 2️⃣ 特征构建 ==========
eps = 1e-9

# ✅ 对数收益率与波动率
df['lr'] = np.log(df['c']).diff().fillna(0.0).rolling(24).mean()
df['volatility_20h'] = df['lr'].rolling(20, min_periods=1).std().fillna(0.0)

# ✅ 趋势特征：MA7 - MA30
df['MA7'] = df['c'].rolling(7*24).mean()
df['MA30'] = df['c'].rolling(30*24).mean()
df['MA_Spread'] = (df['MA7'] - df['MA30']) / (df['MA30'] + eps)

# ✅ 成交量变化
df['Volume_Change'] = df['vol'].pct_change().fillna(0.0)

# 去除缺失
df = df.dropna(subset=['lr', 'volatility_20h', 'MA_Spread', 'Volume_Change']).reset_index(drop=True)

# 选定特征
selected_feats = [
    # 'lr',
    'volatility_20h',
    'MA_Spread',
    # 'Volume_Change'
                  ]
# ========== 3️⃣ 标准化与数据划分 ==========
train_mask = (df['datetime'] >= train_start) & (df['datetime'] < train_end)
test_mask  = (df['datetime'] >= train_end)   & (df['datetime'] < test_end)

scaler = RobustScaler()
scaler.fit(df.loc[train_mask, selected_feats])
df[selected_feats] = scaler.transform(df[selected_feats])

# ========== 4️⃣ HMM 模型定义与训练 ==========
X_train = df.loc[train_mask, selected_feats].values.astype(np.float64)
assert np.isfinite(X_train).all(), "NaN 或 Inf 存在，请检查标准化阶段"

K = 3
n_states = K
n_features = X_train.shape[1]

distributions = [
    Normal(
        means=np.random.normal(0, 0.5, n_features),
        covs=np.ones(n_features),
        covariance_type='diag'
    )
    for _ in range(n_states)
]

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
print("✅ 训练完成！")  # ✅ 不再格式化 logprob，避免错误

# ========== 5️⃣ 保存模型 ==========
with open("trained_hmm_simple.pkl", "wb") as f:
    pickle.dump(model, f)
print("💾 模型已保存为 trained_hmm_simple.pkl")

# ========== 6️⃣ 预测状态 + 拒识处理 ==========
def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return np.asarray(x)

log_gamma = model.predict_proba([X_train])[0]
gamma = np.exp(to_numpy(log_gamma))
gamma = gamma / gamma.sum(axis=1, keepdims=True)
max_p = gamma.max(axis=1)

tau = np.quantile(max_p, 0.3)
print(f"📉 置信度阈值 tau = {tau:.4f}")

state_hard = gamma.argmax(axis=1)
state_with_unknown = np.where(max_p >= tau, state_hard, -1)
df.loc[train_mask, 'state'] = state_with_unknown

# ========== 7️⃣ 绘图 ==========
import colorsys
import matplotlib.colors as mcolors

# ✅ 动态颜色分配
def generate_colors(num_colors):
    """生成区分度高的颜色列表"""
    hues = np.linspace(0, 1, num_colors, endpoint=False)
    colors = [mcolors.rgb2hex(colorsys.hsv_to_rgb(h, 0.65, 0.9)) for h in hues]
    return colors

# 为状态生成颜色
base_colors = generate_colors(K)
color_map = {i: base_colors[i] for i in range(K)}
color_map[-1] = "silver"  # 拒识状态固定灰色

# 自动生成状态名称
state_labels = {i: f"State {i}" for i in range(K)}
state_labels[-1] = "Uncertain (Gray)"

# 绘图数据
df_plot = df.loc[train_mask, ['datetime', 'c', 'state']].copy()
df_plot['state_name'] = df_plot['state'].map(state_labels)

fig = go.Figure()

# 收盘价主线
fig.add_trace(go.Scatter(
    x=df_plot['datetime'],
    y=df_plot['c'],
    mode='lines',
    line=dict(color='black', width=1),
    name='Close Price',
    hovertemplate='时间: %{x}<br>收盘价: %{y:.2f}<extra></extra>'
))

# 状态散点
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
        hovertemplate=f'时间: %{{x}}<br>收盘价: %{{y:.2f}}<br>状态: {state_labels[s]}<extra></extra>'
    ))

fig.update_layout(
    title=dict(
        text=f"ETH 市场状态识别（K={K}，含趋势特征 + 拒识状态）",
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
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import os

# ====== 1 读取数据 ====== 注意以 8:00:00为例 o为当前价格 c为12:00的价格 hl为8-12点的最高最低价
# ---- 宏观经济类 ----
macro_paths = {
xxxxxxxxxxxxxxxxxxxxx
}

# ---- ETH 市场类（共 11 个） ----
eth_paths = {
xxxxxxxxxxxxxxxxx
}


def read_full_csv(path):
    """读取单个CSV：
    - 自动识别时间列（包含 'time' 或 'date'）
    - 去掉时区，统一为 tz-naive
    - 仅保留数值列 + datetime
    - 不做任何聚合/平均，保留原始字段（如 c/h/l/o/vol/buy/sell/...）
    """
    if not os.path.exists(path):
        print(f"⚠️ 文件不存在：{path}")
        return None

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # 时间列
    time_col = next((c for c in df.columns if ('time' in c) or ('date' in c)), None)
    if time_col is None:
        raise ValueError(f"{path} 缺少时间列")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[time_col] = df[time_col].dt.tz_localize(None)  # 统一去时区
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    df = df.rename(columns={time_col: "datetime"})

    # 只保留数值列
    numeric_cols = [c for c in df.columns if c != "datetime" and np.issubdtype(df[c].dtype, np.number)]
    if not numeric_cols:
        raise ValueError(f"{path} 无数值列可用")
    df = df[["datetime"] + numeric_cols]
    return df


def load_all_data(macro_paths, eth_paths):
    """批量读取，给每个数值列加上文件名前缀，避免重名冲突"""
    all_data: dict[str, pd.DataFrame] = {}
    for name, path in {**macro_paths, **eth_paths}.items():
        try:
            df = read_full_csv(path)
            if df is None:
                continue
            # 为防止列名冲突，加前缀（datetime 不加）
            rename_map = {c: f"{name}_{c}" for c in df.columns if c != "datetime"}
            df = df.rename(columns=rename_map)
            all_data[name] = df
        except Exception as e:
            print(f"❌ 读取失败 {name}: {e}")
    return all_data

# ==== 执行读取 ====
all_data = load_all_data(macro_paths, eth_paths)

# ====== 2️⃣ 精确时间对齐合并 并划分训练集测试集 ======
# 确认所有表都有相同的 datetime 频率
for name, df in all_data.items():
    print(f"{name:20s} -> {df['datetime'].min()} ~ {df['datetime'].max()}, {len(df)} rows")

# 以基准表（ETH 1h）为主，做“精确 inner join”
from functools import reduce

dfs = list(all_data.values())
merged = reduce(lambda left, right: pd.merge(left, right, on="datetime", how="inner"), dfs)

merged = merged.sort_values("datetime").reset_index(drop=True)
print("✅ 合并完成：", merged.shape)
print("时间范围：", merged['datetime'].min(), "→", merged['datetime'].max())

# ====== 划分训练 / 测试集（时序切分） ======
train_start = pd.Timestamp("2024-01-06")
train_end   = pd.Timestamp("2025-10-01")
test_end    = pd.Timestamp("2025-11-10")

train_df = merged[(merged["datetime"] >= train_start) & (merged["datetime"] < train_end)].copy()
test_df  = merged[(merged["datetime"] >= train_end) & (merged["datetime"] < test_end)].copy()

print("Train:", train_df["datetime"].min(), "→", train_df["datetime"].max(), train_df.shape)
print("Test :", test_df["datetime"].min(),  "→", test_df["datetime"].max(),  test_df.shape)
# === 输出列名 ===
print("\n📊 表头字段（共 %d 列）：" % len(merged.columns))
print(merged.columns.tolist())

# ====== 3️⃣ 共享基础特征（所有任务通用） ======
# === 基础价格字段 ===
# =========================================
eps = 1e-9

# === 基础价格字段 ===
merged['open']  = merged['eth_hist_o']
merged['high']  = merged['eth_hist_h']
merged['low']   = merged['eth_hist_l']
merged['close'] = merged['eth_hist_c']

# === 均线与趋势相关特征 ===
merged['ma7']   = merged['close'].rolling(7).mean()
merged['ma30']  = merged['close'].rolling(30).mean()
merged['ma_diff_short'] = merged['ma7'] - merged['ma30']

# === MACD（趋势/动量）===
_ema12 = merged['close'].ewm(span=12, adjust=False).mean()   # 12×4h≈48h
_ema26 = merged['close'].ewm(span=26, adjust=False).mean()   # 26×4h≈4天半
merged['macd_line']   = _ema12 - _ema26
merged['macd_signal'] = merged['macd_line'].ewm(span=9, adjust=False).mean()
merged['macd_hist']   = merged['macd_line'] - merged['macd_signal']
merged['macd_slope']  = merged['macd_hist'].diff()

# === Bollinger 布林带宽度（波动形态）===
n_bool = 20
merged['boll_m'] = merged['close'].rolling(window=n_bool).mean()
std = merged['close'].rolling(window=n_bool).std()
merged['boll_h'] = merged['boll_m'] + 2 * std
merged['boll_l'] = merged['boll_m'] - 2 * std
merged['boll_width'] = merged['boll_h'] - merged['boll_l']

# === RSI6 + 随机RSI（震荡 / 超买超卖）===
rsi_windows = 6
delta = merged['close'].diff()
gain_rsi = delta.where(delta > 0, 0)
loss_rsi = -delta.where(delta < 0, 0)
avg_gain = gain_rsi.ewm(alpha=1/rsi_windows, min_periods=rsi_windows).mean()
avg_loss = loss_rsi.ewm(alpha=1/rsi_windows, min_periods=rsi_windows).mean()
rs = avg_gain / (avg_loss + eps)
merged['rsi6'] = 100 - (100 / (1 + rs))
merged['stoch_rsi_raw']  = (merged['rsi6'] - merged['rsi6'].rolling(window=12).min()) \
                       / (merged['rsi6'].rolling(window=12).max() - merged['rsi6'].rolling(window=12).min()) * 100
merged['stoch_rsi']  = merged['stoch_rsi_raw'].rolling(window=6).mean()

# 波动率代理
merged['vol_ratio'] = merged['VIX_c'] / (merged['OVX_c'] + + 1e-8)

# === VIX（恐慌指数跳变与标准化）===
_wz = 15
merged['vix_z'] = (merged['VIX_c'] - merged['VIX_c'].rolling(_wz).mean()) / (merged['VIX_c'].rolling(_wz).std() + eps)
merged['vix_jump10'] = (merged['VIX_c'] / (merged['VIX_c'].ewm(span=3, adjust=False).mean() + eps)) - 1.0

# === OI / Funding 拥挤度 ===
merged['funding_rate'] = merged['eth_oi_weight_o']
merged['oi_change'] = merged['eth_oi_c'].pct_change()
merged['funding_z'] = (merged['funding_rate'] - merged['funding_rate'].rolling(_wz).mean()) / (merged['funding_rate'].rolling(_wz).std() + eps)
merged['oi_z'] = (merged['eth_oi_c'] - merged['eth_oi_c'].rolling(_wz).mean()) / (merged['eth_oi_c'].rolling(_wz).std() + eps)
merged['crowding'] = merged['funding_z'] * merged['oi_z']

# === SPX 共振（跨市场相关 / beta）===
spx_ret = np.log(merged['SPX_c']).diff()
eth_ret = np.log(merged['close']).diff()
_wb = 18
merged['corr_spx_eth'] = eth_ret.rolling(_wb).corr(spx_ret)
merged['beta_spx'] = eth_ret.rolling(_wb).cov(spx_ret) / (spx_ret.rolling(_wb).var() + eps)

# === 波动率滚动特征（VIX / OVX）===
merged['VIX_c_rolling_std_3'] = merged['VIX_c'].rolling(3).std()
merged['OVX_c_rolling_std_3'] = merged['OVX_c'].rolling(3).std()
merged['VIX_c_rolling_std_5'] = merged['VIX_c'].rolling(5).std()

#
N = 20  # 例如过去20小时
merged['ex_oi'] = (merged['eth_oi_c'] - merged['eth_oi_c'].rolling(N).mean()) / \
                  (merged['eth_oi_c'].rolling(N).mean() + 1e-9)

#
merged['VIX_c_rolling_std_10'] = merged['VIX_c'].rolling(window=10).std()
merged['VIX_c_rolling_std_20'] = merged['VIX_c'].rolling(window=20).std()
merged['VIX_c_rolling_mean_10'] = merged['VIX_c'].rolling(window=10).mean()

merged['ret_4h']   = np.log(merged['close']).diff()
merged['dvol_level'] = merged['eth_dvol_c']
merged['rv_24h']  = merged['ret_4h'].rolling(6).std()        # 过去24h波动
merged['dvol_z'] = (merged['dvol_level'] - merged['dvol_level'].rolling(15).mean()) / (merged['dvol_level'].rolling(15).std() + eps)
_rv_z = (merged['rv_24h'] - merged['rv_24h'].rolling(15).mean()) / (merged['rv_24h'].rolling(15).std() + eps)
merged['iv_rv_gap'] = merged['dvol_z'] - _rv_z

# === 特征选择 ===
# 手工特征
features = [
    # 趋势/动量
    'macd_hist', 'macd_slope', 'ma_diff_short',
    # 波动/形态
    'boll_width', 'vol_ratio', 'vix_jump10', 'vix_z',
    # 资金位/仓位
    'oi_change', 'funding_rate', 'crowding',
    # 跨市场关系/宏观 beta
    'beta_spx', 'corr_spx_eth',
    # 震荡/超买超卖
    'stoch_rsi',
    # 特征提取特征
    'VIX_c_rolling_std_3', 'OVX_c_rolling_std_3', 'VIX_c_rolling_std_5',
    # 额外
    # 'VIX_c_rolling_std_10', 'VIX_c_rolling_std_20', 'VIX_c_rolling_mean_10', 'iv_rv_gap', 'ex_oi'
]
print(f"✅ 最终特征总数: {len(features)}")


# ========= 预测目标 ============

merged['open']  = merged['eth_hist_o']       # 开盘价（Open）
merged['high']  = merged['eth_hist_h']       # 最高价（High）
merged['low']   = merged['eth_hist_l']       # 最低价（Low）
merged['close'] = merged['eth_hist_c']       # 收盘价（Close）
eps = 1e-9
# 目标 4
merged['eth_samrt'] = (merged['high'] + merged['low'] + 2 * merged['close']) / 4.0
merged['y_ret_samrt'] = np.log(merged['eth_samrt'].shift(-1) / merged['eth_samrt'])
merged['y_target'] = merged['y_ret_samrt']

merged['y_target'].hist(bins=100)
print(merged['y_target'].describe())
merged = merged.dropna(subset=['y_target']).reset_index(drop=True)
print(" Target已经生成 ")
print(merged[['datetime', 'eth_hist_c', 'y_target']].head())

def hp_filter_kalman(ts: pd.Series, lamb: float = 1600) -> pd.Series:
    """
    最稳定的单边 HP（State-space + Kalman Filter 实现）
    - 不使用未来数据
    - 无断崖
    - 金融时间序列最推荐方式
    """

    y = ts.astype(float).copy().values
    n = len(y)

    # === 状态空间模型 (local linear trend model) ===
    # x_t = [tau_t, tau'_t]
    # tau'_t 是趋势的一阶导数（斜率）

    # 状态转移矩阵
    F = np.array([[1, 1],
                  [0, 1]])

    # 观测矩阵
    H = np.array([[1, 0]])

    # 噪声协方差
    # HP λ 等价于 R / Q
    R = 1.0                   # 观测噪声
    Q = 1.0 / lamb            # 趋势噪声（越小越平滑）

    # 协方差
    Qm = np.array([[Q, 0],
                   [0, Q]])

    P = np.eye(2) * 1e6       # 初始协方差
    x = np.array([y[0], 0.0]) # 初始趋势 + 斜率

    trend = np.zeros(n)

    # === 单边 Kalman Filter ===
    for t in range(n):
        # 预测
        x_pred = F @ x
        P_pred = F @ P @ F.T + Qm

        # 更新
        yt = np.array([y[t]])
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ (yt - H @ x_pred)
        P = (np.eye(2) - K @ H) @ P_pred

        trend[t] = x[0]

    return pd.Series(trend, index=ts.index)


# 平滑
merged['NASX_smooth'] = hp_filter_kalman(merged['NASX_c'], lamb=10000)
merged['VIX_smooth'] = hp_filter_kalman(merged['VIX_c'], lamb=10000)
merged['DXY_smooth'] = hp_filter_kalman(merged['DXY_c'], lamb=10000)
merged['eth_c_smooth'] = hp_filter_kalman(merged['eth_hist_c'], lamb=10000)
merged['xauusd_smooth'] = hp_filter_kalman(merged['XAUUSD_c'], lamb=10000000000000000000000000)
merged['xauusd_det'] = merged['XAUUSD_c'] - merged['xauusd_smooth']


# ======================================================
# 🔁 数据驱动版 nSPC 宏观状态（1 维）
#    用 NASX_smooth / VIX_smooth / DXY_smooth / xauusd_det
# ======================================================

from sklearn.linear_model import Ridge

def rolling_zscore(s: pd.Series, window: int):
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - m) / (sd + 1e-9)

# 1) 构造平滑宏观的 z-score + slope
merged['NASX_slope'] = np.log(merged['NASX_smooth'] + 1e-9).diff()
merged['VIX_slope']  = np.log(merged['VIX_smooth']  + 1e-9).diff()
merged['DXY_slope']  = np.log(merged['DXY_smooth']  + 1e-9).diff()
merged['xauusd_det_slope'] = merged['xauusd_det'].diff()

win_z = 200

merged['NASX_smooth_z']      = rolling_zscore(merged['NASX_smooth'],      win_z)
merged['VIX_smooth_z']       = rolling_zscore(merged['VIX_smooth'],       win_z)
merged['DXY_smooth_z']       = rolling_zscore(merged['DXY_smooth'],       win_z)
merged['xauusd_det_z']       = rolling_zscore(merged['xauusd_det'],       win_z)

merged['NASX_slope_z']       = rolling_zscore(merged['NASX_slope'],       win_z)
merged['VIX_slope_z']        = rolling_zscore(merged['VIX_slope'],        win_z)
merged['DXY_slope_z']        = rolling_zscore(merged['DXY_slope'],        win_z)
merged['xauusd_det_slope_z'] = rolling_zscore(merged['xauusd_det_slope'], win_z)

# 2) 选一个干净的小宏观特征集合
macro_cols = [
    'NASX_smooth_z', 'VIX_smooth_z', 'DXY_smooth_z', 'xauusd_det_z',
    'NASX_slope_z',  'VIX_slope_z',  'DXY_slope_z',  'xauusd_det_slope_z',
]

# 对齐目标
macro_df = merged[['datetime', 'y_target'] + macro_cols].dropna().reset_index(drop=True)

# train / test 划分
train_mask_m = (macro_df['datetime'] >= train_start) & (macro_df['datetime'] < train_end)
test_mask_m  = (macro_df['datetime'] >= train_end)   & (macro_df['datetime'] < test_end)

train_m = macro_df.loc[train_mask_m].copy()
test_m  = macro_df.loc[test_mask_m].copy()

X_train_m = train_m[macro_cols].values
y_train_m = train_m['y_target'].values

X_test_m  = test_m[macro_cols].values
y_test_m  = test_m['y_target'].values

print("宏观线性模型：", X_train_m.shape, X_test_m.shape)

# 3) 用简单的 Ridge 学一条“宏观多空线”
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_m, y_train_m)

print("Ridge coef:", dict(zip(macro_cols, ridge.coef_)))
print("train R2:", ridge.score(X_train_m, y_train_m))
print("test  R2:", ridge.score(X_test_m,  y_test_m))

# 4) 定义 macro_signal_t = w^T x_t （全样本都算一遍）
macro_df['macro_signal'] = macro_df[macro_cols].values @ ridge.coef_.reshape(-1, 1)
# 合并回 merged（按时间对齐）
merged = pd.merge(
    merged,
    macro_df[['datetime', 'macro_signal']],
    on='datetime',
    how='left'
)


def roll_state_constant_alpha(signal_series: pd.Series, alpha: float = 0.1):
    """
    最简单 nSPC 版本：常数 alpha
    state_t = (1 - alpha) * state_{t-1} + alpha * signal_t
    """
    vals = signal_series.fillna(0.0).values
    state_vals = np.zeros(len(vals), dtype=float)
    prev = 0.0
    for i, sig in enumerate(vals):
        prev = (1.0 - alpha) * prev + alpha * sig
        state_vals[i] = prev
    return pd.Series(state_vals, index=signal_series.index)


merged['macro_state'] = roll_state_constant_alpha(merged['macro_signal'], alpha=0.1)
print(merged[['datetime', 'macro_signal', 'macro_state']].tail())

tmp = merged[['macro_state', 'y_target']].dropna()
print(tmp.corr())

tmp['bucket'] = pd.qcut(tmp['macro_state'], 5, labels=False)
print(tmp.groupby('bucket')['y_target'].mean())


import plotly.express as px
import plotly.graph_objects as go

def plot_macro_state_effect(merged):
    """
    交互式检查 macro_state 对 y_target 的影响：
    1) 散点：macro_state vs y_target（颜色区分分位桶）
    2) 柱状：macro_state 分桶下的 y_target 均值
    """
    # 1. 准备数据
    tmp = merged[['datetime', 'macro_state', 'y_target']].dropna().copy()
    if tmp.empty:
        print("⚠️ 没有可用数据（macro_state 或 y_target 全是 NaN）")
        return

    # 分 5 桶（你可以改成 3、4、10）
    tmp['bucket_id'] = pd.qcut(tmp['macro_state'], 5, labels=False)

    # 给桶起个更直观的名字
    label_map = {
        0: 'Q1 最低 macro_state',
        1: 'Q2',
        2: 'Q3 中间',
        3: 'Q4',
        4: 'Q5 最高 macro_state'
    }
    tmp['bucket'] = tmp['bucket_id'].map(label_map)

    # ========== 图 1：散点图（macro_state vs y_target）==========
    fig_scatter = px.scatter(
        tmp,
        x='macro_state',
        y='y_target',
        color='bucket',
        title='macro_state vs y_target（按分位桶着色）',
        labels={
            'macro_state': 'macro_state',
            'y_target': '未来 4h 收益 y_target'
        },
        opacity=0.6
    )
    # 加一点 0 线辅助判断
    fig_scatter.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.5)

    fig_scatter.update_layout(
        hovermode='closest'
    )

    fig_scatter.show()

    # ========== 图 2：柱状图（不同桶的平均 y_target）==========
    bucket_stats = (
        tmp
        .groupby('bucket', observed=True)['y_target']
        .agg(['mean', 'count'])
        .reset_index()
        .sort_values('bucket')
    )

    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=bucket_stats['bucket'],
        y=bucket_stats['mean'],
        text=bucket_stats['count'],
        textposition='outside',
        name='各桶 y_target 均值',
    ))

    fig_bar.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.5)

    fig_bar.update_layout(
        title="macro_state 分桶下 y_target 均值（数字为样本数）",
        xaxis_title="macro_state 分位桶",
        yaxis_title="未来 4h 收益 y_target 的均值",
        hovermode='x'
    )

    fig_bar.show()

plot_macro_state_effect(merged)

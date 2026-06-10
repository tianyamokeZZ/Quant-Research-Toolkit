import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import os

# ====== 1 读取数据 ====== 注意以 8:00:00为例 o为当前价格 c为12:00的价格 hl为8-12点的最高最低价
# ---- 宏观经济类 ----
macro_paths = {
    "VIX": "VIX_4h.csv",              # 芝加哥波动率指数（市场恐慌指标）
    "SPX": "SPX_4h.csv",              # 标普500指数（美股市场走势）
    "NASX": "NASX_4h.csv",            # 纳斯达克综合指数
    "GBTC": "GBTC_4h.csv",            # 灰度比特币信托基金（比特币溢价指标）
    "DXY": "DXY_4h.csv",              # 美元指数（宏观流动性指标）
    "OVX": "OVX_4h.csv",              # 原油波动率指数（能源市场风险指标）
    "XAUUSD": "XAUUSD_4h.csv",        # 现货黄金价格（避险资产代表）
    # new feature
    "TLT": "TLT_4h.csv",
    "SHY": "SHY_4h.csv"
}

# ---- ETH 市场类（共 11 个） ----
eth_paths = {
    "eth_hist":              "eth_1h_hist_4h.csv",           # ETH 历史K线（开高低收成交量）
    "eth_vol_weight":        "eth_1h_volweightohlc_4h.csv",         # 按成交量加权的资金费率
    "eth_oi_weight":         "eth_1h_oiweightohlc_4h.csv",          # 按持仓量加权的资金费率
    "eth_dvol":              "eth_1h_Dvol_4h.csv",           # Deribit 隐含波动率（聚合为4h）
    "eth_liq":               "eth_1h_liq_4h.csv",    # （可选）多空爆仓金额（4h）
    "eth_oi":                "eth_1h_marketoi_4h.csv",       # 市场持仓量（Open Interest）
    "eth_buy_sell":          "eth_1h_buy_sell_4h.csv",       # 主动买入/卖出成交量
    "eth_coinmarketcap":     "eth_1h_coinmarketcap_4h.csv",  # 市值与流通量等指标
    "eth_trans_btc":         "eth_1h_eth_trans_btc_4h.csv",  # ETH/BTC 链上转账比率
    "eth_stable_margin":     "eth_1h_stablecoin-margin_4h.csv", # 稳定币保证金占比
    "eth_timeseries":        "eth_1h_timeseries_4h.csv",     # ETH 价格及波动时间序列（辅助特征）
    # "eth_4h_VWAP":           "eth_4h_vwap.csv",              # 4小时加权平均成交价（VWAP 目标）
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
# 4️⃣ 基于平滑宏观序列的 nSPC 状态因子
#    使用: NASX_smooth, VIX_smooth, DXY_smooth, xauusd_det
# ======================================================


def rolling_zscore(s: pd.Series, window: int):
    """滚动 z-score，用于把水平 / 斜率统一到可比尺度"""
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - m) / (sd + 1e-9)


def compute_alpha_from_signal(sig: float, base: float = 0.05, max_alpha: float = 0.3):
    """
    根据 signal 强度自适应记忆长度：
    - |sig| 越大，alpha 越大 -> 状态更新更快（看短一点）
    - |sig| 越小，alpha 越小 -> 状态更新更慢（看长一点）
    """
    strength = abs(sig)
    alpha = base + (max_alpha - base) * strength
    return alpha


def roll_state(signal_series: pd.Series, base: float = 0.05, max_alpha: float = 0.3):
    """
    nSPC 风格状态递推：
        state_t = (1 - alpha_t) * state_{t-1} + alpha_t * signal_t
    """
    vals = signal_series.fillna(0.0).values
    state_vals = np.zeros(len(vals), dtype=float)
    prev = 0.0
    for i, sig in enumerate(vals):
        alpha_t = compute_alpha_from_signal(sig, base, max_alpha)
        prev = (1.0 - alpha_t) * prev + alpha_t * sig
        state_vals[i] = prev
    return pd.Series(state_vals, index=signal_series.index)


# ---------- 1) 构造平滑后的“斜率 / 变化率” ----------
# 用平滑后的价格做 log-diff，避免毛刺
merged['NASX_slope'] = np.log(merged['NASX_smooth'] + 1e-9).diff()
merged['VIX_slope']  = np.log(merged['VIX_smooth']  + 1e-9).diff()
merged['DXY_slope']  = np.log(merged['DXY_smooth']  + 1e-9).diff()

# 去趋势黄金残差的斜率（看资金是否从黄金流出/流入）
merged['xauusd_det_slope'] = merged['xauusd_det'].diff()

# ---------- 2) 对平滑“水平”和“斜率”做 z-score ----------
win_z = 200  # 看长一点的背景

merged['NASX_smooth_z']        = rolling_zscore(merged['NASX_smooth'],        win_z).clip(-3, 3)
merged['VIX_smooth_z']         = rolling_zscore(merged['VIX_smooth'],         win_z).clip(-3, 3)
merged['DXY_smooth_z']         = rolling_zscore(merged['DXY_smooth'],         win_z).clip(-3, 3)
merged['xauusd_det_z']         = rolling_zscore(merged['xauusd_det'],         win_z).clip(-3, 3)

merged['NASX_slope_z']         = rolling_zscore(merged['NASX_slope'],         win_z).clip(-3, 3)
merged['VIX_slope_z']          = rolling_zscore(merged['VIX_slope'],          win_z).clip(-3, 3)
merged['DXY_slope_z']          = rolling_zscore(merged['DXY_slope'],          win_z).clip(-3, 3)
merged['xauusd_det_slope_z']   = rolling_zscore(merged['xauusd_det_slope'],   win_z).clip(-3, 3)

# ---------- 3) 即时 signal：panic / usx / riskon / vix_reb ----------

# ① Panic（恐慌端）：
#    DXY_slope_z ↑ + VIX_smooth_z 高 + VIX_slope_z ↑ + NASX_slope_z ↓
w_dxy  =  0.8
w_vixl =  0.7
w_vixc =  0.4
w_nasx = -0.6

merged['panic_signal_raw'] = (
    w_dxy  * merged['DXY_slope_z'] +
    w_vixl * merged['VIX_smooth_z'] +
    w_vixc * merged['VIX_slope_z'] +
    w_nasx * merged['NASX_slope_z']
)
merged['panic_signal'] = np.tanh(merged['panic_signal_raw'])

# ② US Exceptionalism（美国例外主义）：
#    DXY_slope_z 温和 > 0 + VIX_smooth_z 低 + NASX_slope_z > 0
dxy_pos = merged['DXY_slope_z'].clip(0, 2)   # 只取“温和向上”

merged['usx_signal_raw'] = (
    0.7 * dxy_pos +
    -0.7 * merged['VIX_smooth_z'] +
    0.8 * merged['NASX_slope_z']
)
merged['usx_signal'] = np.tanh(merged['usx_signal_raw'])

# ③ Global Risk-on（全球风险偏好）：
#    DXY_slope_z < 0（缓慢走弱） + VIX_smooth_z 低 + 黄金残差为负（资金离开黄金）
dxy_neg = (-merged['DXY_slope_z']).clip(0, 2)  # 只取“往下”的部分
gold_off = (-merged['xauusd_det_z']).clip(0, 2)  # 黄金低于趋势 -> risk-on

merged['riskon_signal_raw'] = (
    0.8 * dxy_neg +
    -0.7 * merged['VIX_smooth_z'] +
    0.4 * gold_off
)
merged['riskon_signal'] = np.tanh(merged['riskon_signal_raw'])

# ④ VIX 回落修复：
#    VIX_smooth_z 之前很高 + 现在 VIX_slope_z < 0（往下走）
vix_drop = (-merged['VIX_slope_z']).clip(0, 3)  # 下行斜率（回落）
merged['vix_reb_signal_raw'] = vix_drop + merged['VIX_smooth_z'].clip(0, 3)
merged['vix_reb_signal'] = np.tanh(merged['vix_reb_signal_raw'])

# ---------- 4) 用 nSPC 递推出 4 个状态 ----------
merged['panic_state']   = roll_state(merged['panic_signal'],   base=0.05, max_alpha=0.3)
merged['usx_state']     = roll_state(merged['usx_signal'],     base=0.05, max_alpha=0.3)
merged['riskon_state']  = roll_state(merged['riskon_signal'],  base=0.05, max_alpha=0.3)
merged['vix_reb_state'] = roll_state(merged['vix_reb_signal'], base=0.05, max_alpha=0.3)

print(merged[['datetime', 'panic_state', 'usx_state', 'riskon_state', 'vix_reb_state']].tail())



import plotly.graph_objects as go

def plot_macro_states_interactive(merged, start=None, end=None):
    """
    交互式对比：
    - ETH 平滑价格 z-score: eth_c_smooth_z
    - 4 个 nSPC 状态: panic_state / usx_state / riskon_state / vix_reb_state

    参数:
        merged : 含 datetime, eth_c_smooth, 4 个 state 的 DataFrame
        start  : 可选，起始时间（字符串或 Timestamp）
        end    : 可选，结束时间（字符串或 Timestamp）
    """
    df = merged.copy()

    # 1) ETH 平滑价格做 z-score
    eps = 1e-9
    mean_eth = df['eth_c_smooth'].mean()
    std_eth  = df['eth_c_smooth'].std() + eps
    df['eth_c_smooth_z'] = (df['eth_c_smooth'] - mean_eth) / std_eth

    # 2) 按时间截取
    if start is not None:
        start = pd.to_datetime(start)
        df = df[df['datetime'] >= start]
    if end is not None:
        end = pd.to_datetime(end)
        df = df[df['datetime'] <= end]

    if df.empty:
        print("⚠️ 选定时间区间内没有数据")
        return

    # 3) 画交互式图
    fig = go.Figure()

    # ETH 平滑走势（z-score）
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['eth_c_smooth_z'],
        mode='lines',
        name='ETH smooth z',
        line=dict(width=2)
    ))

    # 4 个状态
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['panic_state'],
        mode='lines',
        name='panic_state',
        opacity=0.8
    ))
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['usx_state'],
        mode='lines',
        name='usx_state',
        opacity=0.8
    ))
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['riskon_state'],
        mode='lines',
        name='riskon_state',
        opacity=0.8
    ))
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['vix_reb_state'],
        mode='lines',
        name='vix_reb_state',
        opacity=0.8
    ))

    # 0 轴
    fig.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.5)

    fig.update_layout(
        title="ETH 平滑走势（z-score）与宏观 nSPC 状态（交互）",
        xaxis_title="时间",
        yaxis_title="标准化数值 / 状态值",
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7,  label="7d",  step="day",  stepmode="backward"),
                    dict(count=30, label="30d", step="day",  stepmode="backward"),
                    dict(count=90, label="90d", step="day",  stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    fig.show()

plot_macro_states_interactive(merged)
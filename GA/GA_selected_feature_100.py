import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import os

# ====== 1 读取数据 ====== 注意以 8:00:00为例 o为当前价格 c为12:00的价格 hl为8-12点的最高最低价
# ---- 宏观经济类 ----
macro_paths = {
xxxxxxxxxxxxxxx
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
train_start = pd.Timestamp("2024-01-03")
train_end   = pd.Timestamp("2025-02-01")
test_start  = train_end
test_start  = pd.Timestamp("2025-02-01")
test_end    = pd.Timestamp("2025-04-01")

train_df = merged[(merged["datetime"] >= train_start) & (merged["datetime"] < train_end)].copy()
test_df  = merged[(merged["datetime"] >= test_start) & (merged["datetime"] < test_end)].copy()

print("Train:", train_df["datetime"].min(), "→", train_df["datetime"].max(), train_df.shape)
print("Test :", test_df["datetime"].min(),  "→", test_df["datetime"].max(),  test_df.shape)
# === 输出列名 ===
print("\n📊 表头字段（共 %d 列）：" % len(merged.columns))
print(merged.columns.tolist())


# 特征平滑函数
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
# 平滑后
# merged['VIX_smooth'] = hp_filter_kalman(merged['VIX_c'], lamb=10000)
# merged['OVX_smooth'] = hp_filter_kalman(merged['VIX_c'], lamb=10000)
# merged['VIX_c_rolling_std_3'] = merged['VIX_smooth'].rolling(3).std()
# merged['OVX_c_rolling_std_3'] = merged['OVX_smooth'].rolling(3).std()
# merged['VIX_c_rolling_std_5'] = merged['VIX_smooth'].rolling(5).std()


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

# 搞宏观模型
merged['NASX_smooth'] = hp_filter_kalman(merged['NASX_c'], lamb=10000)
merged['VIX_smooth'] = hp_filter_kalman(merged['VIX_c'], lamb=10000)
merged['DXY_smooth'] = hp_filter_kalman(merged['DXY_c'], lamb=10000)
# merged['eth_c_smooth'] = hp_filter_kalman(merged['eth_hist_c'], lamb=10000000)
merged['xauusd_smooth'] = hp_filter_kalman(merged['XAUUSD_c'], lamb=10000000000000000000000000)
merged['xauusd_det'] = merged['XAUUSD_c'] - merged['xauusd_smooth']
merged['DXY_smooth_bigbig'] = hp_filter_kalman(merged['DXY_c'], lamb=10000000000000000000000000)
# merged['DXY_det'] = merged['DXY_smooth'] - merged['DXY_smooth_bigbig']
merged['DXY_det'] = merged['DXY_c'] - merged['DXY_smooth_bigbig']


# 扩充1：1）结构型趋势因子（slope acc swing） 2）多字段通用slope（VIX OI Funding Beta 相关性）

# ===== 通用线性回归 slope =====
def rolling_slope(series: pd.Series, window: int):
    """
    对任意时间序列计算滚动线性回归 slope (β)
    y = α + β t
    返回与 series 等长的数组，前 window 个为 nan
    """
    y = series.values.astype(float)
    x = np.arange(window)  # 0,1,...,window-1
    slopes = np.full(len(y), np.nan)

    for i in range(window, len(y)):
        y_win = y[i-window:i]
        beta = np.polyfit(x, y_win, 1)[0]
        slopes[i] = beta

    return pd.Series(slopes, index=series.index)


# ===== 二阶导数 Acceleration =====
def acceleration(series: pd.Series):
    """
    Acc = P_t - 2*P_{t-1} + P_{t-2}
    """
    return series - 2*series.shift(1) + series.shift(2)


# ===== 价格趋势增强 =====
# === A1. 多窗口 Price Slope ===
merged["price_slope_10"] = rolling_slope(merged["close"], 10)
merged["price_slope_20"] = rolling_slope(merged["close"], 20)
merged["price_slope_50"] = rolling_slope(merged["close"], 50)

# === A2. Price Acceleration (趋势二阶导) ===
merged["price_acc"] = acceleration(merged["close"])

# === A3. 局部结构（swing high/low）===
# window=5 可调，越大越“确认”，越小越灵敏
win = 5  # 可以调大/调小

# 过去 win 根K线的最高 / 最低
roll_high = merged["high"].rolling(win, min_periods=win).max()
roll_low  = merged["low"].rolling(win, min_periods=win).min()

merged["swing_high"] = (merged["high"] >= roll_high).astype(int)
merged["swing_low"]  = (merged["low"]  <= roll_low).astype(int)

# === A4. swing 动量（高点相对于最近一次确认的低点）===

merged["last_swing_low"] = merged["low"].where(merged["swing_low"] == 1).ffill()
merged["swing_momentum"] = merged["high"] - merged["last_swing_low"]

merged.drop(columns=["last_swing_low"], inplace=True)


# ===== 对关键因子做 Slope =====

# 对 VIX 方向趋势
merged["vix_slope"] = rolling_slope(merged["VIX_c"], 10)

# 对 OI 方向趋势
merged["oi_slope"] = rolling_slope(merged["eth_oi_c"], 10)

# Funding rate slope
merged["funding_slope"] = rolling_slope(merged["funding_rate"], 10)

# 跨市场 Beta slope
merged["beta_slope"] = rolling_slope(merged["beta_spx"], 20)

# 跨市场 Corr slope
merged["corr_slope"] = rolling_slope(merged["corr_spx_eth"], 20)

# IV slope
merged["iv_slope"] = rolling_slope(merged["dvol_level"], 20)

# RV slope
merged["rv_slope"] = rolling_slope(merged["rv_24h"], 20)

# 扩充2：波动率结构因子


# ===== 1) Realized Volatility 扩展 =====

# RV 的斜率（趋势方向）
merged["rv_24h_slope"] = rolling_slope(merged["rv_24h"], 20)

# 3 日实现波动率
merged["rv_3d"] = merged["ret_4h"].rolling(18).std()

# RV3d 的 Z-score（波动 regime）
merged["rv_3d_z"] = (merged["rv_3d"] - merged["rv_3d"].rolling(60).mean()) / \
                    (merged["rv_3d"].rolling(60).std() + 1e-9)

# RV 的加速度（波动突然爆发）
merged["rv_acc"] = acceleration(merged["rv_24h"])


# ===== 2) OHLC 波动结构 =====

# Parkinson 波动率（高低价）
merged["vol_parkinson"] = (np.log(merged["high"] / (merged["low"] + 1e-9)) ** 2) / (4 * np.log(2))

# Price range (H-L)/O
merged["vol_range"] = (merged["high"] - merged["low"]) / (merged["open"] + 1e-9)

# K线实体长度 / 波动（趋势强度）
merged["vol_body"] = (merged["close"] - merged["open"]) / (merged["high"] - merged["low"] + 1e-9)


# ===== 3) 隐含波动率结构（IV） =====

# IV slope（隐含波动率趋势）
merged["iv_slope2"] = rolling_slope(merged["dvol_level"], 30)

# IV z-score（极端偏离）
merged["iv_z"] = (merged["dvol_level"] - merged["dvol_level"].rolling(20).mean()) / \
                 (merged["dvol_level"].rolling(20).std() + 1e-9)

# IV acceleration（IV 加速度）
merged["iv_acc"] = acceleration(merged["dvol_level"])


# ===== 4) VRP (IV - RV) 结构 =====

# VRP = IV - RV
merged["vrp"] = merged["dvol_level"] - merged["rv_24h"]

# VRP slope（风险溢价方向）
merged["vrp_slope"] = rolling_slope(merged["vrp"], 20)

# VRP z-score（风险偏好 regime）
merged["vrp_z"] = (merged["vrp"] - merged["vrp"].rolling(40).mean()) / \
                  (merged["vrp"].rolling(40).std() + 1e-9)


# ===== 5) VIX/OVX 扩散结构 =====

# 股市波动 vs 能源波动的扩散差
merged["vix_ovx_spread"] = merged["VIX_c"] - merged["OVX_c"]

# 扩散差的趋势方向（代表全球风险结构变化）
merged["vix_ovx_spread_slope"] = rolling_slope(merged["vix_ovx_spread"], 20)

# VIX 加速度（恐慌变化速度）
merged["vix_acc"] = acceleration(merged["VIX_c"])

# ==========================================================
# 🔥 Route 3 - 跨市场结构因子（强推18个）
# 依赖:
#   - rolling_slope()
#   - acceleration()
#   - merged["NASX_c"], merged["SPX_c"], merged["DXY_c"], merged["TLT_c"], merged["SHY_c"]
# ==========================================================
# =============== ① 趋势方向 （Slope） ==================
merged["nasx_slope"] = rolling_slope(merged["NASX_c"], 20)
merged["spx_slope"]  = rolling_slope(merged["SPX_c"], 20)
merged["dxy_slope"]  = rolling_slope(merged["DXY_c"], 20)
merged["tlt_slope"]  = rolling_slope(merged["TLT_c"], 20)
merged["shy_slope"]  = rolling_slope(merged["SHY_c"], 20)
# 科技股 vs 美元（风险偏好差）
merged["macro_risk_spread"] = merged["NASX_c"] - merged["DXY_c"]
# =============== ② 加速度（Acceleration） ==================
merged["nasx_acc"] = acceleration(merged["NASX_c"])
merged["spx_acc"]  = acceleration(merged["SPX_c"])
merged["dxy_acc"]  = acceleration(merged["DXY_c"])
merged["tlt_acc"]  = acceleration(merged["TLT_c"])
# =============== ③ β & Corr 结构 ==================
# β 的趋势
merged["beta_slope2"] = rolling_slope(merged["beta_spx"], 20)
merged["beta_acc"]    = acceleration(merged["beta_spx"])
# Corr 的趋势
merged["corr_slope2"] = rolling_slope(merged["corr_spx_eth"], 20)
merged["corr_acc"]    = acceleration(merged["corr_spx_eth"])
# =============== ④ 美元流动性结构 ==================
# 美元极端强弱（Z-score）
merged["dxy_z"] = (merged["DXY_c"] - merged["DXY_c"].rolling(40).mean()) / \
                  (merged["DXY_c"].rolling(40).std() + 1e-9)
# 风险偏好 × 美元流动性
# 美股涨 & 美元跌 → Crypto 强趋势
merged["liquidity_pulse"] = (-merged["dxy_slope"]) * merged["nasx_slope"]
# =============== ⑤ 全球风险结构扩散 ==================
# 股市 - 债市 风险扩散
merged["equity_bond_spread"] = merged["SPX_c"] - merged["TLT_c"]
# 股市 - 美元 风险扩散
merged["equity_usd_spread"] = merged["SPX_c"] - merged["DXY_c"]
# ==========================================================
# 🔥 完成：跨市场强推因子 18 个
# ==========================================================

# ==========================================================
# 🔥 Route 4 - 资金流结构因子（强推15个）
# 依赖:
#   - rolling_slope()
#   - acceleration()
#   - merged["eth_oi_c"], merged["funding_rate"], merged["crowding"]
#   - 你已有 funding_z, oi_z, crowding 等
# ==========================================================


# ====================== ① OI 结构因子 =========================

# OI slope（方向）
merged["oi_slope2"] = rolling_slope(merged["eth_oi_c"], 20)

# OI acceleration（二阶导）
merged["oi_acc"] = acceleration(merged["eth_oi_c"])

# OI z-score（极端持仓）
merged["oi_zscore"] = (merged["eth_oi_c"] - merged["eth_oi_c"].rolling(40).mean()) / \
                      (merged["eth_oi_c"].rolling(40).std() + 1e-9)

# OI 过去 5 根的波动率
merged["oi_pct_change_5"] = merged["eth_oi_c"].pct_change().rolling(5).std()

# OI range（高低幅度：杠杆堆积程度）
merged["oi_range"] = merged["eth_oi_c"].rolling(10).max() - merged["eth_oi_c"].rolling(10).min()



# ====================== ② Funding 结构因子 =========================

# Funding slope
merged["funding_slope2"] = rolling_slope(merged["funding_rate"], 20)

# Funding acceleration
merged["funding_acc"] = acceleration(merged["funding_rate"])

# Funding z-score（极端值）
merged["funding_z2"] = (merged["funding_rate"] - merged["funding_rate"].rolling(40).mean()) / \
                       (merged["funding_rate"].rolling(40).std() + 1e-9)

# Funding 的绝对强度
merged["funding_abs"] = merged["funding_rate"].abs()

# Funding sign（多空哪边付钱）
merged["funding_sign"] = np.sign(merged["funding_rate"])



# ====================== ③ Crowding 结构因子 =========================

# 拥挤度 slope
merged["crowding_slope"] = rolling_slope(merged["crowding"], 20)

# 拥挤度 acceleration
merged["crowding_acc"] = acceleration(merged["crowding"])

# 杠杆压力（强平压力 = 高 funding × 高 OI）
merged["leverage_pressure"] = merged["funding_abs"] * merged["oi_zscore"]

# squeeze risk（逼空风险 = funding_z >0 且 OI 上升）
merged["squeeze_risk"] = merged["funding_z2"] * np.sign(merged["oi_slope2"])

# cascade risk（强平链条风险 = funding_z × OI 加速度）
merged["cascade_risk"] = merged["funding_z2"] * merged["oi_acc"]

# ==========================================================
# 🔥 完成：Route 4 资金流结构因子（15个）
# ==========================================================
# ==========================================================
# 🔥 Route 5 - 特征变换类因子（偏度、峰度、熵、Regime）
# 需要:
#   merged['ret_4h'], merged['vol_range'], merged['dvol_level'],
#   merged['funding_rate'], rolling_slope, acceleration
# ==========================================================

import scipy.stats as ss
import numpy as np

# Helper: rolling skew/kurt
def rolling_skew(series, window):
    return series.rolling(window).apply(lambda x: ss.skew(x), raw=True)

def rolling_kurt(series, window):
    return series.rolling(window).apply(lambda x: ss.kurtosis(x), raw=True)

def rolling_entropy(series, window, bins=10):
    def _entropy(x):
        hist, _ = np.histogram(x, bins=bins, density=True)
        p = hist[hist > 0]
        return -np.sum(p * np.log(p))
    return series.rolling(window).apply(_entropy, raw=False)
# ==================== ① Skewness (偏度) ======================
merged["ret_skew_24"] = rolling_skew(merged["ret_4h"], 24)
merged["vol_skew_24"] = rolling_skew(merged["vol_range"], 24)
merged["iv_skew_24"]  = rolling_skew(merged["dvol_level"], 24)
# ==================== ② Kurtosis (峰度) ======================
merged["ret_kurt_24"] = rolling_kurt(merged["ret_4h"], 24)
merged["vol_kurt_24"] = rolling_kurt(merged["vol_range"], 24)
merged["iv_kurt_24"]  = rolling_kurt(merged["dvol_level"], 24)
# ==================== ③ Entropy (熵) ========================
merged["price_entropy_24"]   = rolling_entropy(merged["ret_4h"], 24)
merged["vol_entropy_24"]     = rolling_entropy(merged["vol_range"], 24)
merged["funding_entropy_24"] = rolling_entropy(merged["funding_rate"], 24)
# ==================== ④ Regime Switching ====================
# RV high/low regime
rv_rolling = merged["rv_24h"].rolling(60)
merged["rv_regime"] = (merged["rv_24h"] > rv_rolling.mean()).astype(int)
# IV high/low regime
iv_rolling = merged["dvol_level"].rolling(60)
merged["iv_regime"] = (merged["dvol_level"] > iv_rolling.mean()).astype(int)
# Funding 多头/空头 regime
merged["funding_regime"] = np.where(merged["funding_rate"] > 0, 1,
                             np.where(merged["funding_rate"] < 0, -1, 0))
# β regime（高金融化 vs 低金融化）
beta_roll = merged["beta_spx"].rolling(60)
merged["beta_regime"] = (merged["beta_spx"] > beta_roll.mean()).astype(int)
# HP trend slope regime
merged["trend_regime"] = np.sign(rolling_slope(merged["close"], 30))
# 突发波动 regime（volshock）
merged["volshock_regime"] = (merged["rv_24h"] > merged["rv_24h"].rolling(24).mean() * 1.5).astype(int)
# Liquidity regime（Nasdaq risk-on vs DXY 强势）
merged["liquidity_regime"] = np.sign(merged["NASX_c"] - merged["DXY_c"])
# 综合风险聚类（risk cluster）
merged["risk_cluster_regime"] = (
      merged["rv_regime"]
    + merged["iv_regime"]
    + merged["funding_regime"]
).apply(lambda x: np.sign(x))
# ==================== ⑤ Bonus: 波动能量 =======================
merged["volatility_energy"] = (merged["high"] - merged["low"])**2
# ==========================================================
# 完成：Route 5 强推特征变换因子（共 18 个）
# ==========================================================



# === 特征选择 ===
# 手工特征
features = [
    # 新加
    'TLT_c',
    'SHY_c',
    # 'eht_'
    # 趋势/动量
    'macd_hist',
    'macd_slope',
    'ma_diff_short',
    # 买卖量
    'eth_hist_vol',
    # 波动/形态
    'boll_width',
    'vol_ratio',
    'vix_jump10',
    'vix_z',
    # 资金位/仓位
    'oi_change',
    'funding_rate',
    'crowding',
    # 跨市场关系/宏观 beta
    'beta_spx',
    'corr_spx_eth',
    # 震荡/超买超卖
    'stoch_rsi',
    # 特征提取特征
    'VIX_c_rolling_std_3',
    'OVX_c_rolling_std_3',
    'VIX_c_rolling_std_5',
    # 宏观
    'NASX_smooth',
    'DXY_smooth',
    'DXY_det',
    'xauusd_det',
    # 额外
    # 'VIX_c_rolling_std_10', 'VIX_c_rolling_std_20', 'VIX_c_rolling_mean_10', 'iv_rv_gap', 'ex_oi'
    'ex_oi',
    'iv_rv_gap',
    'VIX_c_rolling_std_10',
    'VIX_c_rolling_std_20',
    'VIX_c_rolling_mean_10',
    'eth_coinmarketcap_volume',
    # 第一次扩充
    # price trend advanced
    "price_slope_10", "price_slope_20", "price_slope_50",
    "price_acc",
    "swing_high", "swing_low", "swing_momentum",
    # slopes of key factors
    "vix_slope", "oi_slope", "funding_slope",
    "beta_slope", "corr_slope",
    "iv_slope", "rv_slope",
    # 第二次扩充
    "rv_24h_slope", "rv_3d_z", "rv_acc",
    "vol_parkinson", "vol_range", "vol_body",
    "iv_slope2", "iv_z", "iv_acc",
    "vrp_slope", "vrp_z",
    "vix_acc", "vix_ovx_spread_slope",
    # 第三次扩充
    # 趋势 slope 类
    "nasx_slope", "spx_slope", "dxy_slope", "tlt_slope", "shy_slope", "macro_risk_spread",
    # acceleration 类
    "nasx_acc", "spx_acc", "dxy_acc", "tlt_acc",
    # β & corr 结构类
    "beta_slope2", "beta_acc",
    "corr_slope2", "corr_acc",
    # 美元流动性结构
    "dxy_z", "liquidity_pulse",
    # 全球风险结构扩散
    "equity_bond_spread", "equity_usd_spread",
    # 第四次扩充
    # OI 结构
    "oi_slope2", "oi_acc", "oi_zscore", "oi_pct_change_5", "oi_range",
    # Funding 结构
    "funding_slope2", "funding_acc", "funding_z2",
    "funding_abs", "funding_sign",
    # Crowding 结构
    "crowding_slope", "crowding_acc",
    "leverage_pressure", "squeeze_risk", "cascade_risk",
    # 第五次扩充
    # Skewness
    "ret_skew_24", "vol_skew_24", "iv_skew_24",
    # Kurtosis
    "ret_kurt_24", "vol_kurt_24", "iv_kurt_24",
    # Entropy
    "price_entropy_24", "vol_entropy_24", "funding_entropy_24",
    # Regime features
    "rv_regime", "iv_regime", "funding_regime",
    "beta_regime", "trend_regime", "volshock_regime",
    "liquidity_regime", "risk_cluster_regime",
    # Energy
    "volatility_energy",
]

print(f"✅ 最终特征总数: {len(features)}")


# ========= 预测目标 ============

merged['open']  = merged['eth_hist_o']       # 开盘价（Open）
merged['high']  = merged['eth_hist_h']       # 最高价（High）
merged['low']   = merged['eth_hist_l']       # 最低价（Low）
merged['close'] = merged['eth_hist_c']       # 收盘价（Close）
eps = 1e-9
# 目标 4
# merged['eth_samrt'] = (merged['high'] + merged['low'] + 3 * merged['close']) / 5.0
# merged['eth_samrt'] = (merged['high'] + merged['low'] + 2.5 * merged['close']) / 4.5
merged['eth_samrt'] = (merged['high'] + merged['low'] + 2 * merged['close']) / 4.0
# merged['eth_samrt'] = (merged['high'] + merged['low'] + 1.5 * merged['close']) / 3.5
# merged['eth_samrt'] = (merged['high'] + merged['low'] + merged['close']) / 3.0
# merged['eth_samrt'] = (merged['high'] + merged['low'] + 0.5 * merged['close']) / 2.5
# merged['eth_samrt'] = (merged['high'] + merged['low']) / 2.0
# merged['eth_samrt'] = merged['close']
merged['y_ret_samrt'] = np.log(merged['eth_samrt'].shift(-1) / merged['eth_samrt'])
merged['y_target'] = merged['y_ret_samrt']

merged['y_target'].hist(bins=100)
print(merged['y_target'].describe())
merged = merged.dropna(subset=['y_target']).reset_index(drop=True)
print(" Target已经生成 ")
print(merged[['datetime', 'eth_hist_c', 'y_target']].head())


# ====== 归一化 均值0 方差1 上下限+-4 ======
import numpy as np

CLIP = 4.0
WINDOW = 30 * 6   # 4h 一根 → 30天 ≈ 30*6 = 180 根

# 1) 取出全部特征
X_raw = merged[features].copy()

# 2) 为了只用“过去”的数据，先整体 shift(1)
X_shifted = X_raw.shift(1)
# X_shifted = X_raw

# 3) 用前 WINDOW 根计算滚动均值 / 方差（NaN 会被忽略）
roll_mean = X_shifted.rolling(window=WINDOW, min_periods=WINDOW).mean()
roll_std  = X_shifted.rolling(window=WINDOW, min_periods=WINDOW).std(ddof=0)

# 4) 计算滚动 z-score 并裁剪到 ±4
X_all = (X_raw - roll_mean) / roll_std
X_all = X_all.clip(-CLIP, CLIP)

# === 构建训练 / 测试集 ===
train_mask = (merged["datetime"] >= train_start) & (merged["datetime"] < train_end)
test_mask  = (merged["datetime"] >= test_start) & (merged["datetime"] < test_end)

X_train = X_all.loc[train_mask]
X_test  = X_all.loc[test_mask]
y_train = merged.loc[train_mask, "y_target"]
y_test  = merged.loc[test_mask, "y_target"]
time_test = merged.loc[test_mask, "datetime"]
print(f"特征维度: {X_train.shape[1]}, 训练样本: {len(X_train)}, 测试样本: {len(X_test)}")

# ==========================================
#  🔬 用 PyGAD 做 GA 特征选择（15~25 个特征）
# ==========================================
import pygad
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# ---- 你可以在这里选择 GA 的优化目标 ----
# 可选: "rmse"（测试集 RMSE 越小越好）
#       "sharpe"（测试集回测年化 Sharpe 越大越好）
#       "total_return"（测试集总收益率越高越好）
# GA_OBJECTIVE = "total_return"
# GA_OBJECTIVE = "max_drawdown_pct"
# GA_OBJECTIVE = "rmse"
# GA_OBJECTIVE = "sharpe"
GA_OBJECTIVE = "combo"

MIN_FEATURES = 10
MAX_FEATURES = 16

feature_names = X_train.columns.tolist()
n_features = len(feature_names)

# ======== 额外：特征分组（基于你当前的 29 个特征）========
feature_groups_by_name = [
    # Group 1: 价格趋势 / 形态结构（纯 price-based）
    # 用来抓 4h 级别的方向 + 局部结构（高低点、二阶导）
    [
        'macd_hist',
        'macd_slope',
        'ma_diff_short',

        'price_slope_10',
        'price_slope_20',
        'price_slope_50',
        'price_acc',

        'swing_high',
        'swing_low',
        'swing_momentum',
    ],

    # Group 2: 成交量 & 简单情绪
    # 4h 里比较稳定的量能 / 风险偏好 proxy
    [
        'eth_hist_vol',
        'eth_coinmarketcap_volume',
        'stoch_rsi',
    ],

    # Group 3: VIX / OVX & 价格波动形态（“外部波动 + K线形状”）
    # 用来刻画“外部恐慌 + 本地 K 线波动结构”
    [
        'boll_width',
        'vol_ratio',
        'vix_jump10',
        'vix_z',

        'VIX_c_rolling_std_3',
        'OVX_c_rolling_std_3',
        'VIX_c_rolling_std_5',
        'VIX_c_rolling_std_10',
        'VIX_c_rolling_std_20',
        'VIX_c_rolling_mean_10',

        'vix_slope',
        'vix_acc',
        'vix_ovx_spread_slope',

        'vol_parkinson',
        'vol_range',
        'vol_body',

        'vol_skew_24',
        'vol_kurt_24',
        'volatility_energy',
    ],

    # Group 4: IV / RV / VRP 结构（“期权视角的波动结构”）
    # 专门负责期权隐含波动 & 实现波动 & 风险溢价
    [
        'iv_rv_gap',

        'rv_slope',
        'rv_24h_slope',
        'rv_3d_z',
        'rv_acc',

        'iv_slope',
        'iv_slope2',
        'iv_z',
        'iv_acc',

        'vrp_slope',
        'vrp_z',

        'ret_skew_24',
        'ret_kurt_24',

        'iv_skew_24',
        'iv_kurt_24',

        'price_entropy_24',
    ],

    # Group 5: OI / Funding / Crowding 结构（“链上 & 衍生品 仓位 / 杠杆”）
    # 4h 交易中最核心的一类：什么时候仓位堆积、什么时候容易爆仓 / squeeze
    [
        'oi_change',
        'funding_rate',
        'crowding',
        'ex_oi',

        'oi_slope',
        'oi_slope2',
        'oi_acc',
        'oi_zscore',
        'oi_pct_change_5',
        'oi_range',

        'funding_slope',
        'funding_slope2',
        'funding_acc',
        'funding_z2',
        'funding_abs',
        'funding_sign',

        'crowding_slope',
        'crowding_acc',

        'leverage_pressure',
        'squeeze_risk',
        'cascade_risk',
    ],

    # Group 6: ETH–SPX 相关性 / β 结构（“金融化 / 风险联动”）
    # 专门描述 ETH 和美国股市的联动强度
    [
        'beta_spx',
        'corr_spx_eth',

        'beta_slope',
        'corr_slope',

        'beta_slope2',
        'beta_acc',

        'corr_slope2',
        'corr_acc',

        'beta_regime',
    ],

    # Group 7: 宏观指数 & 利率 / 美元 / 债券扩散（“大环境 Risk-on/off”）
    # NASX / SPX / DXY / TLT / SHY / 黄金 + 各种 slope/acc/spread
    [
        'NASX_smooth',
        'DXY_smooth',
        'DXY_det',
        'xauusd_det',
        'TLT_c',
        'SHY_c',

        'nasx_slope',
        'spx_slope',
        'dxy_slope',
        'tlt_slope',
        'shy_slope',
        'macro_risk_spread',

        'nasx_acc',
        'spx_acc',
        'dxy_acc',
        'tlt_acc',

        'dxy_z',
        'liquidity_pulse',

        'equity_bond_spread',
        'equity_usd_spread',

        'liquidity_regime',
    ],

    # Group 8: 分布尾部 / 熵 / Regime（“状态标签层”）
    # 把各种 regime / entropy 集中到一组，更多是状态编码
    [
        'rv_regime',
        'iv_regime',
        'funding_regime',

        'trend_regime',
        'volshock_regime',
        'risk_cluster_regime',

        'funding_entropy_24',
        'vol_entropy_24',
    ],
]

# 将名字映射到索引
name2idx = {name: i for i, name in enumerate(feature_names)}

feature_groups_idx = []
for group in feature_groups_by_name:
    idxs = []
    for feat in group:
        if feat not in name2idx:
            print(f"⚠️ 分组警告：特征 {feat} 不在 feature_names 里（可能是没放进 features 列表或拼写错误）")
            continue
        idxs.append(name2idx[feat])
    if idxs:
        feature_groups_idx.append(np.array(idxs, dtype=int))

# 每组最多 0~3 个特征
MAX_PER_GROUP = 3
# 如果以后想要“每组至少 1 个”，可以再加 MIN_PER_GROUP = 1 之类的




# ===== 新增：多场景列表（scenario_list） =====
# 先用你当前这一刀 Train/Test 做成第一个场景
# ===== 多场景 Train/Test 设计（GA 用）=====
def make_mask(start, end):
    return (merged["datetime"] >= start) & (merged["datetime"] < end)


# === 场景 1：Test = 2024-10-01 ~ 2024-11-01, Train = 前 8 个月 ===
train1_start = pd.Timestamp("2024-02-01")
train1_end   = pd.Timestamp("2024-10-01")
test1_start  = pd.Timestamp("2024-10-01")
test1_end    = pd.Timestamp("2024-11-01")

train1_mask = make_mask(train1_start, train1_end)
test1_mask  = make_mask(test1_start,  test1_end)

# === 场景 2：Test = 2024-11-01 ~ 2024-12-01, Train = 前 8 个月 ===
train2_start = pd.Timestamp("2024-02-01")
train2_end   = pd.Timestamp("2024-11-01")
test2_start  = pd.Timestamp("2024-11-01")
test2_end    = pd.Timestamp("2024-12-01")

train2_mask = make_mask(train2_start, train2_end)
test2_mask  = make_mask(test2_start,  test2_end)

# === 场景 3：Test = 2024-12-01 ~ 2025-01-01, Train = 前 8 个月 ===
train3_start = pd.Timestamp("2024-02-01")
train3_end   = pd.Timestamp("2024-12-01")
test3_start  = pd.Timestamp("2024-12-01")
test3_end    = pd.Timestamp("2025-01-01")

train3_mask = make_mask(train3_start, train3_end)
test3_mask  = make_mask(test3_start,  test3_end)

# === 场景 4：Test = 2025-01-01 ~ 2025-02-01, Train = 前 8 个月 ===
train4_start = pd.Timestamp("2024-02-01")
train4_end   = pd.Timestamp("2025-01-01")
test4_start  = pd.Timestamp("2025-01-01")
test4_end    = pd.Timestamp("2025-02-01")

train4_mask = make_mask(train4_start, train4_end)
test4_mask  = make_mask(test4_start,  test4_end)

# === 场景 5：Test = 2025-02-01 ~ 2025-03-01, Train = 前 8 个月 ===
train5_start = pd.Timestamp("2024-02-01")
train5_end   = pd.Timestamp("2025-02-01")
test5_start  = pd.Timestamp("2025-02-01")
test5_end    = pd.Timestamp("2025-03-01")

train5_mask = make_mask(train5_start, train5_end)
test5_mask  = make_mask(test5_start,  test5_end)

# === 场景 6：Test = 2025-03-01 ~ 2025-04-01, Train = 前 8 个月 ===
train6_start = pd.Timestamp("2024-02-01")
train6_end   = pd.Timestamp("2025-03-01")
test6_start  = pd.Timestamp("2025-03-01")
test6_end    = pd.Timestamp("2025-04-01")

train6_mask = make_mask(train6_start, train6_end)
test6_mask  = make_mask(test6_start,  test6_end)

# === 场景 7：Test = 2025-04-01 ~ 2025-05-01, Train = 前 8 个月 ===
train7_start = pd.Timestamp("2024-02-01")
train7_end   = pd.Timestamp("2025-04-01")
test7_start  = pd.Timestamp("2025-04-01")
test7_end    = pd.Timestamp("2025-05-01")

train7_mask = make_mask(train7_start, train7_end)
test7_mask  = make_mask(test7_start,  test7_end)

# 为每个场景准备对应的 close / time（给回测函数用）
scenario_list = [
    {
        "name": "S1_Train_20240201_20241001__Test_20241001_20241101",
        "train_mask": train1_mask,
        "test_mask":  test1_mask,
        "close_test": merged.loc[test1_mask, "eth_hist_c"].reset_index(drop=True),
        "time_test":  merged.loc[test1_mask, "datetime"].reset_index(drop=True),
    },
    {
        "name": "S2_Train_20240301_20241101__Test_20241101_20241201",
        "train_mask": train2_mask,
        "test_mask":  test2_mask,
        "close_test": merged.loc[test2_mask, "eth_hist_c"].reset_index(drop=True),
        "time_test":  merged.loc[test2_mask, "datetime"].reset_index(drop=True),
    },
    {
        "name": "S3_Train_20240401_20241201__Test_20241201_20250101",
        "train_mask": train3_mask,
        "test_mask":  test3_mask,
        "close_test": merged.loc[test3_mask, "eth_hist_c"].reset_index(drop=True),
        "time_test":  merged.loc[test3_mask, "datetime"].reset_index(drop=True),
    },
    {
        "name": "S4_Train_20240501_20250101__Test_20250101_20250201",
        "train_mask": train4_mask,
        "test_mask":  test4_mask,
        "close_test": merged.loc[test4_mask, "eth_hist_c"].reset_index(drop=True),
        "time_test":  merged.loc[test4_mask, "datetime"].reset_index(drop=True),
    },
    {
        "name": "S5_Train_20240601_20250201__Test_20250201_20250301",
        "train_mask": train5_mask,
        "test_mask":  test5_mask,
        "close_test": merged.loc[test5_mask, "eth_hist_c"].reset_index(drop=True),
        "time_test":  merged.loc[test5_mask, "datetime"].reset_index(drop=True),
    },
    {
        "name": "S6_Train_20240701_20250301__Test_20250301_20250401",
        "train_mask": train6_mask,
        "test_mask":  test6_mask,
        "close_test": merged.loc[test6_mask, "eth_hist_c"].reset_index(drop=True),
        "time_test":  merged.loc[test6_mask, "datetime"].reset_index(drop=True),
    },
    {
        "name": "S7_Train_20240801_20250401__Test_20250401_20250501",
        "train_mask": train7_mask,
        "test_mask":  test7_mask,
        "close_test": merged.loc[test7_mask, "eth_hist_c"].reset_index(drop=True),
        "time_test":  merged.loc[test7_mask, "datetime"].reset_index(drop=True),
    },
]

print("✅ 场景列表构建完成：")
for s in scenario_list:
    print(s["name"],
          " | Train 样本数:", int(s["train_mask"].sum()),
          " | Test 样本数:",  int(s["test_mask"].sum()))

def run_backtest_simple(
    y_pred_test,
    close_series,
    time_series,
    initial_capital=1_000_000.0,
    trade_margin=1_000_000.0,
    lev_long=1.0,
    lev_short=1.0,
    round_trip_cost_bps=0.0,
    bars_per_day=6,
):
    """
    用你当前脚本的极简 T→T+1 回测逻辑，但去掉了打印，只返回 Sharpe/收益等指标。
    """
    # === 信号生成：完全照你原来的写法 ===
    sig = np.array(y_pred_test, dtype=float)
    tttt = 0
    low, high = np.percentile(sig, [50-tttt, 50+tttt])
    sig_filtered = np.where((sig > low) & (sig < high), 0,
                            np.where(sig >= high, 1, -1))
    signal = pd.Series(sig_filtered)

    close = close_series.reset_index(drop=True).copy()
    time_s = time_series.reset_index(drop=True).copy()

    N = min(len(signal), len(close), len(time_s))
    signal = signal.iloc[:N].reset_index(drop=True)
    close  = close.iloc[:N].reset_index(drop=True)
    time_s = time_s.iloc[:N].reset_index(drop=True)

    # t→t+1 收益
    ret_next = close.pct_change().shift(-1)
    valid_mask = ret_next.notna()
    signal = signal[valid_mask].reset_index(drop=True)
    close  = close[valid_mask].reset_index(drop=True)
    time_s = time_s[valid_mask].reset_index(drop=True)
    ret_next = ret_next[valid_mask].reset_index(drop=True)

    equity = initial_capital
    equity_curve = [equity]
    cost_rate = round_trip_cost_bps / 1e4

    for t in range(len(signal)):
        side = int(signal.iloc[t])
        if side == 0:
            equity_curve.append(equity)
            continue

        px_in  = close.iloc[t]
        px_out = close.iloc[t+1] if t+1 < len(close) else px_in

        margin_used = trade_margin
        lev = lev_long if side == 1 else lev_short
        notional = margin_used * lev

        gross_pnl = notional * side * ret_next.iloc[t]
        cost = notional * cost_rate

        equity += (gross_pnl - cost)
        equity_curve.append(equity)

    equity_curve = pd.Series(equity_curve)
    ret_4h = equity_curve.pct_change().fillna(0)

    # 年化 Sharpe
    BARS_PER_YEAR = bars_per_day * 365
    std_ret = ret_4h.std()
    if std_ret < 1e-12:
        sharpe_annual = 0.0
    else:
        sharpe_annual = (ret_4h.mean() / std_ret) * np.sqrt(BARS_PER_YEAR)

    total_ret_pct = (equity / initial_capital - 1.0) * 100.0

    # 最大回撤
    rolling_max = equity_curve.cummax()
    drawdown = (rolling_max - equity_curve) / rolling_max
    max_dd_pct = drawdown.max() * 100.0

    return {
        "final_equity": equity,
        "total_return_pct": total_ret_pct,
        "sharpe_annual": sharpe_annual,
        "max_drawdown_pct": max_dd_pct,
        "equity_curve": equity_curve,
    }


# LightGBM 参数，和你上面训练时基本一致（可以自己微调）
base_lgbm_params = dict(
    n_estimators=500,
    learning_rate=0.025,
    max_depth=5,
    # max_depth=3,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=0.2,
    reg_alpha=0.4,
    min_child_samples=80,
    random_state=42,
    verbose=-1,
    force_col_wise=True,
    n_jobs=-1,
)

MAX_PER_GROUP = 3


def repair_solution(mask: np.ndarray) -> np.ndarray:
    """
    统一修剪函数：
    1) 每组最多 MAX_PER_GROUP 个 1
    2) 全局特征数在 [MIN_FEATURES, MAX_FEATURES] 内
    """
    m = np.array(mask, dtype=int).copy()

    # ---- 1) 组内约束：每组 ≤ MAX_PER_GROUP ----
    if feature_groups_idx:  # 防御一下
        for g_idxs in feature_groups_idx:
            g_idxs = np.asarray(g_idxs, dtype=int)
            active_in_group = np.where(m[g_idxs] == 1)[0]  # 组内索引（0~len(g)-1）

            if len(active_in_group) > MAX_PER_GROUP:
                # 需要关掉的个数
                n_to_off = len(active_in_group) - MAX_PER_GROUP
                # 在“这一组里被选中的位置”中随机挑 n_to_off 个关掉
                off_in_group = np.random.choice(active_in_group, size=n_to_off, replace=False)
                # 映射回全局索引
                off_global = g_idxs[off_in_group]
                m[off_global] = 0

    # ---- 2) 全局特征数约束：总个数在 [MIN_FEATURES, MAX_FEATURES] ----
    k = int(m.sum())

    # 2.a 超过 MAX_FEATURES：随机关掉多余的 1（不会破坏“每组 ≤ MAX_PER_GROUP”）
    if k > MAX_FEATURES:
        ones_idx = np.where(m == 1)[0]
        n_to_off = k - MAX_FEATURES
        off_idx = np.random.choice(ones_idx, size=n_to_off, replace=False)
        m[off_idx] = 0
        k = int(m.sum())

    # 2.b 少于 MIN_FEATURES：在不违反“每组 ≤ MAX_PER_GROUP”的前提下开一些 0
    if k < MIN_FEATURES:
        zero_idx = np.where(m == 0)[0]
        np.random.shuffle(zero_idx)  # 打乱，随机挑

        for idx in zero_idx:
            # 看看这个 idx 是否属于某个组，如果属于且该组已经满了（=MAX_PER_GROUP）就不能开
            ok_to_turn_on = True
            for g_idxs in feature_groups_idx:
                g_idxs = np.asarray(g_idxs, dtype=int)
                if idx in g_idxs:
                    if m[g_idxs].sum() >= MAX_PER_GROUP:
                        ok_to_turn_on = False
                    break  # 一个 idx 只会在一个组里

            if not ok_to_turn_on:
                continue

            # 可以开
            m[idx] = 1
            k += 1
            if k >= MIN_FEATURES:
                break

    return m


# ============ PyGAD 的适应度函数 ============
def fitness_func(ga_instance, solution, solution_idx):
    """
    solution 是一个长度 = n_features 的 0/1 向量（是否选择特征）。
    现在：对 scenario_list 里的所有 (Train, Test) 场景做打分，
    用“多场景表现”来作为 GA 的 fitness（越大越好）。
    """
    # 先把解修剪成“合法”的：
    #   - 每组 ≤ MAX_PER_GROUP
    #   - 全局个数在 [MIN_FEATURES, MAX_FEATURES]
    mask = repair_solution(solution)
    k = int(mask.sum())

    # 理论上这里已经满足约束了，可以不用再 if
    # 但为了保险，可以留一道防线（几乎不会触发）
    if (k < MIN_FEATURES) or (k > MAX_FEATURES):
        return -1e9
    # 1.b) 分组约束：每一组里最多选 MAX_PER_GROUP 个特征
    # if feature_groups_idx:   # 防御：万一你后面把分组清空了就不检查
    #     for g_idxs in feature_groups_idx:
    #         cnt_g = int(mask[g_idxs].sum())
    #         if cnt_g > MAX_PER_GROUP:
    #             # 这一条染色体在某个组里选太多特征 → 直接给很差的适应度
    #             return -1e9

            # 如果以后想加“每组至少选 1 个”的约束，也可以在这里加：
            # if cnt_g < MIN_PER_GROUP:
            #     return -1e9
    cols = [feature_names[i] for i, m in enumerate(mask) if m == 1]
    if len(cols) == 0:
        return -1e9

    # 2) 在每个场景上分别训练+评估
    rmse_list = []
    sharpe_list = []
    ret_list = []
    dd_list = []

    for scen in scenario_list:
        train_mask_s = scen["train_mask"]
        test_mask_s  = scen["test_mask"]

        # 用全局 X_all / merged 按 mask 切子集
        X_tr = X_all.loc[train_mask_s, cols]
        X_te = X_all.loc[test_mask_s,  cols]
        y_tr = merged.loc[train_mask_s, "y_target"]
        y_te = merged.loc[test_mask_s,  "y_target"]

        # 如果某个场景样本太少，就跳过
        if len(X_tr) < 100 or len(X_te) < 50:
            continue

        model = LGBMRegressor(**base_lgbm_params)
        model.fit(X_tr, y_tr)

        y_pred_test = model.predict(X_te)

        if GA_OBJECTIVE == "rmse":
            rmse = np.sqrt(mean_squared_error(y_te, y_pred_test))
            rmse_list.append(rmse)
        else:
            bt = run_backtest_simple(
                y_pred_test,
                scen["close_test"],
                scen["time_test"],
            )
            sharpe_list.append(bt["sharpe_annual"])
            ret_list.append(bt["total_return_pct"])
            dd_list.append(bt["max_drawdown_pct"])

    # 3) 把多个场景的结果聚合成一个分数（越大越好）
    # 3) 把多个场景的结果聚合成一个分数（越大越好）
    if GA_OBJECTIVE == "rmse":
        if not rmse_list:
            return -1e9
        # 平均 RMSE 越小越好 → 取负
        return -float(np.mean(rmse_list))

    # 回测指标：若一个都没算出来，说明这组 F 太烂/没数据，直接给差评
    if not sharpe_list and not ret_list and not dd_list:
        return -1e9

    # === 组合目标：Sharpe + 收益 - 回撤 ===
    if GA_OBJECTIVE == "combo":
        if (not sharpe_list) or (not ret_list) or (not dd_list):
            return -1e9

        mean_sharpe = float(np.mean(sharpe_list))      # 例如 2~6
        mean_ret_pct = float(np.mean(ret_list))        # 单位：%
        worst_dd_pct = float(np.max(dd_list))          # 单位：%

        # 简单归一化一下百分比，避免量纲差太多
        mean_ret = mean_ret_pct / 100.0
        worst_dd = worst_dd_pct / 100.0

        # 权重可以调，这里先给一个比较直觉的：
        w_sharpe = 1.0   # 夏普权重
        w_ret    = 0.5   # 收益权重（已经 /100）
        w_dd     = 2.0   # 回撤惩罚（已经 /100）

        score = w_sharpe * mean_sharpe + w_ret * mean_ret - w_dd * worst_dd
        return score

    # 保留原来的单目标分支以防你以后想切换
    if GA_OBJECTIVE == "sharpe":
        return float(np.mean(sharpe_list)) if sharpe_list else -1e9

    if GA_OBJECTIVE == "total_return":
        return float(np.mean(ret_list)) if ret_list else -1e9

    if GA_OBJECTIVE == "max_drawdown_pct":
        if not dd_list:
            return -1e9
        worst_dd = np.max(dd_list)  # 百分比
        return -float(worst_dd)

    # 默认：如果 GA_OBJECTIVE 写错了，就用平均 Sharpe 顶上
    return float(np.mean(sharpe_list)) if sharpe_list else -1e9


def on_generation(ga_instance):
    pop = ga_instance.population
    for i in range(pop.shape[0]):
        pop[i, :] = repair_solution(pop[i, :])
    ga_instance.population = pop


# ============ 配置并运行 GA ============
from collections import Counter

# === 可调参数：你自己改这两个数字即可 ===
N_RUNS = 10          # 跑多少条随机路径（不同 random_seed）
TOP_K_PER_RUN = 20   # 每条路径取最后一代里前多少个个体
STABLE_THRESH = 0.33  # 频率阈值，比如 60% 以上视为“稳定特征”

all_feature_sets = []  # 用来存所有 run 的 Top-K 特征子集

for run_idx in range(N_RUNS):
    seed = 2025 + run_idx  # 你也可以自己定义一个 seed 列表

    print(f"\n🚀 开始 GA 特征选择（PyGAD）... Run {run_idx+1}/{N_RUNS}, random_seed={seed}")

    # 每次都重新建一个 GA 实例（唯一改动是 random_seed）
    ga = pygad.GA(
        num_generations=30,  # 迭代次数
        num_parents_mating=30,  # 父代数
        fitness_func=fitness_func,  # 适应度函数
        sol_per_pop=50,
        num_genes=n_features,
        gene_space=[0, 1],
        gene_type=int,
        parent_selection_type="tournament",
        K_tournament=4,
        crossover_type="two_points",
        mutation_type="random",
        mutation_percent_genes=2,
        random_seed=seed,
        stop_criteria=None,
        on_generation=on_generation
    )

    ga.run()

    # ====== 1️⃣ 取出这一条路径的最好一个解（方便打印看一眼） ======
    solution, solution_fitness, solution_idx = ga.best_solution()
    best_mask = np.array(solution, dtype=int)
    best_features = [f for f, m in zip(feature_names, best_mask) if m == 1]

    print(f"[Run {run_idx+1}] ✅ 最佳适应度值: {solution_fitness:.6f}")
    print(f"[Run {run_idx+1}] 最佳特征数量: {len(best_features)}")
    print(f"[Run {run_idx+1}] 最佳特征列表:")
    print(best_features)

    # ====== 2️⃣ 从最后一代里选出 Top-K 个个体 ======
    pop = ga.population                      # shape: (sol_per_pop, num_genes)
    fitness_vals = []

    for idx, sol in enumerate(pop):
        f = fitness_func(ga, sol, idx)       # 重新算一遍 fitness（和 GA 内部同一个函数）
        fitness_vals.append(f)

    fitness_vals = np.array(fitness_vals)
    # 取 fitness 最大的 TOP_K_PER_RUN 个索引
    top_idx = np.argsort(fitness_vals)[-TOP_K_PER_RUN:][::-1]

    print(f"[Run {run_idx+1}] 选出最后一代 Top-{TOP_K_PER_RUN} 个特征子集用于统计频率：")
    for rank, idx in enumerate(top_idx, start=1):
        sol = pop[idx]
        mask = np.array(sol, dtype=int)
        feats = [f for f, m in zip(feature_names, mask) if m == 1]
        fit_val = fitness_vals[idx]

        all_feature_sets.append(feats)

        print(f"  - Top {rank}: fitness={fit_val:.6f}, n_feats={len(feats)}")
        print(f"    特征: {feats}")

# ====== 3️⃣ 统计所有 run + Top-K 的特征出现频率 ======
cnt = Counter()
for feats in all_feature_sets:
    cnt.update(feats)

total_sets = len(all_feature_sets)  # 理论上 = N_RUNS * TOP_K_PER_RUN

# 按出现次数从高到低排序
sorted_feats = sorted(cnt.items(), key=lambda x: x[1], reverse=True)

print("\n📊 特征出现频次（按从高到低排序）:")
for feat, c in sorted_feats:
    freq = c / total_sets
    print(f"{feat}: 出现 {c} 次 / 共 {total_sets} 组 ({freq:.1%})")

# ====== 3.b 按“类别/分组”统计特征被选择的概率 ======
print("\n📂 按类别统计特征被选择的概率：")

for gid, group in enumerate(feature_groups_by_name, start=1):
    # 过滤掉在 GA 中完全没出现过的特征（比如你后来删掉了某个因子）
    group_feats = [f for f in group if f in cnt]

    if not group_feats:
        print(f"\n[Group {gid}]（这一组在本次 GA 中没有任何特征被选中过）")
        continue

    # 统计：有“至少一个本组特征被选中”的 染色体个数
    group_hit_sets = 0
    for feats in all_feature_sets:
        if any(f in feats for f in group_feats):
            group_hit_sets += 1
    group_hit_prob = group_hit_sets / total_sets  # 组层面的命中概率

    # 统计：组内特征的平均选择概率
    per_feat_probs = [cnt[f] / total_sets for f in group_feats]
    avg_prob = float(np.mean(per_feat_probs))

    print(f"\n[Group {gid}] 特征列表: {group_feats}")
    print(f"  - 至少选择该组 ≥1 个特征的概率: {group_hit_prob:.1%}")
    print(f"  - 组内特征【平均】被选择概率: {avg_prob:.1%}")
    print("  - 组内各特征选择概率：")
    for f in group_feats:
        p = cnt[f] / total_sets
        print(f"    · {f}: {cnt[f]} / {total_sets} ({p:.1%})")

# ====== 4️⃣ 按频率挑“稳定特征” ======
stable_features = [feat for feat, c in sorted_feats if c / total_sets >= STABLE_THRESH]

print(f"\n✨ 稳定特征（出现频率 ≥ {STABLE_THRESH:.0%}）：共 {len(stable_features)} 个")
print(stable_features)

# ====== 5️⃣ 用稳定特征构造最终训练 / 测试矩阵 ======
if len(stable_features) == 0:
    print("⚠️ 没有特征达到稳定阈值，建议降低 STABLE_THRESH 或检查 GA 设置。")
    # 兜底：退回最后一次 run 的 best_features
    final_features = best_features
    print("使用最后一次 GA 的最佳特征作为兜底：")
else:
    final_features = stable_features

print(f"\n🎯 最终采用特征数量: {len(final_features)}")
print(final_features)

X_train_ga = X_train[final_features].copy()
X_test_ga  = X_test[final_features].copy()
print(f"🎯 GA 稳健特征维度: {X_train_ga.shape[1]}")




import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class LightBGM:
    """
    训练 + 预测 + 评估：完全按你原来 quick_try 那套写法
    用法：
        m = LightBGM()
        m.fit(X_train, y_train, X_test, y_test, plot_rmse=True)
        y_pred_test = m.predict(X_test)
        imp = m.feature_importance()
    """

    def __init__(self, random_state: int = 42, ):
        self.random_state = random_state

        # ====== 损失函数 ======
        # objective = "mix"
        objective = "regression"
        # objective = "regression_l1"
        # objective = "huber"
        # objective = "fair"
        # objective = "quantile"

        OBJECTIVE_CFG = {
            "regression": ("rmse", {}),
            "regression_l1": ("l1", {}),
            "huber": ("huber", {"alpha": 0.9}),
            "fair": ("fair", {"fair_c": 1.0}),
            "quantile": ("quantile", {"alpha": 0.5}),
        }

        # def mix_obj(y_true, y_pred):
        #     e = y_pred - y_true
        #     ae = np.abs(e)
        #     eps = 1e-6
        #
        #     # ====== 权重：先全部按 0.5 给 ======
        #     w_l2 = 0.4  # L2 / regression RMSE 0.0077左右
        #     w_l1 = 0.0  # L1 / regression_l1 L1 拉完了 不能用 学不到东西
        #     w_huber = 0.6  # Huber 这个管用 曲线平滑 貌似 RSME很低搞的 0.0078 这种 不上翘
        #     w_fair = 0.0  # Fair 这个管用 但是管用到什么程度难说 这个能让rsme也降低
        #     w_q = 0.0  # Quantile 这个没用
        #
        #     # ====== 参数 ======
        #     delta = 0.003  # Huber 阈值
        #     c = 1.0  # Fair 参数
        #     alpha = 0.5  # Quantile 分位数
        #
        #     # grad =
        #     # w_l2    * L2_grad       = w_l2    * e
        #     # w_l1    * L1_grad       = w_l1    * sign(e)
        #     # w_huber * Huber_grad    = w_huber * where(|e|<=delta, e, delta*sign(e))
        #     # w_fair  * Fair_grad     = w_fair  * c*e/(|e|+c)
        #     # w_q     * Quantile_grad = w_q     * where(e>0, 1-alpha, -alpha)
        #     grad = (
        #             w_l2 * e
        #             + w_l1 * np.sign(e)
        #             + w_huber * np.where(ae <= delta, e, delta * np.sign(e))
        #             + w_fair * (c * e / (ae + c))
        #             + w_q * np.where(e > 0, 1 - alpha, -alpha)
        #     )
        #
        #     # hess =
        #     # w_l2    * L2_hess       = w_l2    * 1
        #     # w_l1    * L1_hess       = w_l1    * eps
        #     # w_huber * Huber_hess    = w_huber * where(|e|<=delta, 1, eps)
        #     # w_fair  * Fair_hess     = w_fair  * c^2/(|e|+c)^2
        #     # w_q     * Quantile_hess = w_q     * eps
        #     hess = (
        #             w_l2 * np.ones_like(e)
        #             + w_l1 * eps
        #             + w_huber * np.where(ae <= delta, 1.0, eps)
        #             + w_fair * (c ** 2 / ((ae + c) ** 2))
        #             + w_q * eps
        #     )
        #
        #     return grad, hess

        # Tail-Weighted L2 讲的头头是道 实际上拉完了
        # def mix_obj(y_true, y_pred):
        #     e = y_pred - y_true
        #     eps = 1e-6
        #
        #     q = np.nanquantile(np.abs(y_true), 0.90)
        #     w = 1.0 + 4.0 * np.clip(np.abs(y_true) / (q + eps), 0, 3.0)
        #
        #     grad = w * e
        #     hess = w
        #
        #     return grad, hess

        # 方向 × 幅度损失 Directional + L2 改良
        def mix_obj(y_true, y_pred):
            eps = 1e-6

            e = y_pred - y_true

            s = np.sign(y_true)
            a = np.abs(y_true)

            beta = 300.0
            z = np.clip(beta * s * y_pred, -30, 30)

            # 方向损失
            grad_dir = -a * beta * s / (1.0 + np.exp(z))
            hess_dir = a * beta ** 2 * np.exp(z) / ((1.0 + np.exp(z)) ** 2)

            # L2损失
            grad_l2 = e
            hess_l2 = np.ones_like(e)

            # L2 + 方向损失
            grad = 0.8 * grad_l2 + 0.2 * grad_dir
            hess = 0.8 * hess_l2 + 0.2 * hess_dir

            hess = np.maximum(hess, eps)
            return grad, hess

        # def mix_obj(y_true, y_pred):
        #     eps = 1e-6
        #
        #     e = y_pred - y_true
        #     s = np.sign(y_true)
        #     ay = np.abs(y_true)
        #
        #     # tail weight：越异常，权重越大
        #     q = np.nanquantile(ay, 0.666) + eps
        #     w = np.clip(ay / q, 0.0, 5.0)
        #
        #     # L2：保留回归幅度
        #     grad_l2 = w * e
        #     hess_l2 = w
        #
        #     # Direction Focal：大波动样本方向错了重罚
        #     beta = 300.0
        #     z = np.clip(beta * s * y_pred, -30, 30)
        #
        #     grad_dir = -w * beta * s / (1.0 + np.exp(z))
        #     hess_dir = w * beta ** 2 * np.exp(z) / ((1.0 + np.exp(z)) ** 2)
        #
        #     # 组合：L2 为主，方向为辅
        #     grad = 0.8 * grad_l2 + 0.0 * grad_dir
        #     hess = 0.8 * hess_l2 + 0.0 * hess_dir
        #
        #     hess = np.maximum(hess, eps)
        #     return grad, hess


        self.objective = objective

        if objective == "mix":
            lgb_objective = mix_obj
            self.eval_metric = "rmse"
            extra_params = {}
        else:
            self.eval_metric, extra_params = OBJECTIVE_CFG[objective]
            lgb_objective = objective

        self.model = LGBMRegressor(
            objective=lgb_objective,
            # intervaltree=[[0,1,2],[3,4,5],[6,7,8,9,10,11,12]],
            # interaction_constraints=[
            #     [0, 1, 2],
            #     [3, 4, 5],
            #     # [6, 7, 8, 9, 10, 11, 12],
            # ],
            n_estimators=500,          # 不要太多
            learning_rate=0.025,       # 稍低，慢学一点
            max_depth=5,              # 降低深度（防止细分噪声）
            num_leaves=15,            # 控制叶节点数量
            subsample=0.8,            # 每次用80%的样本
            colsample_bytree=0.8,     # 每棵树用80%的特征
            reg_lambda=0.2,           # L2正则增强
            reg_alpha=0.4,            # 加一点L1
            min_child_samples=80,     # 每个叶子至少80样本（防止过拟合）
            random_state=self.random_state,
            verbose=-1,
            force_col_wise=True,
            n_jobs=-1,
            **extra_params,
        )

        self._evals_result = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sample_weight=None,
        plot_rmse: bool = False
    ):
        # print("\n📈 开始训练 LightGBM 模型（监控 RMSE 曲线）...")

        fit_kwargs = dict(
            X=X_train,
            y=y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_names=['Train', 'Test'],
            eval_metric=self.eval_metric
        )

        # 如果传了权重，就加权训练；没传就还是原来的等权训练
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        self.model.fit(**fit_kwargs)

        # 记录 eval 曲线
        self._evals_result = self.model.evals_result_

        # 预测
        y_pred_train = self.model.predict(X_train)
        y_pred_test  = self.model.predict(X_test)

        # 评估
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2  = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)

        # print(f"✅ 训练集 R²: {train_r2:.4f}, RMSE: {train_rmse:.6f}")
        # print(f"✅ 测试集 R²: {test_r2:.4f}, RMSE: {test_rmse:.6f}, MAE: {mae:.6f}")

        # 输出重要性占比（你原来那样）
        imp_df = pd.DataFrame({
            "feature": self.model.feature_name_,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        s = imp_df["importance"] / (imp_df["importance"].sum() + 1e-12)
        # print(s)

        # 可选画图：只画测试集 RMSE
        if plot_rmse:
            self.plot_test_rmse()

        return y_pred_test

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def evals_result(self):
        return self._evals_result

    def plot_test_rmse(self):
        if self._evals_result is None:
            print("[WARN] 没有 evals_result，请先 fit()")
            return

        results = self._evals_result
        test_rmse_list = results['Test'][self.eval_metric]

        plt.figure(figsize=(10, 5))
        plt.plot(test_rmse_list, label=f'Test {self.eval_metric}', linewidth=2)
        plt.title(f'LightGBM 测试集 {self.eval_metric} 曲线')
        plt.xlabel('Boosting Iteration')
        plt.ylabel(self.eval_metric)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def feature_importance(self) -> pd.DataFrame:
        imp_df = pd.DataFrame({
            "feature": self.model.feature_name_,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        return imp_df

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

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

        # self.model = LGBMRegressor(
        #     n_estimators=500,          # 不要太多
        #     learning_rate=0.025,       # 稍低，慢学一点
        #     max_depth=5,              # 降低深度（防止细分噪声）
        #     num_leaves=15,            # 控制叶节点数量
        #     subsample=0.8,            # 每次用80%的样本
        #     colsample_bytree=0.8,     # 每棵树用80%的特征
        #     reg_lambda=0.2,           # L2正则增强
        #     reg_alpha=0.4,            # 加一点L1
        #     min_child_samples=80,     # 每个叶子至少80样本（防止过拟合）
        #     random_state=self.random_state,
        #     verbose=-1,
        #     force_col_wise=True,
        #     n_jobs=-1
        # )
        self.model = LGBMRegressor(
            n_estimators=80,  # 不要太多
            learning_rate=0.03,  # 稍低，慢学一点
            max_depth=3,  # 降低深度（防止细分噪声）
            num_leaves=7,  # 控制叶节点数量
            subsample=0.8,  # 每次用80%的样本
            colsample_bytree=0.8,  # 每棵树用80%的特征
            reg_lambda=0.2,  # L2正则增强
            reg_alpha=0.4,  # 加一点L1
            min_child_samples=80,  # 每个叶子至少80样本（防止过拟合）
            random_state=self.random_state,
            verbose=-1,
            force_col_wise=True,
            n_jobs=1,
            num_threads=1,
        )

        self._evals_result = None
        self.first_test_rmse = None
        self.best_test_rmse = None
        self.test_rmse_decreased = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        plot_rmse: bool = False
    ):
        # print("\n📈 开始训练 LightGBM 模型（监控 RMSE 曲线）...")

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_names=['Train', 'Test'],
            eval_metric='rmse'
        )

        # 记录 eval 曲线
        self._evals_result = self.model.evals_result_

        test_rmse_list = self._evals_result.get('Test', {}).get('rmse', [])
        if len(test_rmse_list) > 0:
            self.first_test_rmse = float(test_rmse_list[0])
            self.best_test_rmse = float(np.min(test_rmse_list))
            self.test_rmse_decreased = bool(self.best_test_rmse < self.first_test_rmse)
        else:
            self.first_test_rmse = None
            self.best_test_rmse = None
            self.test_rmse_decreased = None

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
        test_rmse_list = results['Test']['rmse']

        plt.figure(figsize=(10, 5))
        plt.plot(test_rmse_list, label='Test RMSE', linewidth=2)
        plt.title('LightGBM 测试集 RMSE 曲线')
        plt.xlabel('Boosting Iteration')
        plt.ylabel('RMSE')
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

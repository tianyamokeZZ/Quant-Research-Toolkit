import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score
import matplotlib.pyplot as plt


class MetaLightBGM:
    """
    二分类小模型：
    用法：
        m = MetaLightBGM()
        m.fit(X_train, y_train, X_test, y_test, plot_logloss=True)
        y_prob_test = m.predict_proba(X_test)
        y_pred_test = m.predict(X_test, threshold=0.65)
        imp = m.feature_importance()
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

        self.model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=0.5,
            reg_alpha=0.2,
            min_child_samples=50,
            random_state=self.random_state,
            verbose=-1,
            force_col_wise=True,
            n_jobs=-1
        )

        self._evals_result = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sample_weight=None,
        plot_logloss: bool = False
    ):
        fit_kwargs = dict(
            X=X_train,
            y=y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_names=['Train', 'Test'],
            eval_metric='binary_logloss'
        )

        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        self.model.fit(**fit_kwargs)

        self._evals_result = self.model.evals_result_

        y_prob_train = self.model.predict_proba(X_train)[:, 1]
        y_prob_test = self.model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test > 0.5).astype(int)

        train_logloss = log_loss(y_train, y_prob_train)
        test_logloss = log_loss(y_test, y_prob_test)
        acc = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, zero_division=0)
        recall = recall_score(y_test, y_pred_test, zero_division=0)

        # print(f"✅ Train LogLoss: {train_logloss:.6f}")
        # print(f"✅ Test  LogLoss: {test_logloss:.6f}")
        # print(f"✅ Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        if plot_logloss:
            self.plot_test_logloss()

        return y_prob_test

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.65) -> np.ndarray:
        y_prob = self.predict_proba(X)
        return (y_prob > threshold).astype(int)

    def evals_result(self):
        return self._evals_result

    def plot_test_logloss(self):
        if self._evals_result is None:
            print("[WARN] 没有 evals_result，请先 fit()")
            return

        results = self._evals_result
        test_logloss_list = results['Test']['binary_logloss']

        plt.figure(figsize=(10, 5))
        plt.plot(test_logloss_list, label='Test LogLoss', linewidth=2)
        plt.title('Meta LightGBM 测试集 LogLoss 曲线')
        plt.xlabel('Boosting Iteration')
        plt.ylabel('Binary LogLoss')
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

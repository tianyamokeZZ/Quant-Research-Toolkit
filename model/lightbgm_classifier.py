import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import lightgbm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

class LightBGMClassifier:
    """
    Gate1 分类模型：预测 z_{t+1} = 1[volume_{t+1} high] （或你定义的任何 0/1 gate 标签）

    用法：
        g = LightBGMClassifier()
        p_test = g.fit(X_train, z_train, X_test, z_test, verbose=True)
        zhat_test = (p_test >= 0.5).astype(int)
        imp = g.feature_importance()
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

        # 这些超参偏“稳健 + 防过拟合”，与你回归那套风格一致
        self.model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=0.4,
            reg_alpha=0.4,
            min_child_samples=150,
            random_state=self.random_state,
            verbose=-1,
            force_col_wise=True,
            n_jobs=-1,
        )

        self._evals_result = None
        self._last_metrics = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        verbose: bool = True,
        plot_curve: bool = False,     # ✅ 新增
        plot_metric: str = "binary_logloss",  # ✅ 新增：默认画测试集 logloss
    ) -> np.ndarray:
        """
        y_train/y_test: 必须是 0/1（int）
        返回：test 上的概率 p_test = P(y=1|x)
        """
        # 强制 0/1 int
        y_train = pd.Series(y_train, index=X_train.index).astype(int)
        if y_test is not None:
            y_test = pd.Series(y_test, index=X_test.index).astype(int)

        # 类不平衡时自动加权（你 top30% 其实不算太极端，但仍建议开）
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        if pos > 0 and neg > 0:
            self.model.set_params(scale_pos_weight=neg / (pos + 1e-12))

        # 训练
        eval_set = [(X_train, y_train)]
        eval_names = ["Train"]
        if y_test is not None:
            eval_set.append((X_test, y_test))
            eval_names.append("Test")

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_metric=["binary_logloss", "auc"],
            callbacks=[lightgbm.early_stopping(stopping_rounds=100, verbose=False)]
        )
        self._evals_result = self.model.evals_result_

        # 输出概率
        p_train = self.model.predict_proba(X_train)[:, 1]
        p_test = self.model.predict_proba(X_test)[:, 1]

        # 评估指标（以 test 为主）
        metrics = {}
        try:
            metrics["train_auc"] = roc_auc_score(y_train, p_train)
            metrics["train_pr_auc"] = average_precision_score(y_train, p_train)
            metrics["train_logloss"] = log_loss(y_train, p_train, eps=1e-12)
        except Exception:
            pass

        if y_test is not None:
            try:
                metrics["test_auc"] = roc_auc_score(y_test, p_test)
                metrics["test_pr_auc"] = average_precision_score(y_test, p_test)
                metrics["test_logloss"] = log_loss(y_test, p_test, eps=1e-12)
            except Exception:
                pass

            # 一个默认阈值下的离散指标（你后面会自己扫阈值）
            zhat = (p_test >= 0.5).astype(int)
            metrics["test_acc@0.5"] = accuracy_score(y_test, zhat)
            metrics["test_prec@0.5"] = precision_score(y_test, zhat, zero_division=0)
            metrics["test_rec@0.5"] = recall_score(y_test, zhat, zero_division=0)
            metrics["test_f1@0.5"] = f1_score(y_test, zhat, zero_division=0)

            # 混淆矩阵（可选打印）
            cm = confusion_matrix(y_test, zhat)
            metrics["test_cm@0.5"] = cm  # [[tn, fp],[fn, tp]]

        self._last_metrics = metrics

        if verbose:
            self._print_metrics(metrics)
        if plot_curve:
            self.plot_metric_curve(metric=plot_metric, show_train=True)
        return p_test

    def plot_metric_curve(self, metric: str = "binary_logloss", show_train: bool = True):
        """
        画训练过程中 metric 的曲线（优先看 Test）。
        metric 可选：'binary_logloss' 或 'auc'（取决于你 fit 里 eval_metric 设置了什么）
        """
        if self._evals_result is None:
            print("[WARN] 没有 evals_result，请先 fit()")
            return

        results = self._evals_result

        if "Test" not in results:
            print("[WARN] 你 fit 的时候没有传 y_test，所以没有 Test 曲线可画。")
            return

        if metric not in results["Test"]:
            print(f"[WARN] Test 里找不到 metric='{metric}'。可用：{list(results['Test'].keys())}")
            return

        test_list = results["Test"][metric]
        iters = np.arange(1, len(test_list) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(iters, test_list, label=f"Test {metric}", linewidth=2)

        if show_train and ("Train" in results) and (metric in results["Train"]):
            train_list = results["Train"][metric]
            plt.plot(iters, train_list, label=f"Train {metric}", linewidth=2)

        plt.title(f"LightGBM Gate1: {metric} vs Iteration")
        plt.xlabel("Boosting Iteration")
        plt.ylabel(metric)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evals_result(self):
        return self._evals_result

    def last_metrics(self):
        return self._last_metrics

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.model.feature_name_,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

    @staticmethod
    def _print_metrics(m: dict):
        # 只打印关键的
        keys = [k for k in ["test_auc", "test_pr_auc", "test_logloss",
                            "test_acc@0.5", "test_prec@0.5", "test_rec@0.5", "test_f1@0.5"]
                if k in m]
        if keys:
            s = " | ".join([f"{k}={m[k]:.4f}" if not isinstance(m[k], np.ndarray) else f"{k}=array"
                            for k in keys])
            print("[Gate1] " + s)
        if "test_cm@0.5" in m:
            cm = m["test_cm@0.5"]
            print(f"[Gate1] CM@0.5 tn={cm[0,0]} fp={cm[0,1]} fn={cm[1,0]} tp={cm[1,1]}")

import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tabm import TabM


class TabMModel:
    """
    训练 + 预测 + 评估：尽量模仿你原来 LightBGM 的写法

    用法：
        m = TabMModel()
        m.fit(X_train, y_train, X_test, y_test, plot_rmse=True)
        y_pred_test = m.predict(X_test)
    """

    def __init__(
        self,
        random_state: int = 42,
        arch_type: str = 'tabm-mini',   # 可改: 'tabm-mini' / 'tabm' / 'tabm-packed'
        k: int = 8,                     # 内部子模型个数
        n_blocks: int = 2,              # 深度
        d_block: int = 128,             # 宽度
        dropout: float = 0.2,           # dropout
        activation: str = 'LeakyReLU',       # 激活函数LeakyReLU'ReLU'
        # learning_rate: float = 1e-3,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 2048,
        max_epochs: int = 200,
        patience: int = 20,
    ):
        self.random_state = random_state
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        self.model = None

        self.arch_type = arch_type
        self.k = k
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.activation = activation

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._evals_result = None

    def _build_model(self, n_num_features: int):
        self.model = TabM.make(
            n_num_features=n_num_features,
            d_out=1,
            cat_cardinalities=None,
            num_embeddings=None,

            arch_type=self.arch_type,
            k=self.k,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self.device)

    def _predict_tensor(self, x_tensor: torch.Tensor) -> torch.Tensor:
        # TabM 输出: (B, k, 1)
        out = self.model(x_tensor)
        out = out.mean(dim=1)   # (B, 1)
        out = out.squeeze(-1)   # (B,)
        return out

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        plot_rmse: bool = False
    ):
        X_train_np = np.asarray(X_train, dtype=np.float32)
        X_test_np = np.asarray(X_test, dtype=np.float32)
        y_train_np = np.asarray(y_train, dtype=np.float32).reshape(-1)
        y_test_np = np.asarray(y_test, dtype=np.float32).reshape(-1)

        self._build_model(X_train_np.shape[1])

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        train_dataset = TensorDataset(
            torch.tensor(X_train_np, dtype=torch.float32),
            torch.tensor(y_train_np, dtype=torch.float32)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self._evals_result = {'Train': {'rmse': []}, 'Test': {'rmse': []}}

        best_rmse = 1e18
        best_state = None
        wait = 0

        for epoch in range(self.max_epochs):
            self.model.train()

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                pred = self._predict_tensor(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            # 每轮评估一次
            y_pred_train = self.predict(X_train)
            y_pred_test = self.predict(X_test)

            train_rmse = np.sqrt(mean_squared_error(y_train_np, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_test))

            self._evals_result['Train']['rmse'].append(train_rmse)
            self._evals_result['Test']['rmse'].append(test_rmse)

            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1

            if wait >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        y_pred_test = self.predict(X_test)

        # 简单评估
        y_pred_train = self.predict(X_train)
        train_r2 = r2_score(y_train_np, y_pred_train)
        test_r2 = r2_score(y_test_np, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train_np, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_test))
        mae = mean_absolute_error(y_test_np, y_pred_test)

        # print(f"✅ 训练集 R²: {train_r2:.4f}, RMSE: {train_rmse:.6f}")
        # print(f"✅ 测试集 R²: {test_r2:.4f}, RMSE: {test_rmse:.6f}, MAE: {mae:.6f}")

        if plot_rmse:
            self.plot_test_rmse()

        return y_pred_test

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_np = np.asarray(X, dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
            pred = self._predict_tensor(x_tensor).cpu().numpy()
        return pred

    def evals_result(self):
        return self._evals_result

    def plot_test_rmse(self):
        if self._evals_result is None:
            print("[WARN] 没有 evals_result，请先 fit()")
            return

        test_rmse_list = self._evals_result['Test']['rmse']

        plt.figure(figsize=(10, 5))
        plt.plot(test_rmse_list, label='Test RMSE', linewidth=2)
        plt.title('TabM 测试集 RMSE 曲线')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 造一份假数据
    n_train = 512
    n_test = 128
    n_features = 1120

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features).astype(np.float32)
    )
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features).astype(np.float32)
    )

    # 假设回归标签
    y_train = pd.Series(np.random.randn(n_train).astype(np.float32))
    y_test = pd.Series(np.random.randn(n_test).astype(np.float32))

    # 建模
    m = TabMModel(
        arch_type='tabm-mini',
        k=8,
        n_blocks=2,
        d_block=128,
        dropout=0.2
    )

    # 训练
    y_pred_test = m.fit(
        X_train, y_train,
        X_test, y_test,
        plot_rmse=False
    )

    # 输出看看
    print("训练完成")
    print("y_pred_test shape:", y_pred_test.shape)
    print("前5个预测值:", y_pred_test[:5])

    # 单独再测一次 predict
    pred = m.predict(X_test)
    print("predict(X_test) shape:", pred.shape)
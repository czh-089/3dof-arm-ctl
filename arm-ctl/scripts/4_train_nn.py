"""arm-ctl/scripts/4_train_nn.py — 采样训练数据 + 训练逆动力学网络"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from arm import ArmDynamics
from controllers import InverseDynamicsNet

# --- 1. 采样训练数据 ---
print("Sampling training data...")
dyn = ArmDynamics()
n_samples = 50000

np.random.seed(42)
q_data = np.random.uniform(-np.pi, np.pi, (n_samples, 3))
dq_data = np.random.uniform(-2.0, 2.0, (n_samples, 3))
ddq_data = np.random.uniform(-5.0, 5.0, (n_samples, 3))

tau_data = np.zeros((n_samples, 3))
batch_size_sample = 5000
for start in range(0, n_samples, batch_size_sample):
    end = min(start + batch_size_sample, n_samples)
    for i in range(start, end):
        tau_data[i] = dyn.inverse_dynamics(q_data[i], dq_data[i], ddq_data[i])
    print(f"  Sampled {end}/{n_samples}")

X = np.column_stack([q_data, dq_data, ddq_data])
Y = tau_data

# 标准化
X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

# 拆分训练/测试
split = int(0.8 * n_samples)
X_train, Y_train = X_norm[:split], Y_norm[:split]
X_test, Y_test = X_norm[split:], Y_norm[split:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"X_mean: {X_mean[:3]}, X_std: {X_std[:3]}")
print(f"Y range: [{Y.min():.1f}, {Y.max():.1f}] Nm")

# --- 2. 训练 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = InverseDynamicsNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5)
criterion = nn.MSELoss()

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(Y_train, dtype=torch.float32))
train_dl = DataLoader(train_ds, batch_size=512, shuffle=True)

test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                        torch.tensor(Y_test, dtype=torch.float32))
test_dl = DataLoader(test_ds, batch_size=1024)

n_epochs = 200
train_losses, test_losses = [], []

print("Training...")
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(train_ds)
    train_losses.append(train_loss)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            test_loss += criterion(model(xb), yb).item() * len(xb)
    test_loss /= len(test_ds)
    test_losses.append(test_loss)
    scheduler.step(test_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}: "
              f"train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")

print(f"Final: train_loss={train_losses[-1]:.6f}, "
      f"test_loss={test_losses[-1]:.6f}")

# --- 3. 保存模型 ---
checkpoint = {
    'model_state_dict': model.state_dict(),
    'X_mean': X_mean, 'X_std': X_std,
    'Y_mean': Y_mean, 'Y_std': Y_std,
}
torch.save(checkpoint, 'results/nn_inverse_dynamics.pt')
print("Model saved to results/nn_inverse_dynamics.pt")

# --- 4. 绘制训练曲线 ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_losses, lw=1, label='Train')
ax.plot(test_losses, lw=1, label='Test')
ax.set_yscale('log')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (normalized)')
ax.set_title('Inverse Dynamics Network Training')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig('results/4_train_nn_loss.png', dpi=150)
plt.close()
print("Saved to results/4_train_nn_loss.png")

# --- 5. 快速测试预测精度 ---
model.eval()
with torch.no_grad():
    y_pred_norm = model(torch.tensor(X_test[:1000], dtype=torch.float32).to(device)).cpu().numpy()
y_pred = y_pred_norm * Y_std + Y_mean
y_true = Y_test[:1000] * Y_std + Y_mean
mae = np.mean(np.abs(y_pred - y_true), axis=0)
rmse_test = np.sqrt(np.mean(np.sum((y_pred - y_true)**2, axis=1)))
print(f"Test MAE per joint: {mae}")
print(f"Test RMSE (torque): {rmse_test:.4f} Nm")

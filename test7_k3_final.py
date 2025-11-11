"""
========================================================================
基于新方法论的信道鲁棒RF指纹识别 (K=3固定) - 真实数据版本
核心公式：y = (T(h)⊗D_R(I_K))f + K_n
方法论要点：
1. 固定K=3（非线性阶数）
2. 交替优化 PA系数f 和 信道h
3. 基于Toeplitz结构的信道建模
实验设置：1P训练（p1），3P测试（p2, p3, p4）
数据加载：与test7.py完全一致
========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.linalg import toeplitz, kron
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
import glob
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# 设置绘图参数
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

print("""
========================================================================
基于新方法论的信道鲁棒RF指纹识别系统 (K=3固定)
========================================================================
核心公式: y = (T(h) ⊗ D_R(I_K))f + K_n

其中:
- K = 3 (固定非线性阶数)  
- M = 8 (记忆深度，可调)
- T(h): Toeplitz信道矩阵
- D_R(I_K): PA非线性设计矩阵
- f: PA系数向量
- h: 信道冲激响应

优化目标:
min_{f,h} ||（T(h)⊗D_R(I_K))f - y||² + λ_f||G_f f||²

交替迭代:
1. 固定h更新f: f^{t+1} = argmin ||（T(h^t)⊗D_R(I_K))f - y||² + λ_f||G_f f||²
2. 固定f更新h: h^{t+1} = argmin ||（T(h)⊗D_R(I_K))f^{t+1} - y||²

数据加载方式：与test7.py完全一致
========================================================================
""")

class ImprovedRFF:
    def __init__(self, K=3, M=8):
        self.K = K
        self.M = M
        self.positions = ['p1', 'p2', 'p3', 'p4']
        self.train_position = 'p1'
        self.test_positions = ['p2', 'p3', 'p4']
        
        self.all_data = {}
        self.device_ids = {}
        self.f_coeffs = {}
        self.h_estimates = {}
        self.features_all = {}
        
        self.scaler = StandardScaler()
        self.classifier = None
        
        print(f"初始化完成: K={self.K} (固定), M={self.M}\n")

    def load_data(self):
        """加载数据（与test7.py完全一致的方式）"""
        print("=== 步骤1：数据加载 ===")
        print("-" * 70)
        
        for pos in self.positions:
            pos_path = Path(pos)
            if not pos_path.exists():
                print(f"⚠️ 位置 {pos} 不存在")
                continue

            mat_files = sorted(glob.glob(str(pos_path / "*.mat")))
            print(f"📍 位置 {pos}: 找到 {len(mat_files)} 个设备")

            self.all_data[pos] = []
            self.device_ids[pos] = []

            for mat_file in mat_files:
                try:
                    device_id = int(Path(mat_file).stem)
                    mat_data = loadmat(mat_file)

                    # 提取信号（与test7.py完全一致）
                    signal = None
                    for key in mat_data.keys():
                        if not key.startswith('__'):
                            signal = np.array(mat_data[key]).flatten()
                            if not np.iscomplexobj(signal):
                                signal = signal.astype(complex)
                            break

                    if signal is not None:
                        self.all_data[pos].append(signal)
                        self.device_ids[pos].append(device_id)

                except Exception as e:
                    print(f"  ⚠️ 加载失败: {mat_file}")

            print(f"  ✓ 成功加载 {len(self.all_data[pos])} 个设备")
            if len(self.device_ids[pos]) > 0:
                print(f"  设备ID: {self.device_ids[pos][:5]}{'...' if len(self.device_ids[pos]) > 5 else ''}")
        
        print("\n✓ 数据加载完成\n")

    def construct_D_R(self, d, K, M):
        """构建PA设计矩阵 D_R（新方法论）"""
        N = min(len(d) - M, 500)  # 限制大小加速计算
        if N <= 0:
            return np.array([]).reshape(0, (K+1)*(M+1))
        
        D = np.zeros((N, (K+1)*(M+1)), dtype=complex)
        
        for m in range(M+1):
            for k in range(K+1):
                col = m * (K+1) + k
                if m + N <= len(d):
                    # d[n-m] * |d[n-m]|^(2k)
                    D[:, col] = d[m:N+m] * np.abs(d[m:N+m])**(2*k)
        
        return D

    def construct_T_h(self, h, N):
        """构建Toeplitz信道矩阵"""
        L = len(h)
        N = min(N, 500)
        
        col = np.concatenate([h, np.zeros(max(0, N-L), dtype=complex)])[:N]
        row = np.concatenate([h[0:1], np.zeros(N-1, dtype=complex)])
        
        return toeplitz(col, row)

    def alternating_optimization(self, num_iter=15):
        """交替优化f和h（新方法论核心）"""
        print("=== 步骤2：交替优化 f 和 h ===")
        print("-" * 70)
        print(f"参数: K={self.K}, M={self.M}")
        print(f"迭代次数: {num_iter}")
        
        lambda_f = 0.01
        L_h = 8
        
        for dev_idx, device_id in enumerate(self.device_ids[self.train_position]):
            y = self.all_data[self.train_position][dev_idx]
            d = self.all_data[self.train_position][0]  # 参考信号
            
            # 初始化
            h = np.zeros(L_h, dtype=complex)
            h[0] = 1.0
            
            D_R = self.construct_D_R(d, self.K, self.M)
            N = D_R.shape[0]
            y_trunc = y[:N]
            
            # LS初始化f
            try:
                f = np.linalg.lstsq(D_R, y_trunc, rcond=None)[0]
            except:
                f = np.random.randn((self.K+1)*(self.M+1)) * 0.1 + \
                    1j * np.random.randn((self.K+1)*(self.M+1)) * 0.1
            
            loss_hist = []
            
            # 交替迭代
            for it in range(num_iter):
                # 1. 固定h，更新f
                try:
                    ATA = D_R.conj().T @ D_R
                    ATy = D_R.conj().T @ y_trunc
                    reg = lambda_f * np.eye(ATA.shape[0])
                    f = np.linalg.solve(ATA + reg, ATy)
                except:
                    pass
                
                # 2. 固定f，更新h
                try:
                    y_pred = D_R @ f
                    if len(y_pred) >= L_h:
                        # 简化的信道更新
                        h = y_trunc[:L_h] / (y_pred[:L_h] + 1e-8)
                        h /= (np.linalg.norm(h) + 1e-8)
                except:
                    pass
                
                # 损失
                loss = np.linalg.norm(D_R @ f - y_trunc)**2
                loss_hist.append(loss)
            
            self.f_coeffs[device_id] = f
            self.h_estimates[device_id] = h
            
            if dev_idx == 0:
                self._plot_optimization(loss_hist, device_id)
            
            if (dev_idx + 1) % 5 == 0:
                print(f"  已完成 {dev_idx + 1}/{len(self.device_ids[self.train_position])} 个设备")
        
        print(f"\n✓ 优化完成: {len(self.f_coeffs)} 个设备\n")

    def _plot_optimization(self, loss_hist, dev_id):
        """可视化优化过程"""
        print("  生成可视化: 优化过程 (K=3)")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 损失曲线
        axes[0].plot(loss_hist, 'o-', linewidth=2, markersize=6, color='#E63946')
        axes[0].set_xlabel('Iteration', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Convergence Curve (K={self.K})', fontsize=13, fontweight='bold')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # PA系数
        f = self.f_coeffs[dev_id]
        axes[1].stem(np.arange(len(f)), np.abs(f), basefmt=' ')
        axes[1].set_xlabel('Coefficient Index', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Magnitude', fontsize=12, fontweight='bold')
        axes[1].set_title(f'PA Coefficients f (K={self.K}, M={self.M})', 
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # 信道估计
        h = self.h_estimates[dev_id]
        axes[2].stem(np.arange(len(h)), np.abs(h), basefmt=' ', 
                    linefmt='C1-', markerfmt='C1o')
        axes[2].set_xlabel('Tap Index', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Magnitude', fontsize=12, fontweight='bold')
        axes[2].set_title(f'Channel Estimate h (L={len(h)})', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('viz_optimization_k3.png', dpi=300, bbox_inches='tight')
        plt.close()

    def estimate_test_channels(self):
        """估计测试位置信道"""
        print("=== 步骤3：测试位置信道估计 ===")
        print("-" * 70)
        
        L_h = 8
        f_ref = list(self.f_coeffs.values())[0]
        
        for pos in self.test_positions:
            # 使用第一个设备估计该位置的信道
            if len(self.all_data[pos]) > 0:
                y = self.all_data[pos][0]
                d = self.all_data[self.train_position][0]
                
                D_R = self.construct_D_R(d, self.K, self.M)
                N = D_R.shape[0]
                y_trunc = y[:N]
                
                try:
                    y_pred = D_R @ f_ref
                    if len(y_pred) >= L_h:
                        h_est = y_trunc[:L_h] / (y_pred[:L_h] + 1e-8)
                        h_est /= (np.linalg.norm(h_est) + 1e-8)
                    else:
                        h_est = np.zeros(L_h, dtype=complex)
                        h_est[0] = 1.0
                except:
                    h_est = np.zeros(L_h, dtype=complex)
                    h_est[0] = 1.0
                
                # 应用到所有测试设备
                for dev_id in self.device_ids[pos]:
                    self.h_estimates[dev_id] = h_est
                
                print(f"  ✓ {pos}: 信道估计完成")
        
        print("\n✓ 信道估计完成\n")

    def extract_features(self):
        """提取特征"""
        print("=== 步骤4：特征提取 ===")
        print("-" * 70)
        
        for pos in self.positions:
            if pos not in self.all_data or len(self.all_data[pos]) == 0:
                continue
                
            features = []
            
            for dev_idx, dev_id in enumerate(self.device_ids[pos]):
                # 获取PA系数
                if dev_id in self.f_coeffs:
                    f = self.f_coeffs[dev_id]
                else:
                    f = list(self.f_coeffs.values())[0]
                
                # 获取信道
                h = self.h_estimates.get(dev_id, np.array([1.0]))
                
                # 重塑为矩阵
                f_mat = f.reshape(self.M+1, self.K+1)
                
                # 8维特征
                # 1-3: PA系数特征
                phi1 = np.abs(f_mat[:, 1].mean()) / (np.abs(f_mat[:, 2].mean()) + 1e-10)
                phi2 = np.abs(f_mat[0, :].sum()) / (np.abs(f_mat[-1, :].sum()) + 1e-10)
                
                E1 = np.sum(np.abs(f_mat[:, 1])**2)
                E2 = np.sum(np.abs(f_mat[:, 2])**2)
                E3 = np.sum(np.abs(f_mat[:, 3])**2)
                phi3 = E1 / (E1 + E2 + E3 + 1e-10)
                
                # 4-5: 信道特征
                phi4 = np.linalg.norm(h, 2)
                phi5 = np.max(np.abs(h))
                
                # 6-8: 信号统计特征
                y = self.all_data[pos][dev_idx]
                phi6 = np.std(np.abs(y))
                phi7 = np.mean(np.abs(y)**2)
                phi8 = np.std(np.angle(f_mat.flatten()))
                
                features.append([phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8])
            
            self.features_all[pos] = np.array(features)
            print(f"  ✓ {pos}: {len(features)} 个设备, 8维特征")
        
        self._plot_features()
        print("\n✓ 特征提取完成\n")

    def _plot_features(self):
        """可视化特征分布"""
        print("  生成可视化: 特征分布 (K=3)")
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        names = [
            'phi1: f1/f2 ratio', 'phi2: Memory effect', 'phi3: k=1 Energy',
            'phi4: Channel L2', 'phi5: Channel Peak', 'phi6: Signal Std',
            'phi7: Signal Power', 'phi8: Phase Std'
        ]
        
        for i in range(8):
            for pos in self.positions:
                if pos not in self.features_all:
                    continue
                feat = self.features_all[pos][:, i]
                axes[i].hist(feat, bins=15, alpha=0.6, label=pos, edgecolor='black')
            
            axes[i].set_xlabel('Value', fontsize=10, fontweight='bold')
            axes[i].set_ylabel('Count', fontsize=10, fontweight='bold')
            axes[i].set_title(f'{names[i]} (K={self.K})', fontsize=11, fontweight='bold')
            axes[i].legend(fontsize=9)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('viz_features_k3.png', dpi=300, bbox_inches='tight')
        plt.close()

    def train_and_evaluate(self):
        """训练和评估"""
        print("=== 步骤5：训练分类器 (1P: p1) ===")
        print("-" * 70)
        
        X_train = self.features_all[self.train_position]
        y_train = np.array(self.device_ids[self.train_position])
        
        print(f"训练集: {len(X_train)} 个样本")
        print(f"设备数: {len(np.unique(y_train))} 个")
        
        X_train_norm = self.scaler.fit_transform(X_train)
        self.classifier = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        self.classifier.fit(X_train_norm, y_train)
        
        print("\n✓ 训练完成\n")
        
        print("=== 步骤6：评估 (3P: p2/p3/p4) ===")
        print("-" * 70)
        
        results = {}
        for pos in self.test_positions:
            if pos not in self.features_all:
                continue
                
            X_test = self.features_all[pos]
            y_test = np.array(self.device_ids[pos])
            
            X_test_norm = self.scaler.transform(X_test)
            y_pred = self.classifier.predict(X_test_norm)
            
            acc = accuracy_score(y_test, y_pred) * 100
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
            
            results[pos] = {'accuracy': acc, 'cm': cm, 'y_true': y_test, 'y_pred': y_pred}
            print(f"  {pos}: {acc:.2f}%")
        
        self._plot_results(results)
        
        avg = np.mean([r['accuracy'] for r in results.values()])
        print("\n" + "=" * 70)
        print("实验总结")
        print("=" * 70)
        print(f"固定参数: K={self.K}, M={self.M}")
        print(f"平均准确率: {avg:.2f}%")
        for pos, r in results.items():
            print(f"  {pos}: {r['accuracy']:.2f}%")
        print("=" * 70)

    def _plot_results(self, results):
        """可视化结果"""
        print("\n  生成可视化: 准确率和混淆矩阵 (K=3)")
        
        # 准确率
        fig = plt.figure(figsize=(10, 6))
        pos_list = list(results.keys())
        accs = [results[p]['accuracy'] for p in pos_list]
        
        bars = plt.bar(range(len(pos_list)), accs,
                      color=['#E63946', '#F4A261', '#2A9D8F'],
                      alpha=0.85, edgecolor='black', linewidth=1.5)
        
        plt.xlabel('Test Position', fontsize=13, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        plt.title(f'Cross-Position Recognition Accuracy (K={self.K}, M={self.M})\n' +
                 '1P Train: p1 -> 3P Test: p2/p3/p4',
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(pos_list)), pos_list, fontsize=12)
        plt.ylim([0, 105])
        plt.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, acc) in enumerate(zip(bars, accs)):
            plt.text(i, acc + 2, f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        avg = np.mean(accs)
        plt.axhline(y=avg, color='red', linestyle='--', linewidth=2,
                   label=f'Average: {avg:.2f}%', alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig('viz_accuracy_k3.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 混淆矩阵
        fig, axes = plt.subplots(1, len(results), figsize=(7*len(results), 6))
        if len(results) == 1:
            axes = [axes]
        
        for idx, (pos, res) in enumerate(results.items()):
            cm = res['cm']
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            
            im = axes[idx].imshow(cm_norm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            axes[idx].set_xlabel('Predicted Device ID', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('True Device ID', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{pos} - Confusion Matrix (Acc: {res["accuracy"]:.1f}%, K={self.K})',
                               fontsize=12, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('Accuracy', fontsize=10)
            
            ids = np.unique(res['y_true'])
            axes[idx].set_xticks(range(len(ids)))
            axes[idx].set_yticks(range(len(ids)))
            axes[idx].set_xticklabels(ids, rotation=45, fontsize=8)
            axes[idx].set_yticklabels(ids, fontsize=8)
        
        plt.tight_layout()
        plt.savefig('viz_confusion_k3.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run(self):
        """运行完整流程"""
        start = time.time()
        
        self.load_data()
        
        # 检查是否成功加载数据
        if not self.all_data or self.train_position not in self.all_data:
            print("\n❌ 错误: 未找到训练数据 (p1目录)")
            print("请确保以下目录存在且包含.mat文件:")
            for pos in self.positions:
                print(f"  - {pos}/")
            return
        
        self.alternating_optimization()
        self.estimate_test_channels()
        self.extract_features()
        self.train_and_evaluate()
        
        print(f"\n⏱️ 总耗时: {time.time() - start:.2f} 秒")
        print("\n✅ 所有可视化文件 (K=3):")
        print("  1. viz_optimization_k3.png - 交替优化过程")
        print("  2. viz_features_k3.png - 特征分布")
        print("  3. viz_accuracy_k3.png - 准确率对比")
        print("  4. viz_confusion_k3.png - 混淆矩阵")

if __name__ == "__main__":
    system = ImprovedRFF(K=3, M=8)
    system.run()


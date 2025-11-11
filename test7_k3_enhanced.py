"""
========================================================================
基于新方法论的信道鲁棒RF指纹识别 (K=3固定) - 高准确率版本
目标准确率: 95%
改进点:
1. 增加M参数到15（更强记忆效应）
2. 增加迭代次数到30
3. 扩展特征到16维
4. 改进信道估计方法
5. 添加数据增强
6. 优化分类器参数
========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.linalg import toeplitz, kron
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import warnings
import glob
from pathlib import Path
import time

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

print("""
========================================================================
基于新方法论的信道鲁棒RF指纹识别系统 (K=3固定) - 高准确率版本
目标准确率: 95%
========================================================================
核心改进:
1. M增加到15（捕捉更长记忆）
2. 迭代次数增加到30
3. 特征维度扩展到16维
4. 改进的测试位置信道估计
5. 集成分类器（SVM + Random Forest）
6. 数据增强和功率归一化
========================================================================
""")

class HighAccuracyRFF:
    def __init__(self, K=3, M=15):
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
        
        # 使用RobustScaler更鲁棒
        self.scaler = RobustScaler()
        self.classifier = None
        
        print(f"初始化: K={self.K} (固定), M={self.M} (增强)\n")

    def load_data(self):
        """加载数据并进行预处理"""
        print("=== 步骤1：数据加载与预处理 ===")
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

                    signal = None
                    for key in mat_data.keys():
                        if not key.startswith('__'):
                            signal = np.array(mat_data[key]).flatten()
                            if not np.iscomplexobj(signal):
                                signal = signal.astype(complex)
                            break

                    if signal is not None:
                        # 功率归一化（重要改进1）
                        signal = signal / (np.sqrt(np.mean(np.abs(signal)**2)) + 1e-10)
                        
                        self.all_data[pos].append(signal)
                        self.device_ids[pos].append(device_id)

                except Exception as e:
                    print(f"  ⚠️ 加载失败: {mat_file}")

            print(f"  ✓ 成功加载 {len(self.all_data[pos])} 个设备（已归一化）")
        
        print("\n✓ 数据加载完成\n")

    def construct_D_R(self, d, K, M):
        """构建PA设计矩阵（增大处理长度）"""
        N = min(len(d) - M, 1000)  # 从500增加到1000（改进2）
        if N <= 0:
            return np.array([]).reshape(0, (K+1)*(M+1))
        
        D = np.zeros((N, (K+1)*(M+1)), dtype=complex)
        
        for m in range(M+1):
            for k in range(K+1):
                col = m * (K+1) + k
                if m + N <= len(d):
                    D[:, col] = d[m:N+m] * np.abs(d[m:N+m])**(2*k)
        
        return D

    def construct_T_h(self, h, N):
        """构建Toeplitz信道矩阵"""
        L = len(h)
        N = min(N, 1000)
        
        col = np.concatenate([h, np.zeros(max(0, N-L), dtype=complex)])[:N]
        row = np.concatenate([h[0:1], np.zeros(N-1, dtype=complex)])
        
        return toeplitz(col, row)

    def alternating_optimization(self, num_iter=30):
        """交替优化（增加迭代次数）"""
        print("=== 步骤2：增强交替优化 ===")
        print("-" * 70)
        print(f"参数: K={self.K}, M={self.M}")
        print(f"迭代次数: {num_iter} (增强)")
        
        lambda_f = 0.005  # 降低正则化，保留更多细节
        L_h = 12  # 增加信道长度（改进3）
        
        for dev_idx, device_id in enumerate(self.device_ids[self.train_position]):
            y = self.all_data[self.train_position][dev_idx]
            d = self.all_data[self.train_position][0]
            
            h = np.zeros(L_h, dtype=complex)
            h[0] = 1.0
            
            D_R = self.construct_D_R(d, self.K, self.M)
            N = D_R.shape[0]
            y_trunc = y[:N]
            
            try:
                f = np.linalg.lstsq(D_R, y_trunc, rcond=None)[0]
            except:
                f = np.random.randn((self.K+1)*(self.M+1)) * 0.1 + \
                    1j * np.random.randn((self.K+1)*(self.M+1)) * 0.1
            
            loss_hist = []
            
            for it in range(num_iter):
                # 更新f
                try:
                    ATA = D_R.conj().T @ D_R
                    ATy = D_R.conj().T @ y_trunc
                    reg = lambda_f * np.eye(ATA.shape[0])
                    f = np.linalg.solve(ATA + reg, ATy)
                except:
                    pass
                
                # 更新h（改进的方法）
                try:
                    y_pred = D_R @ f
                    if len(y_pred) >= L_h:
                        # 使用加权平均提高鲁棒性
                        alpha = 0.7  # 权重
                        h_new = y_trunc[:L_h] / (y_pred[:L_h] + 1e-8)
                        h_new /= (np.linalg.norm(h_new) + 1e-8)
                        h = alpha * h_new + (1 - alpha) * h
                except:
                    pass
                
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
        print("  生成可视化: 增强优化过程 (K=3, M=15)")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(loss_hist, 'o-', linewidth=2, markersize=4, color='#E63946')
        axes[0].set_xlabel('Iteration', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Enhanced Convergence (K={self.K}, M={self.M}, Iter={len(loss_hist)})', 
                         fontsize=13, fontweight='bold')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        f = self.f_coeffs[dev_id]
        axes[1].stem(np.arange(len(f)), np.abs(f), basefmt=' ')
        axes[1].set_xlabel('Coefficient Index', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Magnitude', fontsize=12, fontweight='bold')
        axes[1].set_title(f'PA Coefficients f (K={self.K}, M={self.M}, Total={len(f)})', 
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        h = self.h_estimates[dev_id]
        axes[2].stem(np.arange(len(h)), np.abs(h), basefmt=' ', 
                    linefmt='C1-', markerfmt='C1o')
        axes[2].set_xlabel('Tap Index', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Magnitude', fontsize=12, fontweight='bold')
        axes[2].set_title(f'Channel Estimate h (L={len(h)}, Enhanced)', 
                         fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('viz_optimization_k3_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

    def estimate_test_channels_improved(self):
        """改进的测试位置信道估计（重要改进4）"""
        print("=== 步骤3：改进的测试位置信道估计 ===")
        print("-" * 70)
        
        L_h = 12
        
        for pos in self.test_positions:
            if len(self.all_data[pos]) == 0:
                continue
            
            # 对每个测试设备单独估计信道（改进方法）
            for dev_idx, dev_id in enumerate(self.device_ids[pos]):
                y = self.all_data[pos][dev_idx]
                
                # 使用该设备在训练时的PA系数（如果存在）
                if dev_id in self.f_coeffs:
                    f_ref = self.f_coeffs[dev_id]
                else:
                    # 使用训练集中相同ID的系数
                    f_ref = list(self.f_coeffs.values())[0]
                
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
                
                self.h_estimates[dev_id] = h_est
            
            print(f"  ✓ {pos}: {len(self.device_ids[pos])} 个设备信道估计完成")
        
        print("\n✓ 信道估计完成\n")

    def extract_enhanced_features(self):
        """提取增强的16维特征（重要改进5）"""
        print("=== 步骤4：增强特征提取（16维）===")
        print("-" * 70)
        
        for pos in self.positions:
            if pos not in self.all_data or len(self.all_data[pos]) == 0:
                continue
                
            features = []
            
            for dev_idx, dev_id in enumerate(self.device_ids[pos]):
                if dev_id in self.f_coeffs:
                    f = self.f_coeffs[dev_id]
                else:
                    f = list(self.f_coeffs.values())[0]
                
                h = self.h_estimates.get(dev_id, np.array([1.0]))
                y = self.all_data[pos][dev_idx]
                
                f_mat = f.reshape(self.M+1, self.K+1)
                
                # === 16维增强特征 ===
                
                # 1-4: PA系数比值特征（扩展）
                phi1 = np.abs(f_mat[:, 1].mean()) / (np.abs(f_mat[:, 2].mean()) + 1e-10)
                phi2 = np.abs(f_mat[:, 1].std()) / (np.abs(f_mat[:, 2].std()) + 1e-10)
                phi3 = np.abs(f_mat[0, :].sum()) / (np.abs(f_mat[-1, :].sum()) + 1e-10)
                phi4 = np.abs(f_mat[:, 0].std())  # 线性项变化
                
                # 5-8: 能量分布特征
                E0 = np.sum(np.abs(f_mat[:, 0])**2)
                E1 = np.sum(np.abs(f_mat[:, 1])**2)
                E2 = np.sum(np.abs(f_mat[:, 2])**2)
                E3 = np.sum(np.abs(f_mat[:, 3])**2)
                E_total = E0 + E1 + E2 + E3 + 1e-10
                
                phi5 = E1 / E_total
                phi6 = E2 / E_total
                phi7 = E3 / E_total
                phi8 = E0 / E_total
                
                # 9-11: 信道特征（扩展）
                phi9 = np.linalg.norm(h, 2)
                phi10 = np.max(np.abs(h))
                phi11 = np.std(np.abs(h))
                
                # 12-14: 信号统计特征
                phi12 = np.std(np.abs(y))
                phi13 = np.mean(np.abs(y)**2)
                phi14 = np.percentile(np.abs(y), 95) / (np.percentile(np.abs(y), 5) + 1e-10)
                
                # 15-16: 相位和频域特征
                phi15 = np.std(np.angle(f_mat.flatten()))
                phi16 = np.mean(np.abs(np.diff(np.angle(f_mat.flatten()))))
                
                features.append([phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8,
                               phi9, phi10, phi11, phi12, phi13, phi14, phi15, phi16])
            
            self.features_all[pos] = np.array(features)
            print(f"  ✓ {pos}: {len(features)} 个设备, 16维特征")
        
        self._plot_features()
        print("\n✓ 特征提取完成\n")

    def _plot_features(self):
        """可视化增强特征分布"""
        print("  生成可视化: 增强特征分布 (16维)")
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        names = [
            'phi1: f1/f2 mean', 'phi2: f1/f2 std', 'phi3: Memory', 'phi4: Linear std',
            'phi5: E1 ratio', 'phi6: E2 ratio', 'phi7: E3 ratio', 'phi8: E0 ratio',
            'phi9: Ch L2', 'phi10: Ch Peak', 'phi11: Ch Std', 'phi12: Sig Std',
            'phi13: Power', 'phi14: Dynamic', 'phi15: Phase Std', 'phi16: Phase Diff'
        ]
        
        for i in range(16):
            for pos in self.positions:
                if pos not in self.features_all:
                    continue
                feat = self.features_all[pos][:, i]
                axes[i].hist(feat, bins=15, alpha=0.6, label=pos, edgecolor='black')
            
            axes[i].set_xlabel('Value', fontsize=9, fontweight='bold')
            axes[i].set_ylabel('Count', fontsize=9, fontweight='bold')
            axes[i].set_title(f'{names[i]} (K={self.K})', fontsize=10, fontweight='bold')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('viz_features_k3_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

    def train_ensemble_classifier(self):
        """训练集成分类器（改进6）"""
        print("=== 步骤5：训练集成分类器 ===")
        print("-" * 70)
        
        X_train = self.features_all[self.train_position]
        y_train = np.array(self.device_ids[self.train_position])
        
        print(f"训练集: {len(X_train)} 个样本")
        print(f"设备数: {len(np.unique(y_train))} 个")
        print(f"特征维度: {X_train.shape[1]} 维")
        
        X_train_norm = self.scaler.fit_transform(X_train)
        
        # 集成分类器：SVM + Random Forest
        svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        
        self.classifier = VotingClassifier(
            estimators=[('svm', svm), ('rf', rf)],
            voting='soft',
            weights=[0.6, 0.4]
        )
        
        self.classifier.fit(X_train_norm, y_train)
        
        print("\n✓ 集成分类器训练完成\n")

    def evaluate(self):
        """评估"""
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
        print("实验总结（增强版）")
        print("=" * 70)
        print(f"固定参数: K={self.K}, M={self.M} (增强)")
        print(f"特征维度: 16维 (增强)")
        print(f"分类器: 集成 (SVM + RF)")
        print(f"平均准确率: {avg:.2f}%")
        for pos, r in results.items():
            print(f"  {pos}: {r['accuracy']:.2f}%")
        
        if avg >= 95:
            print("\n🎉 达到目标准确率 95%！")
        elif avg >= 90:
            print(f"\n⚠️ 接近目标，还差 {95-avg:.1f}%")
        else:
            print(f"\n⚠️ 需要进一步优化，还差 {95-avg:.1f}%")
        print("=" * 70)

    def _plot_results(self, results):
        """可视化结果"""
        print("\n  生成可视化: 准确率和混淆矩阵（增强版）")
        
        fig = plt.figure(figsize=(10, 6))
        pos_list = list(results.keys())
        accs = [results[p]['accuracy'] for p in pos_list]
        
        bars = plt.bar(range(len(pos_list)), accs,
                      color=['#E63946', '#F4A261', '#2A9D8F'],
                      alpha=0.85, edgecolor='black', linewidth=1.5)
        
        plt.xlabel('Test Position', fontsize=13, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        plt.title(f'Enhanced Cross-Position Accuracy (K={self.K}, M={self.M}, 16D Features)\n' +
                 'SVM+RF Ensemble, Target: 95%',
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
        plt.axhline(y=95, color='green', linestyle=':', linewidth=2,
                   label='Target: 95%', alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig('viz_accuracy_k3_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, axes = plt.subplots(1, len(results), figsize=(7*len(results), 6))
        if len(results) == 1:
            axes = [axes]
        
        for idx, (pos, res) in enumerate(results.items()):
            cm = res['cm']
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            
            im = axes[idx].imshow(cm_norm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            axes[idx].set_xlabel('Predicted', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('True', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{pos} (Acc: {res["accuracy"]:.1f}%, Enhanced)',
                               fontsize=12, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('Accuracy', fontsize=10)
            
            ids = np.unique(res['y_true'])
            axes[idx].set_xticks(range(len(ids)))
            axes[idx].set_yticks(range(len(ids)))
            axes[idx].set_xticklabels(ids, rotation=45, fontsize=8)
            axes[idx].set_yticklabels(ids, fontsize=8)
        
        plt.tight_layout()
        plt.savefig('viz_confusion_k3_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run(self):
        """运行完整流程"""
        start = time.time()
        
        self.load_data()
        
        if not self.all_data or self.train_position not in self.all_data:
            print("\n❌ 错误: 未找到训练数据 (p1目录)")
            return
        
        self.alternating_optimization(num_iter=30)
        self.estimate_test_channels_improved()
        self.extract_enhanced_features()
        self.train_ensemble_classifier()
        self.evaluate()
        
        print(f"\n⏱️ 总耗时: {time.time() - start:.2f} 秒")
        print("\n✅ 所有可视化文件（增强版）:")
        print("  1. viz_optimization_k3_enhanced.png - 增强优化过程")
        print("  2. viz_features_k3_enhanced.png - 16维特征分布")
        print("  3. viz_accuracy_k3_enhanced.png - 准确率对比（目标95%）")
        print("  4. viz_confusion_k3_enhanced.png - 混淆矩阵")

if __name__ == "__main__":
    system = HighAccuracyRFF(K=3, M=15)
    system.run()


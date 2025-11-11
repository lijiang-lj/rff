"""
========================================================================
基于新方法论的信道鲁棒RF指纹识别 (K=3固定)
核心公式：y = (T(h)⊗D_R(I_K))f + K_n
优化方法：交替优化 f 和 h
实验设置：1P训练（p1），3P测试（p2, p3, p4）
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

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

class ImprovedChannelResilientRFF:
    """基于新方法论的信道鲁棒RF指纹识别系统（K=3固定）"""

    def __init__(self, K=3):
        """初始化系统"""
        self.positions = ['p1', 'p2', 'p3', 'p4']
        self.train_position = 'p1'
        self.test_positions = ['p2', 'p3', 'p4']

        # 固定参数
        self.K = K  # 固定K=3
        self.M = None  # M将通过优化确定

        # 数据存储
        self.all_data = {}
        self.device_ids = {}

        # PA系数和信道估计
        self.f_coeffs = {}
        self.h_estimates = {}

        # 特征
        self.features_all = {}

        # 分类器
        self.scaler = StandardScaler()
        self.classifier = None

        print("="*70)
        print(f"基于新方法论的信道鲁棒RF指纹识别系统 (K={self.K})")
        print("核心公式：y = (T(h)⊗D_R(I_K))f + K_n")
        print("="*70)

    def load_data(self):
        """步骤1：加载数据"""
        print("\n=== 步骤1：数据加载 ===")

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

                    # 提取信号
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

        # 可视化1
        self._visualize_raw_signals()
        print("\n✓ 数据加载完成\n")

    def _visualize_raw_signals(self):
        """可视化1：原始信号对比（K=3标记）"""
        print("🎨 生成可视化1：原始信号对比 (K=3)")

        fig = plt.figure(figsize=(18, 10))
        n_devices = min(3, len(self.all_data['p1']))

        for dev_idx in range(n_devices):
            device_id = self.device_ids['p1'][dev_idx]

            for pos_idx, pos in enumerate(self.positions):
                ax = plt.subplot(n_devices, 4, dev_idx*4 + pos_idx + 1)
                signal = self.all_data[pos][dev_idx]
                t = np.arange(len(signal)) / 1e6

                display_len = min(1000, len(signal))
                ax.plot(t[:display_len], np.abs(signal[:display_len]),
                       linewidth=0.8, color='#2E86AB')

                ax.set_xlabel('时间 (μs)', fontsize=9)
                ax.set_ylabel('幅度', fontsize=9)
                ax.set_title(f'设备{device_id} @ {pos} (K={self.K})', 
                           fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('viz_1_raw_signals_k3.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_1_raw_signals_k3.png\n")

    def optimize_M_adaptive(self, M_range=range(5, 26)):
        """步骤2：自适应M参数估计（K固定为3）"""
        print(f"=== 步骤2：自适应M参数估计（K={self.K}固定）===")

        # 使用p1的第一个设备
        y = self.all_data['p1'][0]
        d = self.all_data['p1'][0]

        M_range = list(M_range)
        rms_errors = []

        print(f"搜索范围：M={M_range[0]}-{M_range[-1]}")

        for M in M_range:
            try:
                # 构建 D_R(I_K) 矩阵
                D_matrix = self._construct_D_R_matrix(d, self.K, M)

                if D_matrix.shape[0] == 0:
                    rms_errors.append(np.inf)
                    continue

                # 简化：h初始化为单位冲激
                L_h = 10
                h_init = np.zeros(L_h, dtype=complex)
                h_init[0] = 1.0

                # 构建 T(h)
                T_h = self._construct_T_h(h_init, D_matrix.shape[0])

                # 构建扩展矩阵
                A_matrix = kron(T_h, D_matrix)

                y_trunc = y[:A_matrix.shape[0]]

                # LS估计
                f_est = np.linalg.lstsq(A_matrix, y_trunc, rcond=None)[0]

                # 重构
                y_reconstructed = A_matrix @ f_est

                # RMS误差
                rms = np.sqrt(np.mean(np.abs(y_trunc - y_reconstructed)**2))
                rms_errors.append(rms)

            except:
                rms_errors.append(np.inf)

            if M % 5 == 0:
                print(f"  已完成 M={M}")

        # 可视化2
        self._visualize_M_search(M_range, rms_errors)

        # 选择最优M
        valid_errors = [e for e in rms_errors if np.isfinite(e)]
        if len(valid_errors) > 0:
            min_idx = np.argmin(rms_errors)
            self.M = M_range[min_idx]
            min_rms = rms_errors[min_idx]
        else:
            self.M = 10
            min_rms = np.inf

        print(f"✓ 最优参数：K={self.K}(固定), M={self.M} (RMS={min_rms:.6f})")
        print()

    def _construct_D_R_matrix(self, d, K, M):
        """构建 D_R 矩阵（新方法论）"""
        N = len(d) - M
        if N <= 0:
            return np.array([]).reshape(0, (K+1)*(M+1))

        D = np.zeros((N, (K+1)*(M+1)), dtype=complex)

        for m in range(M+1):
            for k in range(K+1):
                col_idx = m * (K+1) + k
                if m + N <= len(d):
                    # 非线性基函数：d[n-m] * |d[n-m]|^(2k)
                    D[:, col_idx] = d[m:N+m] * np.abs(d[m:N+m])**(2*k)

        return D

    def _construct_T_h(self, h, N):
        """构建Toeplitz矩阵 T(h)"""
        L_h = len(h)
        # Toeplitz矩阵：第一列为h，第一行为[h[0], 0, ..., 0]
        first_col = np.concatenate([h, np.zeros(N - L_h, dtype=complex)])
        first_row = np.concatenate([h[0:1], np.zeros(N - 1, dtype=complex)])
        T_h = toeplitz(first_col[:N], first_row[:N])
        return T_h

    def _visualize_M_search(self, M_range, rms_errors):
        """可视化2：M参数搜索"""
        print("🎨 生成可视化2：M参数搜索 (K=3)")

        plt.figure(figsize=(12, 7))

        # 过滤无穷值
        valid_indices = [i for i, e in enumerate(rms_errors) if np.isfinite(e)]
        valid_M = [M_range[i] for i in valid_indices]
        valid_errors = [rms_errors[i] for i in valid_indices]

        if len(valid_errors) > 0:
            plt.plot(valid_M, valid_errors, 'o-', linewidth=2, 
                    markersize=8, color='#E63946')

            # 标注最优点
            min_idx = np.argmin(valid_errors)
            plt.plot(valid_M[min_idx], valid_errors[min_idx], 
                    '*', markersize=20, color='cyan', 
                    markeredgewidth=2, markeredgecolor='black')

            plt.text(valid_M[min_idx], valid_errors[min_idx], 
                    f'  最优M={valid_M[min_idx]}', 
                    fontsize=11, fontweight='bold')

        plt.xlabel('记忆深度 M', fontsize=12, fontweight='bold')
        plt.ylabel('RMS 误差', fontsize=12, fontweight='bold')
        plt.title(f'M参数搜索 - RMS误差曲线（K={self.K}固定）', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('viz_2_M_search_k3.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_2_M_search_k3.png\n")

    def alternating_optimization(self, lambda_f=0.01, lambda_m=0.02, 
                                 num_iterations=30):
        """
        步骤3：交替优化 f 和 h（新方法论核心）
        min_{f,h} ||（T(h)⊗D_R(I_K))f - y||² + λ_f||G_f f||² + λ_m Σ||f_Bj||²
        """
        print("=== 步骤3：交替优化 f 和 h（新方法论）===")
        print(f"参数：K={self.K}, M={self.M}")
        print(f"正则化：λ_f={lambda_f}, λ_m={lambda_m}")

        # 对训练位置的每个设备进行优化
        for pos in [self.train_position]:
            print(f"\n处理位置 {pos}")

            for dev_idx, device_id in enumerate(self.device_ids[pos]):
                y = self.all_data[pos][dev_idx]
                d = self.all_data[self.train_position][0]  # 参考符号

                # 初始化
                L_h = 10
                h_current = np.zeros(L_h, dtype=complex)
                h_current[0] = 1.0

                D_R = self._construct_D_R_matrix(d, self.K, self.M)
                N = D_R.shape[0]
                y_trunc = y[:N]

                # LS初始化 f
                T_h = self._construct_T_h(h_current, N)
                A_matrix = kron(T_h, D_R)
                f_current = np.linalg.lstsq(A_matrix, y_trunc, rcond=None)[0]

                loss_history = []

                # 交替迭代
                for iter in range(num_iterations):
                    # 1. 固定h，更新f
                    # min_f ||（T(h)⊗D_R(I_K))f - y||² + λ_f||G_f f||²
                    T_h = self._construct_T_h(h_current, N)
                    A_matrix = kron(T_h, D_R)

                    # 添加Tikhonov正则化
                    ATA = A_matrix.conj().T @ A_matrix
                    ATy = A_matrix.conj().T @ y_trunc
                    reg_f = lambda_f * np.eye(ATA.shape[0])

                    try:
                        f_current = np.linalg.solve(ATA + reg_f, ATy)
                    except:
                        pass

                    # 2. 固定f，更新h
                    # min_h ||（T(h)⊗D_R(I_K))f - y||²
                    # 重构为关于h的线性系统
                    Df = (D_R @ np.eye(D_R.shape[1] if D_R.shape[1] <= len(f_current) 
                                       else len(f_current))) @ f_current[:D_R.shape[1]]

                    if len(Df) >= L_h:
                        # 简化：用循环卷积近似
                        try:
                            # 构建Toeplitz系统求解h
                            B_matrix = toeplitz(Df[:L_h], Df[:L_h])
                            h_current = np.linalg.lstsq(B_matrix, y_trunc[:L_h], 
                                                        rcond=None)[0]
                        except:
                            pass

                    # 计算损失
                    T_h = self._construct_T_h(h_current, N)
                    A_matrix = kron(T_h, D_R)
                    residual = A_matrix @ f_current - y_trunc
                    loss = np.linalg.norm(residual)**2 + lambda_f * np.linalg.norm(f_current)**2
                    loss_history.append(loss)

                # 保存优化结果
                self.f_coeffs[device_id] = f_current
                self.h_estimates[device_id] = h_current

                if dev_idx == 0:
                    # 可视化第一个设备的优化过程
                    self._visualize_alternating_optimization(
                        loss_history, device_id)

            print(f"  ✓ 完成 {len(self.f_coeffs)} 个设备的优化")

        print("\n✓ 交替优化完成\n")

    def _visualize_alternating_optimization(self, loss_history, device_id):
        """可视化3：交替优化过程"""
        print(f"🎨 生成可视化3：交替优化过程 (K=3, 设备{device_id})")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 子图1：损失曲线
        axes[0].plot(loss_history, linewidth=2, color='#E63946')
        axes[0].set_xlabel('迭代次数', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('损失值', fontsize=12, fontweight='bold')
        axes[0].set_title(f'交替优化收敛曲线 (K={self.K})', 
                         fontsize=13, fontweight='bold')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)

        # 子图2：f系数幅度
        f = self.f_coeffs[device_id]
        axes[1].stem(np.arange(len(f)), np.abs(f), basefmt=' ')
        axes[1].set_xlabel('系数索引', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('幅度', fontsize=12, fontweight='bold')
        axes[1].set_title(f'PA系数 f 幅度 (K={self.K}, M={self.M})', 
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # 子图3：h信道估计
        h = self.h_estimates[device_id]
        axes[2].stem(np.arange(len(h)), np.abs(h), basefmt=' ', 
                    linefmt='C1-', markerfmt='C1o')
        axes[2].set_xlabel('抽头索引', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('幅度', fontsize=12, fontweight='bold')
        axes[2].set_title(f'信道估计 h (L={len(h)})', 
                         fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('viz_3_alternating_optimization_k3.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_3_alternating_optimization_k3.png\n")

    def estimate_channel_for_test_positions(self):
        """步骤4：为测试位置估计信道"""
        print("=== 步骤4：测试位置信道估计 ===")

        # 使用训练位置的参考设备
        d_ref = self.all_data[self.train_position][0]
        f_ref = list(self.f_coeffs.values())[0]

        for pos in self.test_positions:
            print(f"处理位置 {pos}")

            # 使用第一个设备估计信道
            y = self.all_data[pos][0]
            D_R = self._construct_D_R_matrix(d_ref, self.K, self.M)
            N = D_R.shape[0]
            y_trunc = y[:N]

            # 简化：用伪逆估计h
            L_h = 10
            Df = D_R @ f_ref[:D_R.shape[1]]

            try:
                if len(Df) >= L_h:
                    B_matrix = toeplitz(Df[:L_h], Df[:L_h])
                    h_est = np.linalg.lstsq(B_matrix, y_trunc[:L_h], rcond=None)[0]
                else:
                    h_est = np.zeros(L_h, dtype=complex)
                    h_est[0] = 1.0
            except:
                h_est = np.zeros(L_h, dtype=complex)
                h_est[0] = 1.0

            # 保存到所有测试设备
            for device_id in self.device_ids[pos]:
                self.h_estimates[device_id] = h_est

            print(f"  ✓ 估计完成")

        # 可视化4
        self._visualize_channel_comparison()

        print("\n✓ 信道估计完成\n")

    def _visualize_channel_comparison(self):
        """可视化4：信道对比"""
        print("🎨 生成可视化4：信道频率响应对比 (K=3)")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, pos in enumerate(self.positions):
            # 获取该位置第一个设备的信道
            device_id = self.device_ids[pos][0]
            h = self.h_estimates.get(device_id, np.array([1.0]))

            # FFT
            H_freq = np.fft.fft(h, n=512)
            freqs = np.fft.fftfreq(512, d=1.0)

            pos_freqs = freqs[:256]
            H_mag = np.abs(H_freq[:256])

            axes[idx].plot(pos_freqs, 20*np.log10(H_mag + 1e-10),
                          linewidth=2, color='#2A9D8F')
            axes[idx].set_xlabel('归一化频率', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('幅度 (dB)', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'位置 {pos} - 信道频率响应 (K={self.K})', 
                               fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim([0, 0.5])

        plt.tight_layout()
        plt.savefig('viz_4_channel_comparison_k3.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_4_channel_comparison_k3.png\n")

    def extract_improved_features(self):
        """步骤5：提取改进特征（基于新方法论）"""
        print("=== 步骤5：特征提取（基于新方法论）===")

        for pos in self.positions:
            print(f"提取位置 {pos} 的特征")

            features = []

            for dev_idx, device_id in enumerate(self.device_ids[pos]):
                # 获取f和h
                if device_id in self.f_coeffs:
                    f = self.f_coeffs[device_id]
                else:
                    f = list(self.f_coeffs.values())[0]

                h = self.h_estimates.get(device_id, np.array([1.0]))

                # === 特征设计（基于K=3的PA系数）===
                # 重塑f为 (M+1) x (K+1) 矩阵
                f_matrix = f.reshape(self.M+1, self.K+1)

                # 特征1: 不同阶数系数的比值
                phi1 = np.abs(f_matrix[:, 1].mean()) / (np.abs(f_matrix[:, 2].mean()) + 1e-10)

                # 特征2: 记忆深度的影响
                phi2 = np.abs(f_matrix[0, :].mean()) / (np.abs(f_matrix[-1, :].mean()) + 1e-10)

                # 特征3: 总能量分布
                energy_k1 = np.sum(np.abs(f_matrix[:, 1])**2)
                energy_k2 = np.sum(np.abs(f_matrix[:, 2])**2)
                energy_k3 = np.sum(np.abs(f_matrix[:, 3])**2)
                phi3 = energy_k1 / (energy_k1 + energy_k2 + energy_k3 + 1e-10)

                # 特征4: 信道特征
                phi4 = np.linalg.norm(h, 2)
                phi5 = np.max(np.abs(h))

                # 特征5: 信号统计特征
                y = self.all_data[pos][dev_idx]
                phi6 = np.std(np.abs(y))
                phi7 = np.mean(np.abs(y)**2)  # 功率

                # 特征6: 相位特征
                phi8 = np.std(np.angle(f_matrix.flatten()))

                features.append([phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8])

            self.features_all[pos] = np.array(features)
            print(f"  ✓ 提取 {len(features)} 个设备，每个 8 维特征")

        # 可视化5
        self._visualize_feature_distribution()

        print("\n✓ 特征提取完成\n")

    def _visualize_feature_distribution(self):
        """可视化5：特征分布"""
        print("🎨 生成可视化5：特征分布对比 (K=3)")

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        feature_names = [
            'φ₁: f₁/f₂比值', 'φ₂: 记忆深度比', 'φ₃: k=1能量占比',
            'φ₄: 信道L2范数', 'φ₅: 信道峰值', 'φ₆: 信号标准差',
            'φ₇: 信号功率', 'φ₈: 相位标准差'
        ]

        for feat_idx in range(8):
            ax = axes[feat_idx]

            for pos in self.positions:
                if pos not in self.features_all:
                    continue

                features = self.features_all[pos][:, feat_idx]
                ax.hist(features, bins=20, alpha=0.5, label=pos, edgecolor='black')

            ax.set_xlabel('特征值', fontsize=10, fontweight='bold')
            ax.set_ylabel('频数', fontsize=10, fontweight='bold')
            ax.set_title(f'{feature_names[feat_idx]} (K={self.K})', 
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('viz_5_feature_distribution_k3.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_5_feature_distribution_k3.png\n")

    def train_classifier(self):
        """步骤6：训练分类器"""
        print("=== 步骤6：分类器训练（1P：p1）===")

        X_train = self.features_all[self.train_position]
        y_train = np.array(self.device_ids[self.train_position])

        print(f"训练集：{len(X_train)} 个样本")

        X_train_norm = self.scaler.fit_transform(X_train)

        self.classifier = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        self.classifier.fit(X_train_norm, y_train)

        print("✓ 训练完成\n")

    def evaluate_classifier(self):
        """步骤7：评估分类器"""
        print("=== 步骤7：分类器评估（3P：p2/p3/p4）===")

        results = {}

        for pos in self.test_positions:
            print(f"\n测试位置：{pos}")

            X_test = self.features_all[pos]
            y_test = np.array(self.device_ids[pos])

            X_test_norm = self.scaler.transform(X_test)
            y_pred = self.classifier.predict(X_test_norm)

            acc = accuracy_score(y_test, y_pred) * 100
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

            results[pos] = {
                'accuracy': acc,
                'confusion_matrix': cm,
                'y_true': y_test,
                'y_pred': y_pred
            }

            print(f"  准确率：{acc:.2f}%")

        # 可视化6-7
        self._visualize_accuracy(results)
        self._visualize_confusion_matrices(results)

        avg_acc = np.mean([r['accuracy'] for r in results.values()])

        print("\n" + "="*70)
        print("实验总结")
        print("="*70)
        print(f"固定参数：K={self.K}, M={self.M}")
        print(f"平均准确率：{avg_acc:.2f}%")
        for pos, res in results.items():
            print(f"  {pos}: {res['accuracy']:.2f}%")
        print("="*70)

    def _visualize_accuracy(self, results):
        """可视化6：准确率对比"""
        print("\n🎨 生成可视化6：准确率对比 (K=3)")

        plt.figure(figsize=(10, 6))

        positions = list(results.keys())
        accuracies = [results[pos]['accuracy'] for pos in positions]

        bars = plt.bar(range(len(positions)), accuracies,
                      color=['#E63946', '#F4A261', '#2A9D8F'],
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        plt.xlabel('测试位置', fontsize=13, fontweight='bold')
        plt.ylabel('准确率 (%)', fontsize=13, fontweight='bold')
        plt.title(f'跨位置识别准确率（K={self.K}, M={self.M}）\n1P训练：p1 → 3P测试：p2/p3/p4',
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(positions)), positions, fontsize=12)
        plt.ylim([0, 105])
        plt.grid(True, alpha=0.3, axis='y')

        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            plt.text(i, acc + 2, f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        avg_acc = np.mean(accuracies)
        plt.axhline(y=avg_acc, color='red', linestyle='--', linewidth=2,
                   label=f'平均: {avg_acc:.2f}%', alpha=0.7)
        plt.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig('viz_6_accuracy_comparison_k3.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_6_accuracy_comparison_k3.png")

    def _visualize_confusion_matrices(self, results):
        """可视化7：混淆矩阵"""
        print("🎨 生成可视化7：混淆矩阵 (K=3)")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for idx, (pos, res) in enumerate(results.items()):
            cm = res['confusion_matrix']
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

            im = axes[idx].imshow(cm_norm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

            axes[idx].set_xlabel('预测设备ID', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('真实设备ID', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{pos} - 混淆矩阵 (准确率: {res["accuracy"]:.1f}%) (K={self.K})',
                               fontsize=12, fontweight='bold')

            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('准确率', fontsize=10)

            device_ids = np.unique(res['y_true'])
            axes[idx].set_xticks(range(len(device_ids)))
            axes[idx].set_yticks(range(len(device_ids)))
            axes[idx].set_xticklabels(device_ids, rotation=45, fontsize=8)
            axes[idx].set_yticklabels(device_ids, fontsize=8)

        plt.tight_layout()
        plt.savefig('viz_7_confusion_matrices_k3.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_7_confusion_matrices_k3.png\n")

    def run_full_pipeline(self):
        """运行完整流程"""
        start_time = time.time()

        self.load_data()
        self.optimize_M_adaptive()
        self.alternating_optimization()
        self.estimate_channel_for_test_positions()
        self.extract_improved_features()
        self.train_classifier()
        self.evaluate_classifier()

        elapsed = time.time() - start_time
        print(f"\n⏱️ 总耗时：{elapsed:.2f} 秒")
        print("\n✅ 所有可视化已保存（K=3版本）：")
        print("  1. viz_1_raw_signals_k3.png - 原始信号对比")
        print("  2. viz_2_M_search_k3.png - M参数搜索")
        print("  3. viz_3_alternating_optimization_k3.png - 交替优化过程")
        print("  4. viz_4_channel_comparison_k3.png - 信道频率响应")
        print("  5. viz_5_feature_distribution_k3.png - 特征分布")
        print("  6. viz_6_accuracy_comparison_k3.png - 准确率对比")
        print("  7. viz_7_confusion_matrices_k3.png - 混淆矩阵")

def main():
    """主函数"""
    system = ImprovedChannelResilientRFF(K=3)
    system.run_full_pipeline()

if __name__ == "__main__":
    main()


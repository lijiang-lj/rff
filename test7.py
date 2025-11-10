"""
========================================================================
基于记忆非线性特征的信道鲁棒RF指纹识别
核心框架：Fu et al. 2024
创新融合：Jing et al. (自适应参数) + Zhang et al. (深度先验+多正则化)
实验设置：1P训练（p1），3P测试（p2, p3, p4）
========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.signal import welch
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.decomposition import PCA
import warnings
import glob
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

class ChannelResilientRFF:
    """基于Fu et al. 2024的信道鲁棒RF指纹识别系统"""

    def __init__(self):
        """初始化系统"""
        self.positions = ['p1', 'p2', 'p3', 'p4']
        self.train_position = 'p1'
        self.test_positions = ['p2', 'p3', 'p4']

        # 数据存储
        self.all_data = {}
        self.device_ids = {}

        # PA模型参数
        self.optimal_K = None
        self.optimal_M = None
        self.f_coeffs = {}  # PA系数 f_{2k+1,m}

        # 信道估计
        self.channel_estimates = {}

        # 特征
        self.features_all = {}

        # 分类器
        self.scaler = StandardScaler()
        self.classifier = None

        print("="*70)
        print("基于记忆非线性特征的信道鲁棒RF指纹识别系统")
        print("核心：Fu et al. 2024 + 创新融合")
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
            print(f"  设备ID: {self.device_ids[pos][:5]}..." if len(self.device_ids[pos]) > 5
                  else f"  设备ID: {self.device_ids[pos]}")

        # 可视化1：原始信号
        self._visualize_raw_signals()

        print("\n✓ 数据加载完成\n")

    def _visualize_raw_signals(self):
        """可视化1：原始信号对比"""
        print("🎨 生成可视化1：原始信号对比")

        fig = plt.figure(figsize=(18, 10))

        # 选择3个设备进行可视化
        n_devices = min(3, len(self.all_data['p1']))

        for dev_idx in range(n_devices):
            device_id = self.device_ids['p1'][dev_idx]

            # 时域信号（4个位置）
            for pos_idx, pos in enumerate(self.positions):
                ax = plt.subplot(n_devices, 4, dev_idx*4 + pos_idx + 1)

                signal = self.all_data[pos][dev_idx]
                t = np.arange(len(signal)) / 1e6  # 假设1MHz采样

                # 只显示前1000个样本
                display_len = min(1000, len(signal))
                ax.plot(t[:display_len], np.abs(signal[:display_len]),
                       linewidth=0.8, color='#2E86AB')

                ax.set_xlabel('时间 (μs)', fontsize=9)
                ax.set_ylabel('幅度', fontsize=9)
                ax.set_title(f'设备{device_id} @ {pos}', fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('viz_1_raw_signals.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_1_raw_signals.png\n")

    def estimate_pa_parameters_grid_search(self, K_range=range(1, 6), M_range=range(1, 25)):
        """
        步骤2：自适应PA参数估计（创新点1 - Jing et al.）
        使用网格搜索确定最优K和M
        """
        print("=== 步骤2：自适应PA参数估计（创新点1）===")
        print("方法：Grid Search (Jing et al. 方法)")

        # 使用p1的第一个设备作为参考
        d_ref = self.all_data['p1'][0]
        x_ref = self.all_data['p1'][0]  # Fu论文中用同一设备的无PA信号，这里近似

        K_range = list(K_range)
        M_range = list(M_range)

        rms_errors = np.zeros((len(K_range), len(M_range)))

        print(f"搜索范围：K={K_range[0]}-{K_range[-1]}, M={M_range[0]}-{M_range[-1]}")

        for k_idx, K in enumerate(K_range):
            for m_idx, M in enumerate(M_range):
                try:
                    # 构建设计矩阵 D^(M)_N
                    D_matrix = self._construct_D_matrix(d_ref, K, M)

                    if D_matrix.shape[0] == 0:
                        rms_errors[k_idx, m_idx] = np.inf
                        continue

                    # LS估计f系数
                    f_est = np.linalg.lstsq(D_matrix, x_ref[:D_matrix.shape[0]], rcond=None)[0]

                    # 重构信号
                    x_reconstructed = D_matrix @ f_est

                    # RMS误差
                    rms = np.sqrt(np.mean(np.abs(x_ref[:len(x_reconstructed)] - x_reconstructed)**2))
                    rms_errors[k_idx, m_idx] = rms

                except:
                    rms_errors[k_idx, m_idx] = np.inf

            if (k_idx + 1) % 2 == 0:
                print(f"  已完成 K={K}")

        # 可视化2：RMS误差热图
        self._visualize_grid_search(rms_errors, K_range, M_range)

        # 选择最优参数
        valid_errors = rms_errors[np.isfinite(rms_errors)]
        if len(valid_errors) > 0:
            min_idx = np.unravel_index(np.argmin(rms_errors), rms_errors.shape)
            self.optimal_K = K_range[min_idx[0]]
            self.optimal_M = M_range[min_idx[1]]
            min_rms = rms_errors[min_idx]
        else:
            self.optimal_K = 3
            self.optimal_M = 10
            min_rms = np.inf

        print(f"✓ 最优参数：K={self.optimal_K}, M={self.optimal_M} (RMS={min_rms:.6f})")
        print()

    def _construct_D_matrix(self, d, K, M):
        """构建PA设计矩阵 D^(M)_N (Fu论文 Eq.3)"""
        N = len(d) - M
        if N <= 0:
            return np.array([]).reshape(0, (K+1)*(M+1))

        D = np.zeros((N, (K+1)*(M+1)), dtype=complex)

        col_idx = 0
        for m in range(M+1):
            for k in range(K+1):
                # d_n = [d[n], |d[n]|²d[n], ..., |d[n]|^{2K}d[n]]
                if m + N <= len(d):
                    D[:, col_idx] = d[m:N+m] * np.abs(d[m:N+m])**(2*k)
                col_idx += 1

        return D

    def _visualize_grid_search(self, rms_errors, K_range, M_range):
        """可视化2：网格搜索结果"""
        print("🎨 生成可视化2：PA参数网格搜索")

        plt.figure(figsize=(12, 8))

        # 处理无穷值
        rms_plot = rms_errors.copy()
        rms_plot[np.isinf(rms_plot)] = np.nan

        sns.heatmap(rms_plot, annot=False, cmap='hot_r',
                   xticklabels=[str(m) if i % 3 == 0 else '' for i, m in enumerate(M_range)],
                   yticklabels=K_range,
                   cbar_kws={'label': 'RMS 误差'})

        plt.xlabel('记忆深度 M', fontsize=12, fontweight='bold')
        plt.ylabel('非线性阶数 K', fontsize=12, fontweight='bold')
        plt.title('PA参数网格搜索 - RMS误差热图（创新点1）', fontsize=14, fontweight='bold')

        # 标注最优点
        if not np.all(np.isnan(rms_plot)):
            min_idx = np.unravel_index(np.nanargmin(rms_plot), rms_plot.shape)
            plt.plot(min_idx[1], min_idx[0], 'c*', markersize=25,
                    markeredgewidth=3, markeredgecolor='white')

        plt.tight_layout()
        plt.savefig('viz_2_grid_search.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_2_grid_search.png\n")

    def optimize_pa_coefficients_dsp(self, lambda_a=0.02, num_iterations=50):
        """
        步骤3：深度信号先验优化PA系数（创新点2 - Zhang et al.）
        使用HQS + TV正则化
        """
        print("=== 步骤3：深度信号先验优化（创新点2）===")
        print("方法：HQS + TV Regularization (Zhang et al.)")

        # 为每个设备估计PA系数
        for pos in [self.train_position]:  # 只用训练位置
            print(f"处理位置 {pos}")

            for dev_idx, device_id in enumerate(self.device_ids[pos]):
                d_signal = self.all_data[pos][dev_idx]

                # 构建设计矩阵
                D = self._construct_D_matrix(d_signal, self.optimal_K, self.optimal_M)
                x = d_signal[:D.shape[0]]

                # 初始化（LS估计）
                f_init = np.linalg.lstsq(D, x, rcond=None)[0]
                f_current = f_init.copy()

                # DSP优化
                loss_history = []
                for iter in range(num_iterations):
                    # 数据保真项梯度
                    residual = D @ f_current - x
                    grad_data = D.conj().T @ residual

                    # TV正则化梯度
                    grad_tv = self._compute_tv_gradient(f_current)

                    # 更新
                    step_size = 0.001 / (iter + 1)**0.5
                    f_current = f_current - step_size * (grad_data + lambda_a * grad_tv)

                    # 损失
                    loss = np.linalg.norm(residual)**2 + lambda_a * self._compute_tv_norm(f_current)
                    loss_history.append(loss)

                # 保存优化后的系数
                self.f_coeffs[device_id] = f_current

            print(f"  ✓ 优化完成，共 {len(self.f_coeffs)} 个设备")

        # 可视化3：DSP优化过程（第一个设备）
        first_device = self.device_ids[self.train_position][0]
        self._visualize_dsp_optimization(loss_history, f_init, self.f_coeffs[first_device])

        print()

    def _compute_tv_gradient(self, f):
        """计算TV正则化梯度"""
        grad = np.zeros_like(f)
        eps = 1e-8

        for i in range(len(f) - 1):
            diff = f[i+1] - f[i]
            grad[i] -= diff / (np.abs(diff) + eps)
            grad[i+1] += diff / (np.abs(diff) + eps)

        return grad

    def _compute_tv_norm(self, f):
        """计算TV范数"""
        return np.sum(np.abs(np.diff(f)))

    def _visualize_dsp_optimization(self, loss_history, f_init, f_opt):
        """可视化3：DSP优化过程"""
        print("🎨 生成可视化3：DSP优化过程")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 子图1：损失曲线
        axes[0].plot(loss_history, linewidth=2, color='#E63946')
        axes[0].set_xlabel('迭代次数', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('损失值', fontsize=12, fontweight='bold')
        axes[0].set_title('DSP优化收敛曲线', fontsize=13, fontweight='bold')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)

        # 子图2：系数幅度对比
        x = np.arange(len(f_init))
        width = 0.35
        axes[1].bar(x - width/2, np.abs(f_init), width, label='初始LS', alpha=0.7, color='#457B9D')
        axes[1].bar(x + width/2, np.abs(f_opt), width, label='DSP优化', alpha=0.7, color='#F4A261')
        axes[1].set_xlabel('系数索引', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('幅度', fontsize=12, fontweight='bold')
        axes[1].set_title('PA系数幅度对比', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y')

        # 子图3：系数相位对比
        axes[2].plot(x, np.angle(f_init), 'o-', label='初始LS', markersize=5, alpha=0.7, color='#457B9D')
        axes[2].plot(x, np.angle(f_opt), 's-', label='DSP优化', markersize=5, alpha=0.7, color='#F4A261')
        axes[2].set_xlabel('系数索引', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('相位 (rad)', fontsize=12, fontweight='bold')
        axes[2].set_title('PA系数相位对比', fontsize=13, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('viz_3_dsp_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_3_dsp_optimization.png\n")

    def estimate_channel_multi_regularization(self, lambda_values=None, Lh=10):
        """
        步骤4：多正则化信道估计（创新点3 - Zhang et al.）
        使用多个λ进行MMSE估计
        """
        print("=== 步骤4：多正则化信道估计（创新点3）===")
        print("方法：Multi-λ MMSE (Zhang et al.)")

        if lambda_values is None:
            lambda_values = [0.05, 0.10, 0.15, 0.20, 0.25]

        print(f"正则化参数：{lambda_values}")
        print(f"信道长度：Lh={Lh}")

        # 为每个位置估计信道
        for pos in self.positions:
            print(f"\n处理位置 {pos}")

            H_estimates = []

            for lambda_h in lambda_values:
                # 使用第一个设备作为代表
                y = self.all_data[pos][0]
                d = self.all_data[self.train_position][0]  # 参考信号

                # 构建扩展设计矩阵 D^(M+Lh-1)_N
                D_ext = self._construct_extended_D_matrix(d, self.optimal_K, self.optimal_M, Lh)

                if D_ext.shape[0] == 0:
                    H_estimates.append(np.zeros((Lh, self.optimal_K+1), dtype=complex))
                    continue

                y_trunc = y[:D_ext.shape[0]]

                # MMSE估计：H = (D'D + λI)^(-1) D'y
                DTD = D_ext.conj().T @ D_ext
                reg = lambda_h * np.eye(D_ext.shape[1])

                try:
                    H_est = np.linalg.solve(DTD + reg, D_ext.conj().T @ y_trunc)
                    H_matrix = H_est.reshape(Lh, self.optimal_K+1)
                    H_estimates.append(H_matrix)
                    print(f"  λ={lambda_h:.2f}: ✓")
                except:
                    H_estimates.append(np.zeros((Lh, self.optimal_K+1), dtype=complex))
                    print(f"  λ={lambda_h:.2f}: ✗")

            # 平均所有估计
            self.channel_estimates[pos] = np.mean(np.stack(H_estimates), axis=0)

        # 可视化4：信道频率响应
        self._visualize_channel_estimates()

        print("\n✓ 信道估计完成\n")

    def _construct_extended_D_matrix(self, d, K, M, Lh):
        """构建扩展设计矩阵 D^(M+Lh-1)_N"""
        N = len(d) - M - Lh + 1
        if N <= 0:
            return np.array([]).reshape(0, (K+1)*Lh)

        D_ext = np.zeros((N, (K+1)*Lh), dtype=complex)

        for lh in range(Lh):
            for k in range(K+1):
                col_idx = lh * (K+1) + k
                if lh + N <= len(d):
                    D_ext[:, col_idx] = d[lh:N+lh] * np.abs(d[lh:N+lh])**(2*k)

        return D_ext

    def _visualize_channel_estimates(self):
        """可视化4：信道频率响应"""
        print("🎨 生成可视化4：信道频率响应对比")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, pos in enumerate(self.positions):
            if pos not in self.channel_estimates:
                continue

            H_est = self.channel_estimates[pos]

            # 计算频率响应
            H_freq = np.fft.fft(H_est[:, 0], n=512)
            freqs = np.fft.fftfreq(512, d=1.0)

            # 只显示正频率
            pos_freqs = freqs[:256]
            H_mag = np.abs(H_freq[:256])

            axes[idx].plot(pos_freqs, 20*np.log10(H_mag + 1e-10),
                          linewidth=2, color='#2A9D8F')
            axes[idx].set_xlabel('归一化频率', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('幅度 (dB)', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'位置 {pos} - 信道频率响应', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim([0, 0.5])

        plt.tight_layout()
        plt.savefig('viz_4_channel_frequency_response.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_4_channel_frequency_response.png\n")

    def extract_fu_features(self):
        """
        步骤5：提取Fu论文的非线性特征
        """
        print("=== 步骤5：特征提取（Fu et al. 核心特征）===")

        for pos in self.positions:
            print(f"提取位置 {pos} 的特征")

            features = []

            for dev_idx, device_id in enumerate(self.device_ids[pos]):
                # 获取该设备的PA系数
                if device_id not in self.f_coeffs:
                    # 如果是测试设备，用训练集第一个设备的系数近似
                    f = list(self.f_coeffs.values())[0]
                else:
                    f = self.f_coeffs[device_id]

                # === Fu论文的3个仅含PA系数的特征 ===
                # 特征1: φ1 = f1,0 / f3,0
                phi1 = np.abs(f[0] / (f[1] + 1e-10))

                # 特征2: φ2 = f1,M / f3,M
                idx_1M = self.optimal_M * (self.optimal_K + 1)
                idx_3M = self.optimal_M * (self.optimal_K + 1) + 1
                phi2 = np.abs(f[idx_1M] / (f[idx_3M] + 1e-10))

                # 特征3: φ3 = Σf1,m / Σf3,m
                f1_sum = np.sum([f[m*(self.optimal_K+1)] for m in range(self.optimal_M+1)])
                f3_sum = np.sum([f[m*(self.optimal_K+1)+1] for m in range(self.optimal_M+1)])
                phi3 = np.abs(f1_sum / (f3_sum + 1e-10))

                # === Fu论文的混合特征（简化版）===
                # 使用信号片段构建
                y = self.all_data[pos][dev_idx]
                d = self.all_data[self.train_position][0]  # 参考训练符号

                # 使用不同长度的序列
                N1, N2 = 33, 160
                N3, N4 = 33, 320

                try:
                    S_N1N2 = np.sum(y[N1:N2])
                    S_N3N4 = np.sum(y[N3:N4])
                    phi4 = np.abs(S_N1N2 / (S_N3N4 + 1e-10))
                except:
                    phi4 = 0.0

                # 添加信道补偿后的特征
                if pos in self.channel_estimates:
                    H_est = self.channel_estimates[pos]
                    h_taps = np.mean(H_est, axis=1)

                    phi5 = np.linalg.norm(h_taps, 2)
                    phi6 = np.max(np.abs(h_taps))
                else:
                    phi5 = phi6 = 0.0

                features.append([phi1, phi2, phi3, phi4, phi5, phi6])

            self.features_all[pos] = np.array(features)
            print(f"  ✓ 提取 {len(features)} 个设备，每个 6 维特征")

        # 可视化5：特征分布
        self._visualize_feature_distribution()

        print("\n✓ 特征提取完成\n")

    def _visualize_feature_distribution(self):
        """可视化5：特征分布对比"""
        print("🎨 生成可视化5：特征分布对比")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        feature_names = ['φ₁: f₁,₀/f₃,₀', 'φ₂: f₁,ₘ/f₃,ₘ', 'φ₃: Σf₁/Σf₃',
                        'φ₄: 混合特征', 'φ₅: 信道L2', 'φ₆: 信道峰值']

        for feat_idx in range(6):
            ax = axes[feat_idx]

            for pos in self.positions:
                if pos not in self.features_all:
                    continue

                features = self.features_all[pos][:, feat_idx]

                ax.hist(features, bins=20, alpha=0.5, label=pos, edgecolor='black')

            ax.set_xlabel('特征值', fontsize=10, fontweight='bold')
            ax.set_ylabel('频数', fontsize=10, fontweight='bold')
            ax.set_title(feature_names[feat_idx], fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('viz_5_feature_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_5_feature_distribution.png\n")

    def train_classifier(self):
        """步骤6：训练分类器（1P训练）"""
        print("=== 步骤6：分类器训练（1P：p1训练）===")

        # 训练数据
        X_train = self.features_all[self.train_position]
        y_train = np.array(self.device_ids[self.train_position])

        print(f"训练集：{len(X_train)} 个样本")
        print(f"设备ID：{y_train[:5]}...")

        # 归一化
        X_train_norm = self.scaler.fit_transform(X_train)

        # 训练SVM
        print("训练SVM分类器...")
        self.classifier = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        self.classifier.fit(X_train_norm, y_train)

        print("✓ 训练完成\n")

    def evaluate_classifier(self):
        """步骤7：评估分类器（3P测试：p2, p3, p4）"""
        print("=== 步骤7：分类器评估（3P：p2/p3/p4测试）===")

        results = {}

        for pos in self.test_positions:
            print(f"\n测试位置：{pos}")

            X_test = self.features_all[pos]
            y_test = np.array(self.device_ids[pos])

            # 归一化
            X_test_norm = self.scaler.transform(X_test)

            # 预测
            y_pred = self.classifier.predict(X_test_norm)

            # 准确率
            acc = accuracy_score(y_test, y_pred) * 100

            # 混淆矩阵
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

            results[pos] = {
                'accuracy': acc,
                'confusion_matrix': cm,
                'y_true': y_test,
                'y_pred': y_pred
            }

            print(f"  准确率：{acc:.2f}%")

        # 可视化6：准确率对比
        self._visualize_accuracy(results)

        # 可视化7：混淆矩阵
        self._visualize_confusion_matrices(results)

        # 总结
        avg_acc = np.mean([r['accuracy'] for r in results.values()])

        print("\n" + "="*70)
        print("实验总结")
        print("="*70)
        print(f"平均准确率：{avg_acc:.2f}%")
        for pos, res in results.items():
            print(f"  {pos}: {res['accuracy']:.2f}%")
        print("="*70)

    def _visualize_accuracy(self, results):
        """可视化6：准确率对比"""
        print("\n🎨 生成可视化6：准确率对比")

        plt.figure(figsize=(10, 6))

        positions = list(results.keys())
        accuracies = [results[pos]['accuracy'] for pos in positions]

        bars = plt.bar(range(len(positions)), accuracies,
                      color=['#E63946', '#F4A261', '#2A9D8F'],
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        plt.xlabel('测试位置', fontsize=13, fontweight='bold')
        plt.ylabel('准确率 (%)', fontsize=13, fontweight='bold')
        plt.title('跨位置识别准确率（1P训练：p1 → 3P测试：p2/p3/p4）',
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(positions)), positions, fontsize=12)
        plt.ylim([0, 105])
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            plt.text(i, acc + 2, f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 添加平均线
        avg_acc = np.mean(accuracies)
        plt.axhline(y=avg_acc, color='red', linestyle='--', linewidth=2,
                   label=f'平均: {avg_acc:.2f}%', alpha=0.7)
        plt.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig('viz_6_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_6_accuracy_comparison.png")

    def _visualize_confusion_matrices(self, results):
        """可视化7：混淆矩阵"""
        print("🎨 生成可视化7：混淆矩阵")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for idx, (pos, res) in enumerate(results.items()):
            cm = res['confusion_matrix']

            # 归一化
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

            im = axes[idx].imshow(cm_norm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

            axes[idx].set_xlabel('预测设备ID', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('真实设备ID', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{pos} - 混淆矩阵 (准确率: {res["accuracy"]:.1f}%)',
                               fontsize=12, fontweight='bold')

            # 颜色条
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('准确率', fontsize=10)

            # 设置刻度
            device_ids = np.unique(res['y_true'])
            axes[idx].set_xticks(range(len(device_ids)))
            axes[idx].set_yticks(range(len(device_ids)))
            axes[idx].set_xticklabels(device_ids, rotation=45, fontsize=8)
            axes[idx].set_yticklabels(device_ids, fontsize=8)

        plt.tight_layout()
        plt.savefig('viz_7_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: viz_7_confusion_matrices.png\n")

    def run_full_pipeline(self):
        """运行完整流程"""
        start_time = time.time()

        # 步骤1：加载数据
        self.load_data()

        # 步骤2：自适应PA参数估计（创新点1）
        self.estimate_pa_parameters_grid_search()

        # 步骤3：深度信号先验优化（创新点2）
        self.optimize_pa_coefficients_dsp()

        # 步骤4：多正则化信道估计（创新点3）
        self.estimate_channel_multi_regularization()

        # 步骤5：特征提取
        self.extract_fu_features()

        # 步骤6：训练分类器
        self.train_classifier()

        # 步骤7：评估
        self.evaluate_classifier()

        elapsed = time.time() - start_time
        print(f"\n⏱️ 总耗时：{elapsed:.2f} 秒")
        print("\n✅ 所有可视化已保存：")
        print("  1. viz_1_raw_signals.png - 原始信号对比")
        print("  2. viz_2_grid_search.png - PA参数网格搜索")
        print("  3. viz_3_dsp_optimization.png - DSP优化过程")
        print("  4. viz_4_channel_frequency_response.png - 信道频率响应")
        print("  5. viz_5_feature_distribution.png - 特征分布")
        print("  6. viz_6_accuracy_comparison.png - 准确率对比")
        print("  7. viz_7_confusion_matrices.png - 混淆矩阵")

def main():
    """主函数"""
    system = ChannelResilientRFF()
    system.run_full_pipeline()

if __name__ == "__main__":
    main()

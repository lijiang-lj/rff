%% 星座图版测试脚本 - 非线性指纹分析
clear; clc; close all;

% ==============================
% 步骤1: 配置环境与参数
% ==============================
addpath('FDE_functions'); 
input_file = '38.mat';         
output_file = 'NLfinger_results_38.mat';

% 检查必要文件
required_files = {'OFDM_STF_LTF.mat', 'Tx_OFDM_preamble.mat', input_file};
for i = 1:length(required_files)
    if ~exist(required_files{i}, 'file')
        error('缺少必要文件: %s', required_files{i});
    end
end

% ==============================
% 步骤2: 执行指纹提取
% ==============================
fprintf('处理数据: %s\n', input_file);
tic;
Get_NLfinger_raw_F(input_file, output_file);
processing_time = toc;
fprintf('指纹提取完成! 耗时: %.2f 秒\n', processing_time);

% ==============================
% 步骤3: 加载结果 - 准备星座图分析
% ==============================
load(output_file, 'Store_NLfinger', 'Get_count_point');

% 基础信息
num_segments = size(Store_NLfinger, 1);
fingerprint_dim = size(Store_NLfinger, 2);
fprintf('\n===== 结果分析 =====\n');
fprintf('有效段数量: %d\n', num_segments);
fprintf('指纹维度: %d\n', fingerprint_dim);

% ==============================
% 步骤4: 星座图可视化
% ==============================
figure('Name', '非线性指纹星座图分析', 'Position', [100, 100, 1400, 800]);

% 1. 所有指纹点的综合星座图
subplot(2, 3, [1, 2, 4, 5]);
hold on;

% 为不同维度的指纹分配不同颜色
colors = jet(fingerprint_dim);
markers = {'o', '+', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};

for dim = 1:fingerprint_dim
    % 选择标记样式
    marker_idx = mod(dim-1, length(markers)) + 1;
    
    % 绘制当前维度的所有指纹点
    scatter(real(Store_NLfinger(:, dim)), ...
            imag(Store_NLfinger(:, dim)), ...
            40, colors(dim, :), markers{marker_idx}, 'LineWidth', 1.5);
end

% 添加参考线和标注
plot([-1, 1]*max(abs([xlim, ylim])), [0, 0], 'k--', 'LineWidth', 0.5);
plot([0, 0], [-1, 1]*max(abs([xlim, ylim])), 'k--', 'LineWidth', 0.5);
title('所有指纹维度综合星座图');
xlabel('实部');
ylabel('虚部');
grid on;
axis equal;

% 添加维度图例（只显示前10维避免拥挤）
if fingerprint_dim > 10
    legend_dims = round(linspace(1, fingerprint_dim, 10));
else
    legend_dims = 1:fingerprint_dim;
end
h_leg = arrayfun(@(d) plot(NaN, NaN, markers{mod(d-1, length(markers))+1}, ...
    'Color', colors(d, :), 'MarkerSize', 8, 'LineWidth', 1.5), legend_dims);
legend(h_leg, cellstr(num2str(legend_dims', '维度 %d')), 'Location', 'bestoutside');

hold off;

% 2. 前6个维度的详细星座图
subplot(2, 3, 3);
num_dims_to_show = min(6, fingerprint_dim);
for dim = 1:num_dims_to_show
    subplot(2, 3, 3);
    hold on;
    scatter(real(Store_NLfinger(:, dim)), ...
            imag(Store_NLfinger(:, dim)), ...
            50, colors(dim, :), 'filled', 'MarkerEdgeColor', 'k');
end
title(sprintf('前%d维指纹星座图', num_dims_to_show));
xlabel('实部');
ylabel('虚部');
grid on;
hold off;

% 3. 选择点分布分析
subplot(2, 3, 6);
histogram(Get_count_point, 'BinWidth', 1, 'FaceColor', [0.2, 0.6, 0.8]);
title('特征选择点分布');
xlabel('选择点索引');
ylabel('出现次数');
grid on;
xlim([0 40]);

% 保存图像
saveas(gcf, 'NLfinger_Constellation_Analysis.png');

% ==============================
% 步骤5: 深度星座图分析 (单独窗口)
% ==============================
fig2 = figure('Name', '指纹维度详细星座图', 'Position', [200, 200, 1200, 800]);

% 计算需要展示的行列数
num_rows = ceil(sqrt(fingerprint_dim));
num_cols = ceil(fingerprint_dim/num_rows);

for dim = 1:fingerprint_dim
    subplot(num_rows, num_cols, dim);
    
    % 绘制当前维度的星座图
    scatter(real(Store_NLfinger(:, dim)), ...
            imag(Store_NLfinger(:, dim)), ...
            40, 'b', 'filled', 'MarkerEdgeColor', 'k');
    
    % 添加统计信息
    hold on;
    plot(mean(real(Store_NLfinger(:, dim))), mean(imag(Store_NLfinger(:, dim))), ...
        'r+', 'MarkerSize', 12, 'LineWidth', 2);
    
    % 计算并绘制95%置信椭圆
    data = [real(Store_NLfinger(:, dim)), imag(Store_NLfinger(:, dim))];
    if size(data, 1) > 2
        ellipse_params = error_ellipse(data, 0.95);
        plot(ellipse_params(:,1), ellipse_params(:,2), 'r-', 'LineWidth', 1.5);
    end
    
    hold off;
    
    title(sprintf('维度 %d', dim));
    xlabel('实部');
    ylabel('虚部');
    grid on;
    axis equal;
end

% 保存详细星座图
saveas(fig2, 'Detailed_Fingerprint_Constellations.png');

% ==============================
% 辅助函数: 计算置信椭圆
% ==============================
function ellipse = error_ellipse(data, conf)
    % 计算置信椭圆
    mean_val = mean(data);
    cov_mat = cov(data);
    [eig_vec, eig_val] = eig(cov_mat);
    
    % 计算卡方值
    chi_val = sqrt(chi2inv(conf, 2));
    
    % 生成椭圆点
    theta = linspace(0, 2*pi, 100);
    ellipse = (eig_vec * sqrt(eig_val)) * [cos(theta); sin(theta)] * chi_val;
    ellipse = ellipse' + repmat(mean_val, size(ellipse, 2), 1);
end

% ==============================
% 步骤6: 统计分析与报告
% ==============================
% 计算统计指标
fingerprint_stats = struct();
for dim = 1:fingerprint_dim
    re = real(Store_NLfinger(:, dim));
    im = imag(Store_NLfinger(:, dim));
    mag = abs(Store_NLfinger(:, dim));
    phase = angle(Store_NLfinger(:, dim));
    
    fingerprint_stats(dim).Real = struct(...
        'Mean', mean(re), 'Std', std(re), 'Min', min(re), 'Max', max(re));
    fingerprint_stats(dim).Imag = struct(...
        'Mean', mean(im), 'Std', std(im), 'Min', min(im), 'Max', max(im));
    fingerprint_stats(dim).Magnitude = struct(...
        'Mean', mean(mag), 'Std', std(mag), 'Min', min(mag), 'Max', max(mag));
    fingerprint_stats(dim).Phase = struct(...
        'Mean', mean(phase), 'Std', std(phase), 'Min', min(phase), 'Max', max(phase));
end

% 保存统计结果
save(output_file, 'fingerprint_stats', '-append');

% 生成星座图分析报告
report_filename = 'constellation_analysis_report.txt';
fid = fopen(report_filename, 'w');
fprintf(fid, '非线性指纹星座图分析报告\n');
fprintf(fid, '生成时间: %s\n', datetime);
fprintf(fid, '数据文件: %s\n', input_file);
fprintf(fid, '有效段数量: %d\n', num_segments);
fprintf(fid, '处理时间: %.2f秒\n', processing_time);
fprintf(fid, '\n===== 指纹维度统计摘要 =====\n');

% 只显示前5维详细统计
for dim = 1:min(5, fingerprint_dim)
    fprintf(fid, '\n维度 %d:\n', dim);
    fprintf(fid, '  实部: 均值=%.4f, 标准差=%.4f\n', ...
            fingerprint_stats(dim).Real.Mean, fingerprint_stats(dim).Real.Std);
    fprintf(fid, '  虚部: 均值=%.4f, 标准差=%.4f\n', ...
            fingerprint_stats(dim).Imag.Mean, fingerprint_stats(dim).Imag.Std);
    fprintf(fid, '  模值: 均值=%.4f, 标准差=%.4f\n', ...
            fingerprint_stats(dim).Magnitude.Mean, fingerprint_stats(dim).Magnitude.Std);
end

fprintf(fid, '\n===== 选择点分布统计 =====\n');
fprintf(fid, '均值: %.2f ± %.2f\n', mean(Get_count_point), std(Get_count_point));
fprintf(fid, '范围: %d 到 %d\n', min(Get_count_point), max(Get_count_point));

fclose(fid);
fprintf('分析报告已保存为: %s\n', report_filename);
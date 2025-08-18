import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesFeatureEngineer:
    """
    高维时间序列特征工程器
    """

    def __init__(self, window_sizes=[5, 10, 20, 60],
                 feature_groups=None,
                 target_cols=None):
        """
        Parameters:
        -----------
        window_sizes : list
            滚动窗口大小列表
        feature_groups : dict
            特征分组字典
        target_cols : list
            目标列名列表
        """
        self.window_sizes = window_sizes
        self.feature_groups = feature_groups
        self.target_cols = target_cols
        self.feature_importance = {}

    def create_temporal_features(self, df, columns=None):
        """
        创建时间相关特征

        Returns:
        --------
        df_temporal : DataFrame
            包含时间特征的数据框
        """
        print("创建时间特征...")
        df_result = df.copy()

        if columns is None:
            columns = [col for col in df.columns
                       if col not in ['date_id'] and not col.endswith('_is_active')]

        new_features = {}

        for window in self.window_sizes:
            print(f"  处理窗口大小: {window}")

            for col in columns:
                if col in df.columns:
                    # 1. 滚动统计量
                    # 均值
                    new_features[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()

                    # 标准差（波动性）
                    new_features[f'{col}_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()

                    # 最大值和最小值
                    new_features[f'{col}_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                    new_features[f'{col}_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()

                    # 滚动峰值和谷值
                    new_features[f'{col}_rolling_peak_trough_{window}'] = df[col].rolling(window=window).max() - df[
                        col].rolling(window=window).min()

                    # 滚动中位数
                    new_features[f'{col}_rolling_median_{window}'] = df[col].rolling(window=window).median()

                    # 滚动自相关
                    new_features[f'{col}_rolling_autocorr_{window}'] = df[col].rolling(window=window).apply(
                        lambda x: x.autocorr())

                    # 2. 变化率特征
                    # 简单收益率
                    new_features[f'{col}_return_{window}'] = df[col].pct_change(periods=window)
                    # 对数收益率
                    new_features[f'{col}_log_return_{window}'] = np.log(df[col] / df[col].shift(window))
                    # 移动平均收益率
                    new_features[f'{col}_moving_avg_return_{window}'] = df[col].pct_change().rolling(window=window).mean()
                    # 指数加权平均收益率
                    new_features[f'{col}_ewma_{window}'] = df[col].ewm(span=window).mean().pct_change()
                    # 波动率
                    new_features[f'{col}_volatility_{window}'] = np.log(df[col] / df[col].shift(1)).rolling(window=window).std()

                    # 3. 技术指标
                    # 相对位置（当前值在窗口中的相对位置）
                    rolling_max = df[col].rolling(window=window, min_periods=1).max()
                    rolling_min = df[col].rolling(window=window, min_periods=1).min()
                    range_val = rolling_max - rolling_min
                    range_val[range_val == 0] = 1  # 避免除零
                    new_features[f'{col}_rsi_{window}'] = (df[col] - rolling_min) / range_val

        # 4. Lag特征（更短期的）
        print("  创建Lag特征...")
        for col in columns:
            if col in df.columns:
                for lag in [1, 2, 3, 4, 5, 6, 7]:
                    new_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    new_features[f'{col}_lag_{lag}_return'] = df[col].pct_change(periods=lag).shift(lag) # 滞后收益率特征
                    new_features[f'{col}_lag_{lag}_ma'] = df[col].shift(lag).rolling(window=window).mean() # 滞后移动平均
                    new_features[f'{col}_lag_{lag}_std'] = df[col].shift(lag).rolling(window=window).std() # 滞后最大值和最小值
                    new_features[f'{col}_trend'] = df[col] / df[col].shift(1) - 1 # 滞后特征的趋势

        # 5. 差分特征
        print("  创建差分特征...")
        for col in columns:
            if col in df.columns:
                new_features[f'{col}_diff_1'] = df[col].diff(1)
                new_features[f'{col}_diff_7'] = df[col].diff(7)
                seasonal_period = 12  # 季节性周期为12个月
                new_features[f'{col}_seasonal_diff_{seasonal_period}'] = df[col] - df[col].shift(seasonal_period)
                new_features[f'{col}_log_diff'] = np.log(df[col]) - np.log(df[col].shift(1)) # 对数差分

        # 合并新特征
        df_result = pd.concat([df_result, pd.DataFrame(new_features)], axis=1)

        print(f"  ✓ 创建了 {len(new_features)} 个时间特征")
        return df_result

class kalmanFilterModel:
    def __init__(self):
        pass

    def apply_kalman_filter(self, series, Q = 1e-5, R = 0.1):

        from filterpy.kalman import KalmanFilter
        # 初始化卡尔曼滤波器
        kf = KalmanFilter(dim_x=len(series), dim_z=1)
        # 状态转移矩阵
        kf.F = np.array([[1, 1],
                         [0,1]])
        # 测量矩阵
        kf.H = np.array([[1, 0]])

        # 测量噪声协方差
        kf.R = R

        # 过程噪声协方差
        kf.Q = np.array([[Q, 0],
                         [0, Q]])

        # 初始状态估计
        kf.x = np.array([series.iloc[0],0])

        kf.P *= 100

        # 应用滤波
        filtered_values = []
        for value in series.fillna(method='ffill').fillna(method='bfill').fillna(0):
            kf.predict()
            kf.update(value)
            filtered_values.append(kf.x[0]) # 获取滤波后状态

        return pd.Series(filtered_values, index = series.index)

    def grid_search(self, series, param_grid):
        """
        使用网格搜索来选择最优的Q与R
        :param series: 输入的时间序列数据
        :param param_grid: 包含Q与R参数的字典
        :return: 最优Q与R组合及其对应的性能
        """

        best_params = None
        best_score = float('inf')
        best_filtered_values = None

        for Q in param_grid['Q']:
            for R in param_grid['R']:
                # 获取滤波结果
                filtered_series = self.apply_kalman_filter(series, Q, R)
                # 计算MSE最为评估标准
                mse = mean_squared_error(series, filtered_series)

                if mse < best_score:
                    best_score = mse
                    best_params = {'Q':Q, 'R':R}
                    best_filtered_values = filtered_series

        return best_params, best_score, best_filtered_values


    def optimize_kalman_filter(self, series, asset1_series, asset2_series,
                               Q_range =[1e-6, 1e-5, 1e-4, 1e-3],
                               R_range =[0.01, 0.05, 0.1, 0.15, 0.5, 1.0]):
        """
        :param series:  原始价差序列
        :param asset1_series:  资产1的价格序列
        :param asset2_series:  资产2的价格序列
        :param Q_range:  过程噪声协方差的搜索范围
        :param R_range:  测量噪声协方差的搜索范围
        :return:
        ------
        best_Q: float 最优的Q值
        best_R: float 最优的R值
        best_score: float 最优得分
        """
        from sklearn.metrics import mean_squared_error
        import itertools


    def create_cross_sectional_features(self, df, feature_groups=None, pairs_file='target_pairs.csv'):
        """
        创建横截面特征（不同资产间的关系）

        Parameters:
        -----------
        df : DataFrame
            输入数据
        feature_groups : dict
            特征分组
        pairs_file : str
            包含资产对信息的CSV文件路径
        """
        print("\n创建横截面特征...")
        df_result = df.copy()

        if feature_groups is None:
            feature_groups = self._auto_group_features(df)

        new_features = {}

        # 1. 组内特征
        print("  创建组内统计特征...")
        for group_name, group_cols in feature_groups.items():
            valid_cols = [col for col in group_cols if col in df.columns]

            if len(valid_cols) > 1:
                # 组内均值
                new_features[f'{group_name}_mean'] = df[valid_cols].mean(axis=1)

                # 组内标准差
                new_features[f'{group_name}_std'] = df[valid_cols].std(axis=1)

                # 组内最大最小值
                new_features[f'{group_name}_max'] = df[valid_cols].max(axis=1)
                new_features[f'{group_name}_min'] = df[valid_cols].min(axis=1)

                # 组内排名特征
                for col in valid_cols[:5]:  # 限制数量避免特征爆炸
                    rank = df[valid_cols].rank(axis=1, pct=True)[col]
                    new_features[f'{col}_rank_in_{group_name}'] = rank

        # 2. 跨组特征（比率和差异）
        print("  创建跨组关系特征...")
        group_means = {}
        for group_name, group_cols in feature_groups.items():
            valid_cols = [col for col in group_cols if col in df.columns]
            if valid_cols:
                group_means[group_name] = df[valid_cols].mean(axis=1)

        # 创建重要的跨组比率
        if 'JPX' in group_means and 'US_Stock' in group_means:
            new_features['JPX_US_ratio'] = group_means['JPX'] / (group_means['US_Stock'] + 1e-8)

        if 'FX' in group_means and 'US_Stock' in group_means:
            new_features['FX_US_ratio'] = group_means['FX'] / (group_means['US_Stock'] + 1e-8)

        # 3. 相关性特征（滚动相关性）- 从target_pairs.csv读取
        print("  创建滚动相关性特征...")

        # 读取资产对信息
        key_pairs = []
        try:
            import os
            if os.path.exists(pairs_file):
                pairs_df = pd.read_csv(pairs_file)
                print(f"    从 {pairs_file} 读取了 {len(pairs_df)} 个资产对")

                # 解析pair列，格式可能是：
                # 1. 单个资产: "US_Stock_VT_adj_close"
                # 2. 两个资产的差: "LME_PB_Close - US_Stock_VT_adj_close"
                for idx, row in pairs_df.iterrows():
                    pair_str = row['pair'].strip()

                    # 检查是否包含减号（表示两个资产的差）
                    if ' - ' in pair_str:
                        # 分割成两个资产
                        asset1, asset2 = pair_str.split(' - ')
                        asset1 = asset1.strip()
                        asset2 = asset2.strip()

                        # 检查这两个资产是否在数据列中存在
                        if asset1 in df.columns and asset2 in df.columns:
                            key_pairs.append((asset1, asset2))
                        else:
                            # 尝试模糊匹配
                            cols1 = [col for col in df.columns if asset1 in col or col in asset1]
                            cols2 = [col for col in df.columns if asset2 in col or col in asset2]
                            if cols1 and cols2:
                                key_pairs.append((cols1[0], cols2[0]))

                    else:
                        # 单个资产的情况，需要找另一个相关资产配对
                        # 这里我们可以选择与基准资产配对，比如VT（全球股票指数）
                        if pair_str in df.columns:
                            # 与一个基准资产配对
                            benchmark_cols = ['US_Stock_VT_adj_close', 'FX_USDJPY', 'US_Stock_GLD_adj_close']
                            for benchmark in benchmark_cols:
                                if benchmark in df.columns and benchmark != pair_str:
                                    key_pairs.append((pair_str, benchmark))
                                    break

                # 去重
                key_pairs = list(set(key_pairs))
                print(f"    成功构建了 {len(key_pairs)} 个唯一资产对")

            else:
                print(f"    警告：未找到 {pairs_file}，使用默认资产对")
                # 使用默认的资产对
                key_pairs = [
                    ('US_Stock_GLD_adj_close', 'FX_USDJPY'),
                    ('US_Stock_XLE_adj_close', 'US_Stock_CVX_adj_close'),
                    ('LME_AH_Close', 'US_Stock_FCX_adj_close'),
                ]

        except Exception as e:
            print(f"    读取资产对文件时出错: {e}")
            print("    使用默认资产对")
            key_pairs = [
                ('US_Stock_GLD_adj_close', 'FX_USDJPY'),
                ('US_Stock_XLE_adj_close', 'US_Stock_CVX_adj_close'),
                ('LME_AH_Close', 'US_Stock_FCX_adj_close'),
            ]

        # 创建滚动相关性特征
        correlation_count = 0
        max_correlations = 100  # 限制最大相关性特征数量，避免特征爆炸

        for i, (col1, col2) in enumerate(key_pairs):
            if correlation_count >= max_correlations:
                print(f"    达到最大相关性特征数量限制 ({max_correlations})，停止创建")
                break

            if col1 in df.columns and col2 in df.columns:
                for window in [20, 60]:
                    corr = df[col1].rolling(window=window, min_periods=10).corr(df[col2])
                    # 创建简短的特征名
                    feature_name = f'pair_{i}_corr_{window}'
                    new_features[feature_name] = corr
                    correlation_count += 1

        print(f"    创建了 {correlation_count} 个相关性特征")

        # 合并新特征
        df_result = pd.concat([df_result, pd.DataFrame(new_features)], axis=1)

        print(f"  ✓ 创建了 {len(new_features)} 个横截面特征")
        return df_result

    def create_market_regime_features(self, df):
        """
        创建市场状态特征
        """
        print("\n创建市场状态特征...")
        df_result = df.copy()
        new_features = {}

        # 1. 波动率regime
        volatility_cols = [col for col in df.columns if 'std_' in col]
        if volatility_cols:
            # 市场整体波动率
            new_features['market_volatility'] = df[volatility_cols].mean(axis=1)

            # 波动率regime（高/中/低）
            vol_percentiles = df[volatility_cols].mean(axis=1).quantile([0.33, 0.67])
            new_features['volatility_regime'] = pd.cut(
                df[volatility_cols].mean(axis=1),
                bins=[-np.inf, vol_percentiles[0.33], vol_percentiles[0.67], np.inf],
                labels=[0, 1, 2]
            ).astype(float)

        # 2. 趋势强度
        # 使用移动平均的斜率
        ma_cols = [col for col in df.columns if '_ma_20' in col]
        if ma_cols:
            for col in ma_cols[:10]:  # 限制数量
                # 20日均线的5日变化率
                ma_change = df[col].diff(5) / (df[col].shift(5) + 1e-8)
                new_features[f'{col}_trend_strength'] = ma_change

        # 3. 市场情绪指标
        # 使用避险资产vs风险资产的比率
        safe_assets = ['US_Stock_GLD_adj_close', 'US_Stock_IAU_adj_close', 'FX_USDJPY']
        risk_assets = ['US_Stock_XLE_adj_close', 'US_Stock_FCX_adj_close']

        safe_cols = [col for col in safe_assets if col in df.columns]
        risk_cols = [col for col in risk_assets if col in df.columns]

        if safe_cols and risk_cols:
            safe_mean = df[safe_cols].mean(axis=1)
            risk_mean = df[risk_cols].mean(axis=1)
            new_features['risk_on_off_ratio'] = risk_mean / (safe_mean + 1e-8)

        # 合并新特征
        df_result = pd.concat([df_result, pd.DataFrame(new_features)], axis=1)

        print(f"  ✓ 创建了 {len(new_features)} 个市场状态特征")
        return df_result

    def create_pca_features(self, df, n_components=50, feature_groups=None):
        """
        使用PCA降维创建主成分特征
        """
        print(f"\n创建PCA特征 (保留{n_components}个主成分)...")
        df_result = df.copy()

        if feature_groups is None:
            feature_groups = self._auto_group_features(df)

        pca_features = {}

        for group_name, group_cols in feature_groups.items():
            valid_cols = [col for col in group_cols
                          if col in df.columns and not col.endswith('_is_active')]

            if len(valid_cols) > 3:  # 只对有足够特征的组做PCA
                print(f"  处理 {group_name} 组 ({len(valid_cols)} 个特征)...")

                # 准备数据
                group_data = df[valid_cols].fillna(0)

                # 标准化
                scaler = StandardScaler()
                group_data_scaled = scaler.fit_transform(group_data)

                # PCA
                n_comp = min(n_components, len(valid_cols), len(df))
                pca = PCA(n_components=n_comp)
                pca_result = pca.fit_transform(group_data_scaled)

                # 保存主成分
                for i in range(min(10, n_comp)):  # 每组最多保留10个主成分
                    pca_features[f'{group_name}_PC{i + 1}'] = pca_result[:, i]

                # 保存解释方差比例
                explained_var = pca.explained_variance_ratio_[:min(10, n_comp)].sum()
                print(f"    前{min(10, n_comp)}个主成分解释了 {explained_var:.2%} 的方差")

        # 合并PCA特征
        df_result = pd.concat([df_result, pd.DataFrame(pca_features)], axis=1)

        print(f"  ✓ 创建了 {len(pca_features)} 个PCA特征")
        return df_result

    def create_interaction_features(self, df, important_features=None, max_interactions=50):
        """
        创建重要特征间的交互项
        """
        print(f"\n创建交互特征 (最多{max_interactions}个)...")
        df_result = df.copy()

        if important_features is None:
            # 选择一些关键特征
            important_features = [
                'US_Stock_GLD_adj_close',
                'FX_USDJPY',
                'FX_CADCHF',
                'FX_NZDCAD',
                'FX_GBPCAD',
                'FX_EURUSD',
                'volume',
                'Volatility',
                'US_Stock_XLE_adj_close',
                'LME_AH_Close',
                'LME_CA_Close',
                'LME_ZS_Close',
                'LME_PB_Close',
                'JPX_Gold_Standard_Futures_Open',
                'JPX_Platinum_Standard_Futures_Open',
                'JPX_Platinum_Standard_Futures_Close',
                'JPX_Gold_Standard_Futures_Close'
            ]

        # 过滤存在的特征
        valid_features = [f for f in important_features if f in df.columns]

        interaction_features = {}
        count = 0

        # 创建乘积交互
        for i, feat1 in enumerate(valid_features):
            for feat2 in valid_features[i + 1:]:
                if count >= max_interactions:
                    break

                # 乘积
                interaction_features[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]

                # 比率
                interaction_features[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)

                count += 2

        # 合并交互特征
        df_result = pd.concat([df_result, pd.DataFrame(interaction_features)], axis=1)

        print(f"  ✓ 创建了 {len(interaction_features)} 个交互特征")
        return df_result

    def _auto_group_features(self, df):
        """
        自动将特征分组
        """
        groups = {
            'LME': [],
            'JPX': [],
            'US_Stock': [],
            'FX': []
        }

        for col in df.columns:
            if col.startswith('LME'):
                groups['LME'].append(col)
            elif col.startswith('JPX'):
                groups['JPX'].append(col)
            elif col.startswith('US_Stock'):
                groups['US_Stock'].append(col)
            elif col.startswith('FX'):
                groups['FX'].append(col)

        # 进一步细分US_Stock
        groups['US_Stock_price'] = [col for col in groups['US_Stock'] if 'close' in col or 'open' in col]
        groups['US_Stock_volume'] = [col for col in groups['US_Stock'] if 'volume' in col]

        return groups

    def create_all_features(self, df,
                            temporal=True,
                            cross_sectional=True,
                            market_regime=True,
                            pca=True,
                            interaction=True):
        """
        创建所有特征的主函数
        """
        print("=" * 60)
        print("开始全面特征工程")
        print("=" * 60)

        df_result = df.copy()
        original_cols = len(df.columns)

        # 1. 时间特征
        if temporal:
            # 只对价格相关的列创建时间特征（避免特征爆炸）
            price_cols = [col for col in df.columns
                          if any(x in col for x in ['close', 'Close', 'open', 'Open'])
                          and not col.endswith('_is_active')][:30]  # 限制数量
            df_result = self.create_temporal_features(df_result, columns=price_cols)

        # 2. 横截面特征
        if cross_sectional:
            df_result = self.create_cross_sectional_features(df_result)

        # 3. 市场状态特征
        if market_regime:
            df_result = self.create_market_regime_features(df_result)

        # 4. PCA特征
        if pca:
            df_result = self.create_pca_features(df_result, n_components=20)

        # 5. 交互特征
        if interaction:
            df_result = self.create_interaction_features(df_result, max_interactions=30)

        print(f"\n特征工程完成!")
        print(f"原始特征数: {original_cols}")
        print(f"最终特征数: {len(df_result.columns)}")
        print(f"新增特征数: {len(df_result.columns) - original_cols}")

        return df_result


# ===========================
# 完整的特征工程流程示例
# ===========================

def complete_feature_engineering_pipeline():
    """
    完整的特征工程流程，包含所有步骤
    """

    print("=" * 60)
    print("开始完整的特征工程流程")
    print("=" * 60)

    # Step 1: 读取已处理缺失值的数据
    print("\nStep 1: 读取数据...")

    # 这个df是你之前处理过缺失值的训练数据
    # 如果你已经运行过缺失值处理，应该有一个 'train_final.csv' 或类似的文件
    df = pd.read_csv('train_final.csv')  # 或者使用你保存的处理后数据文件名
    print(f"  数据维度: {df.shape}")
    print(f"  列数: {len(df.columns)}")

    # Step 2: 读取目标数据（用于后续特征选择）
    labels_df = pd.read_csv('train_labels.csv')
    print(f"  目标数据维度: {labels_df.shape}")

    # Step 3: 创建特征工程器实例
    print("\nStep 2: 初始化特征工程器...")

    # 直接使用已经定义的类，不需要导入
    engineer = TimeSeriesFeatureEngineer(
        window_sizes=[5, 10, 20, 60],  # 滚动窗口大小
        target_cols=[col for col in labels_df.columns if col != 'date_id']
    )

    # Step 4: 逐步创建不同类型的特征

    # 4.1 时间特征（只对价格列，避免特征爆炸）
    print("\nStep 3: 创建时间特征...")
    price_cols = [col for col in df.columns
                  if any(x in col.lower() for x in ['close', 'open', 'high', 'low'])
                  and not col.endswith('_is_active')][:30]  # 限制数量

    df_with_temporal = engineer.create_temporal_features(df, columns=price_cols)
    print(f"  当前特征数: {len(df_with_temporal.columns)}")

    # 4.2 横截面特征（包括从target_pairs.csv读取的资产对）
    print("\nStep 4: 创建横截面特征...")
    df_with_cross = engineer.create_cross_sectional_features(
        df_with_temporal,
        pairs_file='target_pairs.csv'  # 使用target_pairs.csv中的资产对
    )
    print(f"  当前特征数: {len(df_with_cross.columns)}")

    # 4.3 市场状态特征
    print("\nStep 5: 创建市场状态特征...")
    df_with_market = engineer.create_market_regime_features(df_with_cross)
    print(f"  当前特征数: {len(df_with_market.columns)}")

    # 4.4 PCA特征（降维）
    print("\nStep 6: 创建PCA特征...")
    df_with_pca = engineer.create_pca_features(
        df_with_market,
        n_components=20  # 每组保留20个主成分
    )
    print(f"  当前特征数: {len(df_with_pca.columns)}")

    # 4.5 交互特征
    print("\nStep 7: 创建交互特征...")
    # 选择重要的特征进行交互
    important_features = [
        'US_Stock_GLD_adj_close',
        'FX_USDJPY',
        'US_Stock_XLE_adj_close',
        'LME_AH_Close'
    ]
    # 过滤实际存在的特征
    important_features = [f for f in important_features if f in df_with_pca.columns]

    df_with_interactions = engineer.create_interaction_features(
        df_with_pca,
        important_features=important_features,
        max_interactions=30
    )
    print(f"  最终特征数: {len(df_with_interactions.columns)}")

    # Step 5: 特征选择（如果特征太多）
    if len(df_with_interactions.columns) > 300:  # 如果特征超过300个
        print(f"\nStep 8: 特征选择（从{len(df_with_interactions.columns)}个特征中选择200个）...")

        from sklearn.feature_selection import mutual_info_regression

        # 合并特征和目标
        merged_df = pd.merge(df_with_interactions, labels_df, on='date_id', how='inner')

        # 选择一个目标进行特征重要性评估
        feature_cols = [col for col in df_with_interactions.columns if col != 'date_id']
        target_col = 'target_0'  # 使用第一个目标

        X = merged_df[feature_cols].fillna(0)
        y = merged_df[target_col].fillna(0)

        # 计算互信息
        mi_scores = mutual_info_regression(X, y, random_state=42)

        # 选择top 200特征
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        top_features = feature_importance.head(200)['feature'].tolist()

        # 保留选中的特征
        df_final = df_with_interactions[['date_id'] + top_features]

        # 保存特征重要性
        feature_importance.to_csv('feature_importance.csv', index=False)
        print(f"  特征重要性已保存")
    else:
        df_final = df_with_interactions

    # Step 6: 保存最终特征
    print(f"\nStep 9: 保存最终特征...")
    df_final.to_csv('train_features_complete.csv', index=False)
    print(f"  ✓ 特征已保存至: train_features_complete.csv")
    print(f"  最终数据维度: {df_final.shape}")

    # Step 7: 特征统计
    print("\n特征类型统计:")
    feature_stats = {
        'lag特征': len([c for c in df_final.columns if 'lag_' in c]),
        '移动平均': len([c for c in df_final.columns if '_ma_' in c]),
        '波动率': len([c for c in df_final.columns if '_std_' in c]),
        '收益率': len([c for c in df_final.columns if '_return_' in c]),
        '相关性': len([c for c in df_final.columns if '_corr_' in c]),
        'PCA特征': len([c for c in df_final.columns if '_PC' in c]),
        '交互特征': len([c for c in df_final.columns if '_x_' in c or '_div_' in c]),
        '市场状态': len([c for c in df_final.columns if 'regime' in c or 'volatility' in c]),
        '指示器': len([c for c in df_final.columns if '_is_active' in c])
    }

    for feat_type, count in feature_stats.items():
        if count > 0:
            print(f"  {feat_type}: {count}")

    return df_final


# ===========================
# 简化版本（如果你想快速测试）
# ===========================

def quick_feature_engineering():
    """
    快速版本的特征工程，只创建最重要的特征
    """
    print("执行快速特征工程...")

    # 读取数据
    df = pd.read_csv('train_final.csv')  # 处理过缺失值的数据

    # 只创建最重要的特征
    new_features = pd.DataFrame(index=df.index)

    # 1. 关键列的移动平均
    key_cols = ['US_Stock_GLD_adj_close', 'FX_USDJPY', 'US_Stock_VT_adj_close']
    for col in key_cols:
        if col in df.columns:
            new_features[f'{col}_ma_20'] = df[col].rolling(window=20, min_periods=1).mean()
            new_features[f'{col}_std_20'] = df[col].rolling(window=20, min_periods=1).std()
            new_features[f'{col}_return_5'] = df[col].pct_change(periods=5)

    # 2. 读取target_pairs.csv创建相关性特征
    pairs_df = pd.read_csv('target_pairs.csv')

    correlation_features = []
    for idx, row in pairs_df.head(50).iterrows():  # 只用前50个
        pair_str = row['pair'].strip()

        if ' - ' in pair_str:
            asset1, asset2 = pair_str.split(' - ')
            asset1, asset2 = asset1.strip(), asset2.strip()

            if asset1 in df.columns and asset2 in df.columns:
                new_features[f'pair_{idx}_diff'] = df[asset1] - df[asset2]

                corr = df[asset1].rolling(window = 20, min_periods=10).corr(df[asset2])
                new_features[f'pair_{idx}_corr_20'] = corr

    df_final = pd.concat([df, new_features], axis=1)

    df_final.to_csv('train_features_quick.csv', index=False)
    print(f'快速特征工程完成！特征数：{len(df_final.columns)}')

    return df_final


if __name__ == "__main__":
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # 运行快速版本
        df_features = quick_feature_engineering()
    else:
        # 运行完整版本
        df_features = complete_feature_engineering_pipeline()

    print("\n✅ 特征工程完成！")
    print("下一步：使用生成的特征文件进行建模")
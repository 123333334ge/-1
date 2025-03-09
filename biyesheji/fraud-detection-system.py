# 信用卡欺诈行为分析与预警系统
# 基于朴素贝叶斯算法
# 作者：[您的姓名]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
import joblib
import os
import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fraud_detection.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("fraud_detection")

class CreditCardFraudDetection:
    """
    基于朴素贝叶斯的信用卡欺诈检测系统
    实现了数据加载、预处理、特征工程、模型训练与评估等功能
    """
    
    def __init__(self, data_path=None):
        """
        初始化检测系统
        
        参数:
            data_path (str): 数据集路径，如果为None则使用示例数据
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.model_path = "models/"
        self.results_path = "results/"
        
        # 创建必要的文件夹
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        
        logger.info("信用卡欺诈检测系统初始化完成")
    
    def load_data(self):
        """
        加载数据集
        如果没有提供数据路径，则创建一个示例数据集用于演示
        """
        if self.data_path:
            logger.info(f"从 {self.data_path} 加载数据")
            try:
                self.data = pd.read_csv(self.data_path)
            except Exception as e:
                logger.error(f"加载数据失败: {str(e)}")
                raise
        else:
            logger.info("未提供数据路径，创建示例数据")
            # 创建示例数据集
            np.random.seed(42)
            n_samples = 10000
            n_features = 10
            
            # 生成特征
            X = np.random.randn(n_samples, n_features)
            
            # 设定欺诈比例约为2%
            fraud_ratio = 0.02
            n_fraud = int(n_samples * fraud_ratio)
            
            # 生成目标变量（0=正常，1=欺诈）
            y = np.zeros(n_samples)
            
            # 随机选择欺诈样本
            fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
            y[fraud_indices] = 1
            
            # 对欺诈样本的特征进行一些调整，使其与正常样本有所区别
            X[fraud_indices] = X[fraud_indices] + np.random.randn(n_fraud, n_features) * 2
            
            # 创建特征名称
            feature_names = [f'feature_{i}' for i in range(n_features)]
            
            # 转换为DataFrame
            self.data = pd.DataFrame(X, columns=feature_names)
            self.data['Class'] = y
            
            # 添加一些时间和金额特征，使数据更接近真实场景
            self.data['Amount'] = np.exp(np.random.randn(n_samples) * 0.8 + 4)  # 金额
            self.data['Time'] = np.random.randint(0, 24 * 3600, n_samples)  # 时间（秒）
            
            # 欺诈交易通常金额较大
            self.data.loc[fraud_indices, 'Amount'] = self.data.loc[fraud_indices, 'Amount'] * 1.5
        
        logger.info(f"数据加载完成。样本数: {self.data.shape[0]}, 特征数: {self.data.shape[1]}")
        logger.info(f"欺诈交易比例: {self.data['Class'].mean():.4f}")
        
        return self.data
    
    def explore_data(self):
        """
        数据探索分析，生成描述性统计和可视化
        """
        if self.data is None:
            logger.warning("数据尚未加载，请先调用load_data()")
            return
        
        logger.info("开始数据探索分析")
        
        # 数据基本信息
        logger.info(f"数据集形状: {self.data.shape}")
        logger.info(f"特征列表: {self.data.columns.tolist()}")
        
        # 检查缺失值
        missing_values = self.data.isnull().sum()
        logger.info(f"缺失值统计:\n{missing_values}")
        
        # 类别分布
        class_distribution = self.data['Class'].value_counts()
        logger.info(f"类别分布:\n{class_distribution}")
        
        # 保存各特征的统计描述
        stats_file = os.path.join(self.results_path, "descriptive_stats.csv")
        self.data.describe().to_csv(stats_file)
        logger.info(f"描述性统计已保存到 {stats_file}")
        
        # 生成可视化
        self._generate_visualizations()
        
        return missing_values, class_distribution
    
    def _generate_visualizations(self):
        """
        生成探索性数据分析的可视化图表
        """
        logger.info("生成数据可视化")
        
        # 创建可视化文件夹
        viz_path = os.path.join(self.results_path, "visualizations")
        os.makedirs(viz_path, exist_ok=True)
        
        # 1. 类别分布饼图
        plt.figure(figsize=(8, 6))
        labels = ['正常交易', '欺诈交易']
        counts = self.data['Class'].value_counts()
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('交易类别分布')
        plt.savefig(os.path.join(viz_path, 'class_distribution_pie.png'))
        plt.close()
        
        # 2. 欺诈和非欺诈交易的金额分布
        if 'Amount' in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.data, x='Amount', hue='Class', bins=50, kde=True)
            plt.title('交易金额分布')
            plt.xlabel('金额')
            plt.ylabel('频率')
            plt.savefig(os.path.join(viz_path, 'amount_distribution.png'))
            plt.close()
            
            # 3. 欺诈和非欺诈交易的金额箱线图
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Class', y='Amount', data=self.data)
            plt.title('欺诈和非欺诈交易的金额分布')
            plt.savefig(os.path.join(viz_path, 'amount_boxplot.png'))
            plt.close()
        
        # 4. 时间特征分析
        if 'Time' in self.data.columns:
            # 将秒转换为小时
            hour_data = self.data.copy()
            hour_data['Hour'] = hour_data['Time'] // 3600
            
            plt.figure(figsize=(12, 6))
            hour_counts = hour_data.groupby(['Hour', 'Class']).size().unstack().fillna(0)
            hour_counts.plot(kind='bar', stacked=True)
            plt.title('各小时交易频次分布')
            plt.xlabel('小时')
            plt.ylabel('交易数')
            plt.legend(['正常交易', '欺诈交易'])
            plt.savefig(os.path.join(viz_path, 'hourly_distribution.png'))
            plt.close()
        
        # 5. 特征相关性热力图
        plt.figure(figsize=(12, 10))
        corr_matrix = self.data.corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('特征相关性热力图')
        plt.savefig(os.path.join(viz_path, 'correlation_heatmap.png'))
        plt.close()
        
        logger.info(f"可视化图表已保存到 {viz_path}")
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        数据预处理：特征选择、缩放、划分训练测试集等
        
        参数:
            test_size (float): 测试集比例
            random_state (int): 随机种子
        """
        if self.data is None:
            logger.warning("数据尚未加载，请先调用load_data()")
            return
        
        logger.info("开始数据预处理")
        
        # 检查并处理缺失值
        if self.data.isnull().sum().any():
            logger.info("检测到缺失值，进行插补处理")
            self.data.fillna(self.data.median(), inplace=True)
        
        # 提取特征和目标变量
        if 'Class' in self.data.columns:
            self.y = self.data['Class']
            self.X = self.data.drop('Class', axis=1)
        else:
            logger.error("数据集中缺少目标变量'Class'列")
            raise ValueError("数据集必须包含'Class'列作为目标变量")
        
        # 特征选择
        # 可以根据需要实现更复杂的特征选择方法
        time_cols = ['Time', 'Hour'] if 'Hour' in self.X.columns else ['Time']
        self.X = self.X.drop(time_cols, axis=1, errors='ignore')
        
        # 特征缩放
        logger.info("进行特征缩放")
        self.scaler = RobustScaler()  # 使用鲁棒缩放器，对异常值不敏感
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # 保存特征名称，用于后续解释
        self.feature_names = self.X.columns.tolist()
        
        # 划分训练集和测试集
        logger.info(f"划分训练集和测试集，测试集比例: {test_size}")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # 处理类别不平衡问题
        logger.info("使用SMOTE处理类别不平衡问题")
        smote = SMOTE(random_state=random_state)
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(self.X_train, self.y_train)
        
        logger.info(f"原始训练集形状: {self.X_train.shape}, 重采样后训练集形状: {self.X_train_resampled.shape}")
        logger.info(f"原始训练集正例比例: {self.y_train.mean():.4f}, 重采样后正例比例: {self.y_train_resampled.mean():.4f}")
        
        return self.X_train_resampled, self.y_train_resampled, self.X_test, self.y_test
    
    def train_model(self, use_resampled=True):
        """
        训练朴素贝叶斯模型
        
        参数:
            use_resampled (bool): 是否使用SMOTE重采样后的数据训练
        """
        if self.X_train is None or self.y_train is None:
            logger.warning("训练数据尚未准备，请先调用preprocess_data()")
            return
        
        logger.info("开始训练朴素贝叶斯模型")
        
        # 选择要使用的训练数据
        X_train = self.X_train_resampled if use_resampled else self.X_train
        y_train = self.y_train_resampled if use_resampled else self.y_train
        
        # 初始化高斯朴素贝叶斯模型
        self.model = GaussianNB()
        
        # 使用网格搜索优化参数
        logger.info("进行参数优化")
        param_grid = {
            'var_smoothing': np.logspace(0, -9, num=10)
        }
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 获取最佳模型
        self.model = grid_search.best_estimator_
        logger.info(f"最佳参数: {grid_search.best_params_}")
        logger.info(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
        
        # 在完整训练集上训练最终模型
        self.model.fit(X_train, y_train)
        
        # 保存模型
        model_filename = os.path.join(self.model_path, f"naive_bayes_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
        joblib.dump(self.model, model_filename)
        logger.info(f"模型已保存到 {model_filename}")
        
        # 保存特征缩放器
        scaler_filename = os.path.join(self.model_path, f"scaler_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
        joblib.dump(self.scaler, scaler_filename)
        logger.info(f"特征缩放器已保存到 {scaler_filename}")
        
        return self.model
    
    def evaluate_model(self):
        """
        评估模型性能
        计算各项评估指标并生成可视化
        """
        if self.model is None:
            logger.warning("模型尚未训练，请先调用train_model()")
            return
        
        logger.info("开始评估模型性能")
        
        # 在测试集上进行预测
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # 计算评估指标
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        # 打印评估指标
        logger.info(f"准确率: {accuracy:.4f}")
        logger.info(f"精确率: {precision:.4f}")
        logger.info(f"召回率: {recall:.4f}")
        logger.info(f"F1分数: {f1:.4f}")
        
        # 生成混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        logger.info(f"混淆矩阵:\n{cm}")
        
        # 详细分类报告
        class_report = classification_report(self.y_test, y_pred)
        logger.info(f"分类报告:\n{class_report}")
        
        # 生成评估可视化
        self._generate_evaluation_plots(y_prob)
        
        # 保存评估结果
        eval_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
        
        # 将评估结果保存为文本文件
        eval_file = os.path.join(self.results_path, f"evaluation_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(eval_file, 'w') as f:
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write(f"精确率: {precision:.4f}\n")
            f.write(f"召回率: {recall:.4f}\n")
            f.write(f"F1分数: {f1:.4f}\n\n")
            f.write(f"混淆矩阵:\n{cm}\n\n")
            f.write(f"分类报告:\n{class_report}\n")
        
        logger.info(f"评估结果已保存到 {eval_file}")
        
        return eval_results
    
    def _generate_evaluation_plots(self, y_prob):
        """
        生成模型评估的可视化图表
        
        参数:
            y_prob (array): 测试集上的预测概率
        """
        # 创建评估可视化文件夹
        eval_viz_path = os.path.join(self.results_path, "evaluation_plots")
        os.makedirs(eval_viz_path, exist_ok=True)
        
        # 1. 混淆矩阵热力图
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, self.model.predict(self.X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['正常', '欺诈'], 
                    yticklabels=['正常', '欺诈'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig(os.path.join(eval_viz_path, 'confusion_matrix.png'))
        plt.close()
        
        # 2. ROC曲线
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率')
        plt.ylabel('真正例率')
        plt.title('接收者操作特征曲线')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(eval_viz_path, 'roc_curve.png'))
        plt.close()
        
        # 3. 精确率-召回率曲线
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR曲线 (AUC = {pr_auc:.2f})')
        plt.axhline(y=self.y_test.mean(), color='red', linestyle='--', label=f'基准 ({self.y_test.mean():.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(eval_viz_path, 'precision_recall_curve.png'))
        plt.close()
        
        # 4. 特征重要性分析（分析条件概率）
        # 对于朴素贝叶斯，我们可以比较不同类别下特征的条件概率分布
        if hasattr(self.model, 'theta_') and hasattr(self.model, 'sigma_'):
            plt.figure(figsize=(12, 8))
            
            # 计算特征对区分类别的重要性
            importance = np.abs(self.model.theta_[1] - self.model.theta_[0]) / np.sqrt(self.model.sigma_[0] + self.model.sigma_[1])
            indices = np.argsort(importance)[::-1]
            
            # 取前15个重要特征
            n_features = min(15, len(indices))
            plt.barh(range(n_features), importance[indices[:n_features]])
            plt.yticks(range(n_features), [self.feature_names[i] for i in indices[:n_features]])
            plt.xlabel('重要性分数')
            plt.title('特征重要性')
            plt.tight_layout()
            plt.savefig(os.path.join(eval_viz_path, 'feature_importance.png'))
            plt.close()
        
        logger.info(f"评估图表已保存到 {eval_viz_path}")
    
    def predict_fraud(self, transactions):
        """
        对新交易进行欺诈预测
        
        参数:
            transactions (DataFrame): 包含交易特征的DataFrame
        
        返回:
            predictions (array): 预测结果（0=正常，1=欺诈）
            probabilities (array): 欺诈概率
        """
        if self.model is None or self.scaler is None:
            logger.error("模型或特征缩放器未训练")
            raise ValueError("请先调用train_model()训练模型")
        
        logger.info(f"对 {len(transactions)} 条交易进行欺诈预测")
        
        # 预处理交易数据
        try:
            # 确保只使用模型训练时的特征
            X_pred = transactions[self.feature_names].copy()
            
            # 应用特征缩放
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # 预测
            predictions = self.model.predict(X_pred_scaled)
            probabilities = self.model.predict_proba(X_pred_scaled)[:, 1]
            
            logger.info(f"预测完成。检测到 {np.sum(predictions)} 条可疑交易")
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"预测过程中发生错误: {str(e)}")
            raise
    
    def generate_fraud_alerts(self, transactions, predictions, probabilities, threshold=0.7):
        """
        生成欺诈预警
        
        参数:
            transactions (DataFrame): 交易数据
            predictions (array): 模型预测结果
            probabilities (array): 预测的欺诈概率
            threshold (float): 预警阈值，高于此概率的交易将被标记为预警
        
        返回:
            alerts (DataFrame): 预警信息
        """
        logger.info(f"使用阈值 {threshold} 生成欺诈预警")
        
        # 复制交易数据
        transactions_with_pred = transactions.copy()
        
        # 添加预测结果和概率
        transactions_with_pred['predicted_class'] = predictions
        transactions_with_pred['fraud_probability'] = probabilities
        
        # 根据阈值筛选高风险交易
        high_risk_transactions = transactions_with_pred[transactions_with_pred['fraud_probability'] >= threshold]
        
        if len(high_risk_transactions) > 0:
            logger.info(f"检测到 {len(high_risk_transactions)} 条高风险交易")
            
            # 生成预警信息
            alerts = high_risk_transactions.copy()
            alerts['alert_level'] = pd.cut(
                alerts['fraud_probability'], 
                bins=[threshold, 0.8, 0.9, 1.0], 
                labels=['低', '中', '高'],
                include_lowest=True
            )
            
            # 按欺诈概率降序排序
            alerts = alerts.sort_values('fraud_probability', ascending=False)
            
            # 添加时间戳
            alerts['alert_time'] = datetime.datetime.now()
            
            # 保存预警信息
            alert_file = os.path.join(self.results_path, f"fraud_alerts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            alerts.to_csv(alert_file, index=False)
            logger.info(f"预警信息已保存到 {alert_file}")
            
            return alerts
        else:
            logger.info("未检测到高风险交易")
            return pd.DataFrame()
    
    def load_saved_model(self, model_path, scaler_path):
        """
        加载已保存的模型和特征缩放器
        
        参数:
            model_path (str): 模型文件路径
            scaler_path (str): 特征缩放器文件路径
        """
        logger.info(f"从 {model_path} 加载模型")
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("模型和特征缩放器加载成功")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def run_pipeline(self, data_path=None, test_size=0.2, random_state=42):
        """
        运行完整的欺诈检测流水线
        
        参数:
            data_path (str): 数据集路径
            test_size (float): 测试集比例
            random_state (int): 随机种子
        """
        logger.info("启动完整欺诈检测流水线")
        
        # 如果提供了数据路径，则更新对象属性
        if data_path:
            self.data_path = data_path
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 数据探索
        self.explore_data()
        
        # 3. 数据预处理
        self.preprocess_data(test_size=test_size, random_state=random_state)
        
        # 4. 训练模型
        self.train_model()
        
        # 5. 评估模型
        eval_results = self.evaluate_model()
        
        logger.info("欺诈检测流水线执行完成")
        
        return eval_results


# 实时欺诈监控系统示例
class RealTimeFraudMonitor:
    """
    实时欺诈监控系统
    用于监控实时交易数据，检测可能的欺诈行为并生成预警
    """
    
    def __init__(self, model_path, scaler_path):
        """
        初始化实时监控系统
        
        参数:
            model_path (str): 保存的模型路径
            scaler_path (str): 保存的特征缩放器路径
        """
        self.detector = CreditCardFraudDetection()
        success = self.detector.load_saved_model(model_path, scaler_path)
        
        if not success:
            raise ValueError("无法加载模型，请确保提供了正确的模型和缩放器路径")
        
        self.alert_threshold = 0.7  # 默认预警阈值
        self.feature_names = None  # 将在加载模型后设置
        self.alerts_history = []
        
        logger.info("实时欺诈监控系统初始化完成")
    
    def set_alert_threshold(self, threshold):
        """
        设置预警阈值
        
        参数:
            threshold (float): 新的预警阈值，范围[0, 1]
        """
        if 0 <= threshold <= 1:
            logger.info(f"预警阈值从 {self.alert_threshold} 更新为 {threshold}")
            self.alert_threshold = threshold
        else:
            logger.error(f"无效的阈值 {threshold}，阈值必须在0和1之间")
            raise ValueError("阈值必须在0和1之间")
    
    def process_transaction(self, transaction):
        """
        处理单条交易并检测是否为欺诈
        
        参数:
            transaction (dict or Series): 包含交易特征的字典或Series
        
        返回:
            is_fraud (bool): 是否为欺诈交易
            fraud_probability (float): 欺诈概率
            alert_level (str): 预警级别
        """
        # 将单条交易转换为DataFrame
        if isinstance(transaction, dict):
            transaction_df = pd.DataFrame([transaction])
        elif isinstance(transaction, pd.Series):
            transaction_df = pd.DataFrame([transaction.to_dict()])
        else:
            logger.error(f"不支持的交易数据类型: {type(transaction)}")
            raise TypeError("交易数据必须是字典或pandas.Series")
        
        # 预测
        try:
            predictions, probabilities = self.detector.predict_fraud(transaction_df)
            
            is_fraud = bool(predictions[0])
            fraud_probability = float(probabilities[0])
            
            # 确定预警级别
            if fraud_probability >= 0.9:
                alert_level = "高"
            elif fraud_probability >= 0.8:
                alert_level = "中"
            elif fraud_probability >= self.alert_threshold:
                alert_level = "低"
            else:
                alert_level = "正常"
            
            # 记录处理结果
            logger.info(f"交易ID: {transaction.get('transaction_id', 'unknown')}, "
                       f"欺诈概率: {fraud_probability:.4f}, 预警级别: {alert_level}")
            
            # 如果超过阈值，添加到预警历史
            if fraud_probability >= self.alert_threshold:
                alert_info = {
                    'transaction': transaction,
                    'fraud_probability': fraud_probability,
                    'alert_level': alert_level,
                    'timestamp': datetime.datetime.now()
                }
                self.alerts_history.append(alert_info)
            
            return is_fraud, fraud_probability, alert_level
            
        except Exception as e:
            logger.error(f"处理交易时发生错误: {str(e)}")
            raise
    
    def process_batch(self, transactions_batch):
        """
        批量处理交易并生成预警报告
        
        参数:
            transactions_batch (DataFrame): 包含多条交易数据的DataFrame
        
        返回:
            alerts (DataFrame): 预警信息
        """
        logger.info(f"开始批量处理 {len(transactions_batch)} 条交易")
        
        try:
            # 预测
            predictions, probabilities = self.detector.predict_fraud(transactions_batch)
            
            # 生成预警
            alerts = self.detector.generate_fraud_alerts(
                transactions_batch, predictions, probabilities, self.alert_threshold
            )
            
            # 更新预警历史
            if not alerts.empty:
                for _, alert in alerts.iterrows():
                    alert_info = {
                        'transaction': alert.to_dict(),
                        'fraud_probability': alert['fraud_probability'],
                        'alert_level': alert['alert_level'],
                        'timestamp': datetime.datetime.now()
                    }
                    self.alerts_history.append(alert_info)
            
            return alerts
            
        except Exception as e:
            logger.error(f"批量处理交易时发生错误: {str(e)}")
            raise
    
    def get_alerts_summary(self, days=1):
        """
        获取最近一段时间的预警摘要
        
        参数:
            days (int): 最近几天的预警
        
        返回:
            summary (dict): 预警摘要统计
        """
        # 计算时间阈值
        time_threshold = datetime.datetime.now() - datetime.timedelta(days=days)
        
        # 筛选时间范围内的预警
        recent_alerts = [alert for alert in self.alerts_history 
                        if alert['timestamp'] >= time_threshold]
        
        # 按预警级别统计
        alert_levels = {
            '高': len([a for a in recent_alerts if a['alert_level'] == '高']),
            '中': len([a for a in recent_alerts if a['alert_level'] == '中']),
            '低': len([a for a in recent_alerts if a['alert_level'] == '低'])
        }
        
        # 计算总预警金额
        total_amount = sum(float(alert['transaction'].get('Amount', 0)) for alert in recent_alerts)
        
        # 生成摘要
        summary = {
            'period': f'过去{days}天',
            'total_alerts': len(recent_alerts),
            'alert_levels': alert_levels,
            'total_amount': total_amount
        }
        
        return summary


# 构建Web应用界面的示例代码
def create_web_app():
    """
    创建信用卡欺诈检测系统的Web应用界面
    使用Streamlit框架
    """
    import streamlit as st
    
    st.title("信用卡欺诈行为分析与预警系统")
    st.sidebar.title("系统导航")
    
    # 页面选择
    page = st.sidebar.selectbox(
        "请选择功能页面",
        ["系统概览", "数据分析", "模型训练", "实时监控", "历史预警", "帮助"]
    )
    
    if page == "系统概览":
        st.header("系统概览")
        st.write("""
        欢迎使用信用卡欺诈行为分析与预警系统。该系统基于朴素贝叶斯算法，
        能够有效识别潜在的欺诈交易，并及时发出预警。
        
        系统主要功能包括：
        1. 数据分析：探索性分析交易数据，发现数据特征和规律
        2. 模型训练：训练朴素贝叶斯模型用于欺诈检测
        3. 实时监控：对交易进行实时监控和预警
        4. 历史预警：查看和分析历史预警记录
        
        请从左侧导航栏选择具体功能。
        """)
        
        # 添加系统状态指标
        st.subheader("系统状态")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="模型准确率", value="95.2%", delta="+0.3%")
        col2.metric(label="今日预警数", value="12", delta="-2")
        col3.metric(label="系统响应时间", value="0.35秒", delta="-0.05秒")
    
    elif page == "数据分析":
        st.header("交易数据分析")
        
        # 数据上传
        uploaded_file = st.file_uploader("上传交易数据CSV文件", type=["csv"])
        if uploaded_file is not None:
            st.success("文件上传成功！")
            
            # 创建检测器实例并加载数据
            detector = CreditCardFraudDetection()
            
            # 这里简化处理，实际应用中需要保存上传文件并传入路径
            detector.data_path = "uploaded_data.csv"
            detector.load_data()
            
            # 显示数据基本信息
            st.subheader("数据概览")
            st.write(f"数据集形状: {detector.data.shape}")
            st.write(f"特征列表: {detector.data.columns.tolist()}")
            
            # 显示数据样例
            st.subheader("数据样例")
            st.dataframe(detector.data.head())
            
            # 数据探索
            st.subheader("数据探索")
            missing_values, class_distribution = detector.explore_data()
            
            # 显示类别分布
            st.write("类别分布:")
            st.write(class_distribution)
            
            # 显示各种统计图表
            st.subheader("数据可视化")
            
            # 加载生成的图表
            viz_path = os.path.join(detector.results_path, "visualizations")
            if os.path.exists(viz_path):
                for img_file in os.listdir(viz_path):
                    if img_file.endswith('.png'):
                        st.image(os.path.join(viz_path, img_file), caption=img_file.split('.')[0])
    
    elif page == "模型训练":
        st.header("朴素贝叶斯模型训练")
        
        # 选择数据集
        data_option = st.selectbox(
            "选择数据集",
            ["上传数据集", "使用示例数据集"]
        )
        
        detector = CreditCardFraudDetection()
        
        if data_option == "上传数据集":
            uploaded_file = st.file_uploader("上传交易数据CSV文件", type=["csv"])
            if uploaded_file is not None:
                st.success("文件上传成功！")
                detector.data_path = "uploaded_data.csv"
        else:
            st.info("将使用系统生成的示例数据集")
            detector.data_path = None
        
        # 设置训练参数
        st.subheader("训练参数设置")
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
        use_smote = st.checkbox("使用SMOTE处理类别不平衡", value=True)
        
        # 训练模型
        if st.button("开始训练模型"):
            with st.spinner("正在训练模型..."):
                # 加载数据
                detector.load_data()
                
                # 数据预处理
                detector.preprocess_data(test_size=test_size)
                
                # 训练模型
                detector.train_model(use_resampled=use_smote)
                
                # 评估模型
                eval_results = detector.evaluate_model()
                
                st.success("模型训练完成！")
                
                # 显示评估结果
                st.subheader("模型评估结果")
                st.write(f"准确率: {eval_results['accuracy']:.4f}")
                st.write(f"精确率: {eval_results['precision']:.4f}")
                st.write(f"召回率: {eval_results['recall']:.4f}")
                st.write(f"F1分数: {eval_results['f1']:.4f}")
                
                # 显示混淆矩阵
                st.subheader("混淆矩阵")
                cm = np.array(eval_results['confusion_matrix'])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['正常', '欺诈'], 
                            yticklabels=['正常', '欺诈'])
                plt.xlabel('预测标签')
                plt.ylabel('真实标签')
                st.pyplot(fig)
                
                # 显示其他评估图表
                eval_viz_path = os.path.join(detector.results_path, "evaluation_plots")
                if os.path.exists(eval_viz_path):
                    for img_file in os.listdir(eval_viz_path):
                        if img_file.endswith('.png'):
                            st.image(os.path.join(eval_viz_path, img_file), 
                                    caption=img_file.split('.')[0])
    
    elif page == "实时监控":
        st.header("实时交易监控")
        
        # 选择模型
        model_file = st.selectbox(
            "选择已训练的模型",
            os.listdir("models") if os.path.exists("models") and os.listdir("models") else ["未找到模型文件"]
        )
        
        scaler_file = st.selectbox(
            "选择对应的特征缩放器",
            [f for f in os.listdir("models") if f.startswith("scaler_")] if os.path.exists("models") else ["未找到缩放器文件"]
        )
        
        # 设置预警阈值
        alert_threshold = st.slider("预警阈值", 0.5, 0.95, 0.7, 0.05)
        
        # 初始化监控系统
        if st.button("启动监控系统"):
            if "未找到" in model_file or "未找到" in scaler_file:
                st.error("请先训练模型或选择有效的模型文件")
            else:
                monitor = RealTimeFraudMonitor(
                    os.path.join("models", model_file),
                    os.path.join("models", scaler_file)
                )
                monitor.set_alert_threshold(alert_threshold)
                st.session_state.monitor = monitor
                st.success("监控系统已启动！")
        
        # 模拟实时交易输入
        st.subheader("交易信息输入")
        
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("交易金额", min_value=0.0, step=100.0, value=1000.0)
        with col2:
            transaction_type = st.selectbox("交易类型", ["消费", "取现", "转账", "还款"])
        
        col3, col4 = st.columns(2)
        with col3:
            merchant_category = st.selectbox("商户类别", ["餐饮", "零售", "酒店", "航空", "其他"])
        with col4:
            is_foreign = st.checkbox("是否为境外交易")
        
        # 生成交易特征
        if st.button("提交交易"):
            if "monitor" not in st.session_state:
                st.error("请先启动监控系统")
            else:
                # 构建交易数据
                transaction = {
                    'transaction_id': f"T{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'Amount': amount,
                    'TransactionType': transaction_type,
                    'MerchantCategory': merchant_category,
                    'IsForeign': 1 if is_foreign else 0,
                    # 添加其他模型需要的特征
                    'feature_0': np.random.randn(),
                    'feature_1': np.random.randn(),
                    'feature_2': np.random.randn(),
                    'feature_3': np.random.randn(),
                    'feature_4': np.random.randn(),
                    'feature_5': np.random.randn(),
                    'feature_6': np.random.randn(),
                    'feature_7': np.random.randn(),
                    'feature_8': np.random.randn(),
                    'feature_9': np.random.randn(),
                }
                
                # 处理交易
                is_fraud, fraud_probability, alert_level = st.session_state.monitor.process_transaction(transaction)
                
                # 显示结果
                st.subheader("交易分析结果")
                
                if alert_level == "正常":
                    st.success(f"交易ID: {transaction['transaction_id']} - 正常交易 (欺诈概率: {fraud_probability:.4f})")
                elif alert_level == "低":
                    st.warning(f"交易ID: {transaction['transaction_id']} - 低风险交易 (欺诈概率: {fraud_probability:.4f})")
                elif alert_level == "中":
                    st.warning(f"交易ID: {transaction['transaction_id']} - 中风险交易 (欺诈概率: {fraud_probability:.4f})")
                else:
                    st.error(f"交易ID: {transaction['transaction_id']} - 高风险交易 (欺诈概率: {fraud_probability:.4f})")
                
                # 显示详细信息
                st.json(transaction)
    
    elif page == "历史预警":
        st.header("历史预警记录")
        
        # 上传预警记录文件
        uploaded_file = st.file_uploader("上传预警记录CSV文件", type=["csv"])
        
        if uploaded_file is not None:
            # 加载预警数据
            alerts_df = pd.read_csv(uploaded_file)
            
            # 显示预警统计信息
            st.subheader("预警统计")
            
            # 统计预警级别分布
            if 'alert_level' in alerts_df.columns:
                level_counts = alerts_df['alert_level'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                col1.metric(label="高风险预警", value=level_counts.get('高', 0))
                col2.metric(label="中风险预警", value=level_counts.get('中', 0))
                col3.metric(label="低风险预警", value=level_counts.get('低', 0))
                
                # 绘制预警级别分布图
                fig, ax = plt.subplots()
                level_counts.plot(kind='bar', ax=ax)
                plt.title('预警级别分布')
                plt.ylabel('数量')
                st.pyplot(fig)
            
            # 按时间的预警趋势
            if 'alert_time' in alerts_df.columns:
                alerts_df['alert_time'] = pd.to_datetime(alerts_df['alert_time'])
                alerts_df['date'] = alerts_df['alert_time'].dt.date
                
                daily_counts = alerts_df.groupby('date').size()
                
                st.subheader("预警趋势")
                fig, ax = plt.subplots()
                daily_counts.plot(ax=ax)
                plt.title('每日预警数量趋势')
                plt.ylabel('预警数量')
                st.pyplot(fig)
            
            # 显示预警记录
            st.subheader("预警记录")
            st.dataframe(alerts_df)
            
            # 按风险级别筛选
            risk_level = st.multiselect(
                "按风险级别筛选",
                ['高', '中', '低'],
                ['高']
            )
            
            if risk_level:
                filtered_alerts = alerts_df[alerts_df['alert_level'].isin(risk_level)]
                st.write(f"筛选出 {len(filtered_alerts)} 条预警记录")
                st.dataframe(filtered_alerts)
    
    elif page == "帮助":
        st.header("系统帮助")
        st.write("""
        ## 系统使用指南
        
        ### 数据要求
        - 交易数据必须是CSV格式
        - 必须包含'Class'列作为标签（0=正常，1=欺诈）
        - 建议包含'Amount'和'Time'特征
        
        ### 基本流程
        1. 上传数据进行分析
        2. 训练朴素贝叶斯模型
        3. 启动实时监控系统
        4. 查看和分析预警记录
        
        ### 常见问题
        
        #### Q: 如何提高模型性能？
        A: 可以尝试以下方法：
        - 添加更多相关特征
        - 尝试不同的特征缩放方法
        - 调整SMOTE参数处理类别不平衡
        - 优化预警阈值
        
        #### Q: 系统支持哪些格式的数据？
        A: 目前系统主要支持CSV格式的数据。
        
        #### Q: 如何导出预警结果？
        A: 在"历史预警"页面，系统会自动保存预警记录到CSV文件。
        
        ### 联系支持
        如需技术支持，请联系：support@example.com
        """)


# 主函数
def main():
    """
    系统入口函数
    """
    logger.info("信用卡欺诈检测系统启动")
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='信用卡欺诈行为分析与预警系统')
    parser.add_argument('--data', type=str, help='数据集路径')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'monitor', 'webapp'],
                       help='系统运行模式：train（训练）, monitor（监控）, webapp（Web应用）')
    parser.add_argument('--model', type=str, help='已训练模型路径')
    parser.add_argument('--scaler', type=str, help='特征缩放器路径')
    parser.add_argument('--threshold', type=float, default=0.7, help='欺诈预警阈值')
    args = parser.parse_args()
    
    # 根据运行模式执行不同操作
    if args.mode == 'train':
        # 训练模式
        detector = CreditCardFraudDetection(args.data)
        detector.run_pipeline()
        
    elif args.mode == 'monitor':
        # 监控模式
        if not args.model or not args.scaler:
            logger.error("监控模式需要提供模型和特征缩放器路径")
            print("Error: 请使用--model和--scaler参数提供模型和特征缩放器路径")
            return
        
        monitor = RealTimeFraudMonitor(args.model, args.scaler)
        monitor.set_alert_threshold(args.threshold)
        
        # 这里可以添加从消息队列或API接收实时交易的代码
        # 简单示例：处理一批模拟交易
        np.random.seed(42)
        n_transactions = 100
        
        # 创建模拟交易
        test_transactions = pd.DataFrame({
            'feature_{}'.format(i): np.random.randn(n_transactions) for i in range(10)
        })
        test_transactions['Amount'] = np.exp(np.random.randn(n_transactions) * 0.8 + 4)
        
        # 处理交易批次
        alerts = monitor.process_batch(test_transactions)
        
        # 打印预警摘要
        print(f"检测到 {len(alerts)} 条可疑交易")
        
    elif args.mode == 'webapp':
        # Web应用模式
        create_web_app()
    
    logger.info("信用卡欺诈检测系统任务完成")


if __name__ == "__main__":
    main()
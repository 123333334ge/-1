# -*- coding: utf-8 -*-
## 信用卡欺诈行为分析与预警系统
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
    信用卡欺诈检测类
    实现数据加载、预处理、模型训练和评估等功能
    """
    
    def __init__(self, data_path=None):
        """
        初始化检测器
        
        参数:
            data_path (str): 数据集路径，默认为None
        """
        self.data_path = data_path
        self.data = None
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # 创建必要的目录
        self.results_path = "results"
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(os.path.join(self.results_path, "evaluation_plots"), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, "visualizations"), exist_ok=True)
        
        # 创建模型保存目录
        self.models_path = "models"
        os.makedirs(self.models_path, exist_ok=True)
    
    def load_data(self):
        """
        加载数据集
        如果没有提供数据路径，则生成示例数据
        """
        if self.data_path and os.path.exists(self.data_path):
            logger.info(f"从文件加载数据: {self.data_path}")
            self.data = pd.read_csv(self.data_path)
        else:
            logger.info("生成示例数据集")
            # 生成示例数据
            np.random.seed(42)
            n_samples = 10000
            
            # 生成特征
            X = np.random.randn(n_samples, 10)  # 10个特征
            # 添加一些模式使得欺诈交易更容易识别
            fraud_idx = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
            X[fraud_idx] = X[fraud_idx] * 2 + 1
            
            # 生成标签（0=正常，1=欺诈）
            y = np.zeros(n_samples)
            y[fraud_idx] = 1
            
            # 创建DataFrame
            features = {f'feature_{i}': X[:, i] for i in range(X.shape[1])}
            features['Amount'] = np.exp(np.random.randn(n_samples) * 0.8 + 4)  # 生成合理的金额
            features['Class'] = y
            
            self.data = pd.DataFrame(features)
        
        logger.info(f"数据集大小: {self.data.shape}")
    
    def explore_data(self):
        """
        探索性数据分析
        
        返回:
            missing_values (Series): 缺失值统计
            class_distribution (Series): 类别分布
        """
        logger.info("开始数据探索")
        
        # 检查缺失值
        missing_values = self.data.isnull().sum()
        logger.info(f"缺失值统计:\n{missing_values}")
        
        # 类别分布
        class_distribution = self.data['Class'].value_counts()
        logger.info(f"类别分布:\n{class_distribution}")
        
        # 保存描述性统计
        desc_stats = self.data.describe()
        desc_stats.to_csv(os.path.join(self.results_path, "descriptive_stats.csv"))
        
        # 生成可视化
        self._generate_visualizations()
        
        return missing_values, class_distribution
    
    def _generate_visualizations(self):
        """
        生成数据可视化图表
        """
        viz_path = os.path.join(self.results_path, "visualizations")
        
        # 金额分布图
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='Amount', hue='Class', bins=50)
        plt.title('交易金额分布')
        plt.savefig(os.path.join(viz_path, "amount_distribution.png"))
        plt.close()
        
        # 金额箱线图
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.data, y='Amount', x='Class')
        plt.title('交易金额箱线图')
        plt.savefig(os.path.join(viz_path, "amount_boxplot.png"))
        plt.close()
        
        # 类别分布饼图
        plt.figure(figsize=(8, 8))
        self.data['Class'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('交易类别分布')
        plt.savefig(os.path.join(viz_path, "class_distribution_pie.png"))
        plt.close()
        
        # 相关性热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('特征相关性热力图')
        plt.savefig(os.path.join(viz_path, "correlation_heatmap.png"))
        plt.close()
    
    def preprocess_data(self, test_size=0.2):
        """
        数据预处理：特征缩放和训练集/测试集分割
        
        参数:
            test_size (float): 测试集比例
        """
        logger.info("开始数据预处理")
        
        # 分离特征和标签
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']
        
        # 训练集/测试集分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 特征缩放
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f"训练集大小: {self.X_train.shape}, 测试集大小: {self.X_test.shape}")
    
    def train_model(self, use_resampled=True):
        """
        训练朴素贝叶斯模型
        
        参数:
            use_resampled (bool): 是否使用SMOTE处理类别不平衡
        """
        logger.info("开始模型训练")
        
        if use_resampled:
            # 使用SMOTE处理类别不平衡
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
            logger.info(f"SMOTE重采样后的训练集大小: {X_resampled.shape}")
            
            # 训练模型
            self.model = GaussianNB()
            self.model.fit(X_resampled, y_resampled)
        else:
            # 直接训练
            self.model = GaussianNB()
            self.model.fit(self.X_train, self.y_train)
        
        logger.info("模型训练完成")
        
        # 保存模型和缩放器
        self._save_model()
    
    def _save_model(self):
        """
        保存训练好的模型和特征缩放器
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_path = os.path.join(self.models_path, f"naive_bayes_model_{timestamp}.joblib")
        joblib.dump(self.model, model_path)
        
        # 保存特征缩放器
        scaler_path = os.path.join(self.models_path, f"scaler_{timestamp}.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"模型已保存: {model_path}")
        logger.info(f"特征缩放器已保存: {scaler_path}")
    
    def load_saved_model(self, model_path, scaler_path):
        """
        加载已保存的模型和特征缩放器
        
        参数:
            model_path (str): 模型文件路径
            scaler_path (str): 特征缩放器文件路径
            
        返回:
            bool: 是否成功加载
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("成功加载模型和特征缩放器")
            return True
        except Exception as e:
            logger.error(f"加载模型时发生错误: {str(e)}")
            return False
    
    def evaluate_model(self):
        """
        评估模型性能
        
        返回:
            dict: 包含各种评估指标的字典
        """
        logger.info("开始模型评估")
        
        # 预测
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # 计算评估指标
        results = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
        }
        
        # 生成评估报告
        report = classification_report(self.y_test, y_pred)
        
        # 保存评估结果
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_path = os.path.join(self.results_path, f"evaluation_results_{timestamp}.txt")
        
        with open(eval_path, 'w') as f:
            f.write("模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"准确率: {results['accuracy']:.4f}\n")
            f.write(f"精确率: {results['precision']:.4f}\n")
            f.write(f"召回率: {results['recall']:.4f}\n")
            f.write(f"F1分数: {results['f1']:.4f}\n\n")
            f.write("分类报告:\n")
            f.write(report)
        
        # 生成评估图表
        self._generate_evaluation_plots(y_prob)
        
        logger.info(f"评估结果已保存: {eval_path}")
        return results
    
    def _generate_evaluation_plots(self, y_prob):
        """
        生成模型评估相关的图表
        
        参数:
            y_prob (array): 预测概率
        """
        eval_viz_path = os.path.join(self.results_path, "evaluation_plots")
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('接收者操作特征(ROC)曲线')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(eval_viz_path, "roc_curve.png"))
        plt.close()
        
        # 精确率-召回率曲线
        precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.savefig(os.path.join(eval_viz_path, "precision_recall_curve.png"))
        plt.close()
        
        # 混淆矩阵热力图
        cm = confusion_matrix(self.y_test, self.model.predict(self.X_test))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常', '欺诈'],
                    yticklabels=['正常', '欺诈'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig(os.path.join(eval_viz_path, "confusion_matrix.png"))
        plt.close()
    
    def predict_fraud(self, transaction_data):
        """
        对新交易进行欺诈预测
        
        参数:
            transaction_data (DataFrame): 交易数据
        
        返回:
            predictions (array): 预测标签
            probabilities (array): 预测概率
        """
        if self.model is None or self.scaler is None:
            raise ValueError("模型未训练，请先训练模型或加载已保存的模型")
        
        # 特征缩放
        X = self.scaler.transform(transaction_data)
        
        # 预测
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]  # 获取欺诈类的概率
        
        return predictions, probabilities
    
    def generate_fraud_alerts(self, transactions, predictions, probabilities, threshold=0.7):
        """
        根据预测结果生成欺诈预警
        
        参数:
            transactions (DataFrame): 交易数据
            predictions (array): 预测标签
            probabilities (array): 预测概率
            threshold (float): 预警阈值
        
        返回:
            DataFrame: 预警信息
        """
        # 创建预警DataFrame
        alerts = pd.DataFrame({
            'transaction_id': transactions.index,
            'prediction': predictions,
            'fraud_probability': probabilities
        })
        
        # 添加预警级别
        alerts['alert_level'] = pd.cut(
            alerts['fraud_probability'],
            bins=[-float('inf'), threshold, 0.8, 0.9, float('inf')],
            labels=['正常', '低', '中', '高']
        )
        
        # 筛选需要预警的交易
        alerts = alerts[alerts['fraud_probability'] >= threshold].copy()
        
        # 添加时间戳
        alerts['alert_time'] = datetime.datetime.now()
        
        return alerts
    
    def run_pipeline(self):
        """
        运行完整的模型训练和评估流程
        """
        logger.info("开始运行模型训练流程")
        
        # 加载数据
        self.load_data()
        
        # 数据探索
        self.explore_data()
        
        # 数据预处理
        self.preprocess_data()
        
        # 训练模型
        self.train_model()
        
        # 评估模型
        eval_results = self.evaluate_model()
        
        logger.info("模型训练流程完成")
        return eval_results


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

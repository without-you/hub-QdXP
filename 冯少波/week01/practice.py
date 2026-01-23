import jieba
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

"""
jieba:中文分词库
jieba.lcut():对单个句子进行中文分词，返回词语列表
jieba.cut():返回生成器，适合处理大文本
jieba.add_word() : 添加专业词汇
jieba.del_word() : 删除某个词汇
jieba.load_userdict(): 批量增加词典
"""
def test_jieba_cut():
    with open("test_data.txt", "r", encoding='utf-8') as f:
        for line in f:
            print(line)
            line = line.strip()
            if not line:
                continue

            # 分词（生成器，不占内存）
            seg_generator = jieba.cut(line)
            word_cnt = {}
            for word in seg_generator:
                if word not in word_cnt:
                    word_cnt[word] = 0
                word_cnt[word] += 1
            print("当前行词频：", word_cnt)

def test_jieba_api():
    text = "我想看狂飙电视剧"
    # res = jieba.cut(text)
    # res = jieba.lcut_for_search(text) # ['我', '爱', '吃', '北京', '烤鸭', '北京烤鸭']
    # jieba.del_word("狂飙") # ['我', '想', '看', '狂', '飙', '电视剧']
    res = jieba.lcut(text)
    print("res = ", res)

"""
逻辑回归（LogisticRegression）: 算概率
"""
def test_logis():
    # 模拟特征
    input_feature = np.array([[0,1],[1,1],[1,0]])
    labels = np.array(["Weather-Query","FilmTele-Play","Other"])

    # 创建模型
    model = LogisticRegression()
    # 训练模型
    model.fit(input_feature,labels)
    # 预测新样本
    pred = model.predict([[0,1]])
    print("预测类别：",pred)

"""
决策树：
模拟人做判断的过程，比如先问 “文本里有没有‘天气’？”→有→归为 Weather-Query；没有→再问 “有没有‘播放’？”→有→归为 FilmTele-Play
"""
def test_decision_tree():
    input_feature = np.array([[0,1],[1,1],[1,0]])
    labels = np.array(["Weather-Query","FilmTele-Play","Other"])
    model = DecisionTreeClassifier()
    model.fit(input_feature,labels)
    pred = model.predict([[1,0]])
    print("预测类型：",pred)

    print("decision rules:",export_text(model))

"""
随机森林：
决策树加强版，造n棵决策树，让每棵树投票，少数服从多数，效果比单决策树好，不易过拟合，工业常用算法
如：电商商品分类、风控分类
"""
def test_random_tree():
    input_feature = np.array([[0,1], [1,0], [0,0], [0,1], [1,0]])
    labels = np.array(["Weather-Query", "FilmTele-Play", "Other", "Weather-Query", "FilmTele-Play"])

    model = RandomForestClassifier()
    model.fit(input_feature,labels)

    pred = model.predict([[0,0]])
    print("random decision result:", pred)

"""
朴素贝叶斯 （文本分类专用）
基于 “概率统计”，比如统计 “Weather-Query 类别里‘天气’这个词出现的概率”，再算新文本属于各个类别的概率，选最大的
"""
def test_mnb():
    input_feature = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])  # 天气、播放、闹钟的词频
    labels = np.array(["Weather-Query", "FilmTele-Play", "Alarm-Update"])

    model = MultinomialNB()
    model.fit(input_feature,labels)

    pred = model.predict([[0,0,1]])
    print("multinomial nb result:", pred)

"""
支持向量机（SVC）—— 复杂数据分类
找一条 “最优分界线”，把不同类别的数据分开（比如把 Weather-Query 和 FilmTele-Play 的样本用线隔开），哪怕数据复杂也能找
"""
def test_svc():
    input_feature = np.array([[0, 1], [1, 0], [0, 0], [0.1, 1], [1, 0.1]])
    labels = np.array(["Weather-Query", "FilmTele-Play", "Other", "Weather-Query", "FilmTele-Play"])

    model = SVC()
    model.fit(input_feature, labels)
    pred = model.predict([[0.1, 1]])
    print("预测类别：", pred)  # 输出：['Weather-Query']

"""
神经网络（MLPClassifier）—— 复杂任务 "天花板"
模拟人脑神经元，多层 “运算” 后输出类别，能学复杂的特征关系（比如文本里的隐藏语义）
多标签分类、语义分类、大数据量分类
输入层（接收特征） → 隐藏层（提取规律） → 输出层（输出类别）
"""
def test_mlp():
    # 特征：5维向量（代表5个核心关键词的词频：天气、播放、闹钟、出行、其他
    input_feature = np.array([
        [1, 0, 0, 0, 0],  # 查明天北京的天气 → Weather-Query
        [0, 1, 0, 0, 0],  # 播放狂飙电视剧 → FilmTele-Play
        [0, 0, 1, 0, 0],  # 设置早上8点闹钟 → Alarm-Update
        [0, 0, 0, 1, 0],  # 查去上海的高铁票 → Travel-Query
        [0, 0, 0, 0, 1],  # 今天吃什么 → Other
        [1, 0, 0, 0, 0],  # 后天的气温是多少 → Weather-Query
        [0, 1, 0, 0, 0],  # 播放周杰伦的歌 → Music-Play（扩展类别）
        [0, 0, 1, 0, 0],  # 修改闹钟时间为7点 → Alarm-Update
    ])
    # 标签：对应每个特征的真实类别（和分类场景完全匹配）
    labels = np.array([
        "Weather-Query",
        "FilmTele-Play",
        "Alarm-Update",
        "Travel-Query",
        "Other",
        "Weather-Query",
        "Music-Play",
        "Alarm-Update"
    ])

    # ConvergenceWarning: Stochastic Optimizer: Maximum iterations (100) reached and the optimization hasn't converged yet
    # 收敛警告：随机优化器：已达到最大迭代次数（100次），但优化尚未收敛
    model = MLPClassifier(
        hidden_layer_sizes=(5,), # 1层隐藏层，10个神经元, 8 个样本用 10 个神经元容易 “过拟合”，5 个神经元更适配小样本，收敛更快
        max_iter=500, # 100次训练轮数,增加训练轮数，给模型足够时间调整参数到最优
        activation='relu', # 激活函数
        random_state=42, # 固定随机种子，每次结果都一样
        learning_rate_init=0.01 # 0.001学习率，调大学习率，让每轮调参的 “步子” 变大，更快走到误差最小的位置
    )
    model.fit(input_feature,labels)
    pred = model.predict([[1,0,0,0,0]])
    print("预测类别：", pred)  # 输出：['Weather-Query']

if __name__ == "__main__":
    # test_jieba_cut()
    # test_jieba_api()
    # test_logis()
    # test_decision_tree()
    # test_random_tree()
    # test_mnb()
    # test_svc()
    test_mlp()
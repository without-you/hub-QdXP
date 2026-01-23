#1、配置好python环境，包括常见的jieba、sklearn、pytorch等
#（注意：这个作业截图上传提交）
# 2、使用 dataset.csv数据集完成文本分类操作，需要尝试2种不同的模型。（注意：这个作业代码实操提交）

import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

dataset = pd.read_csv("dataset.csv",sep='\t',header=None,nrows=1)
# print(dataset[0].values)
# print(dataset[1].values)

# 分词
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
# print(input_sentence)

# 把分词后的文本，转换成词矩阵，用数字代表每个词在句子出现次数，模型只认数字，不认文,max_features:保留出现最多的N个词，stop_words:过滤停用词
vector = CountVectorizer(max_features=1000,stop_words=["的","这"])
# 建立词汇表，词语+编号,eg: {"我":0, "想看":1, "狂飙":2, "帮我":3, "查":4, "天气":5, "爱吃":6, "北京烤鸭":7}
vector.fit(input_sentence)
# 转成词频矩阵,词汇出现的次数（数字特征）eg: [[1,1,1,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]]
input_feature = vector.transform(input_sentence.values)
# 上面两行可合并为下面:
# input_feature = vector.fit_transform(input_sentence.values)

# K 近邻分类器
model = KNeighborsClassifier()
model.fit(input_feature,dataset[1].values)

client = OpenAI(
    api_key="sk-b2850aa9aff64528962998e0933d3912",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


"""
机器学习：文本分类，对输入文本完成类型划分
"""
def text_classify_using_ml(text:str) -> str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

"""
1、官网地址：https://bailian.console.aliyun.com/cn-beijing/?spm=a2c4g.11186623.0.0.156d533ao52Mhu&tab=api#/api/?type=model&url=2712576

"""
def text_classify_using_llm(text:str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {
                "role":"user",
                "content":f"""帮我进行文本分类：{text}
             输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
FilmTele-Play            
Video-Play               
Music-Play              
Radio-Listen           
Alarm-Update        
Travel-Query        
HomeAppliance-Control  
Weather-Query          
Calendar-Query      
TVProgram-Play      
Audio-Play       
Other
"""
             }
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    print('start')
    # print("机器学习：",text_classify_using_ml("帮我查北京的天气"))
    # print("大模型:",text_classify_using_llm("帮我查北京的天气"))


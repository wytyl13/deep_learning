'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-10-28 08:44:32
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-10-28 08:44:32
 * @Description: rstrip or lstrip,去除左空格或者右
 * 空格。里面不加参数默认去除空格，否则去除制定参数
 * 但是注意参数必须是开头或者结尾，否则使用replace
 * 更为详细的字符串操作可以使用help(str)
 * \d == [0-9]
 * \D == [^0-9]
 * \s == [' ']
 * \S == [^' ']
 * pip install spacy==3.7.0
 * pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz
 * 注意以上两个模块的版本的一致性
***********************************************************************'''
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.text import Text
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import ne_chunk
import spacy
nlp_en = spacy.load('en_core_web_sm')
# nlp_zh = spacy.load('zh_core_web_sm')
from spacy import displacy
from collections import Counter



def main1_re():
    input = 'a我a爱你aa Edison a001aa狗a01aa天空aa01 '
    print(input)
    print(input.strip('a'))
    print(input.replace('a', ''))
    print(input.find('a'))
    # 是否全是字母（只包含汉字或者字符）
    print(input.isalpha())
    # 是否全是数字
    print(input.isdigit())
    # 按照空格分割
    print(input.split(' '))
    # join拼接分割后的字符串
    print(''.join(input.split(' ')))

    pattern1 = re.compile(r'.')
    pattern2 = re.compile(r'[abc]')
    pattern3 = re.compile(r'[a-zA-Z]')
    pattern4 = re.compile(r'[^a-zA-Z]')
    pattern5 = re.compile(r'[a-zA-Z]|[0-9]')
    # 匹配所有的，符合条件的不符合条件的都匹配
    pattern6 = re.compile(r'\d*')
    # 匹配数字连续出现3次的，
    # 当然也可以使用1,2匹配1到2之间的范围
    pattern7 = re.compile(r'\d{2}')

    list1 = re.findall(pattern1, input)
    list2 = re.findall(pattern2, input)
    list3 = re.findall(pattern3, input)
    list4 = re.findall(pattern4, input)
    list5 = re.findall(pattern5, input)
    list6 = re.findall(pattern6, input)
    list7 = re.findall(pattern7, input)
    match1 = re.search(pattern7, input)
    result1 = re.sub(pattern6, 'hello', input)
    result2 = re.sub(pattern7, 'hello', input)
    # 返回一个元祖，包含结果和替换的次数
    result3 = re.subn(pattern7, 'hello', input)
    # print(list1, list2, list3, list4, list5)
    # 按照规则去切片
    result4 = re.split(pattern7, input)
    print(list6)
    print(list7)
    print(match1.group())
    print(result1)
    print(result2)
    print(result3)
    print(result4)


def main2():
    input_str = "Today's weather is good, very windy\
                and sunny, we have no classes in the\
                afternoon, we have to play basketball tomorrow Edison went to Tsinghua university."
    tokens = word_tokenize(input_str)
    words = [word.lower() for word in tokens]
    word_set = set(words)
    print(tokens)
    t = Text(tokens)
    nums = t.count("have")
    print(nums)
    print(stopwords.raw('english').replace('\n', '  '))
    # 和停用词的交集，也就是本案例句子中的停用词
    word_intersection = word_set.intersection(set(stopwords.words\
                    ('english')))
    print(word_intersection)

    # 然后我们可以过滤掉停用词
    filtered = [word for word in word_set if(word not in word_intersection)] 
    # 当然我们也可以直接从停用词中过滤
    filtered_ = [word for word in word_set if(word not in stopwords.words('english'))]
    print(filtered)
    print(filtered_)
    # 词性标注
    tags = pos_tag(tokens)
    print(tags)
    # 命名实体识别
    ne = ne_chunk(tags)
    print(ne)
    # 使用nlp实体去分词或者分句
    doc = nlp_en('weather is good, very windy and sunny. we have no classes in the afternoon.')
    # 输出词性和命名体识别
    for token in doc:
        print('{}-{}'.format(token, token.pos_))
    for sent in doc.sents:
        print(sent)
    for ent in doc.ents:
        print('{}-{}'.format(ent, ent.label_))
    # display the correspond param in jupyter.
    # display.render(doc, style = 'ent', jupyter=True)
def read_file(file_name):
    with open(file_name, 'r') as file:
        return file.read()


def find_ent(processed_text, ent_label):
    c = Counter()
    for ent in processed_text.ents:
        print(ent.label_)
        if ent.label_ == ent_label:
            c[ent.lemma_] += 1
    return c

def main3():
    text = read_file('c:/users/80521/desktop/spacy_test.txt')
    processed_text = nlp_en(text)
    sentences = [s for s in processed_text.sents]
    print(len(sentences))
    print(sentences)
    ent_list = find_ent(processed_text, 'PERSON')
    print(ent_list)



if __name__ == "__main__":
    # main2()
    main3()
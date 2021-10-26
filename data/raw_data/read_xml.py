import os
import json
import random
import xml.etree.ElementTree as ET

def read_txt():
    """处理测试数据
    """
    fr1 = open('CGED-Test-2016/CGED16_HSK_Test_Input.txt').readlines()
    fr2 = open('CGED-Test-2016/CGED16_HSK_Test_Truth.txt').readlines()
    tempDict = dict()
    for f1 in fr1:
        arrays1 = f1.split('\t')
        # print(arrays1)
        assert len(arrays1) == 2
        id = arrays1[0].replace('(sid=', '').replace(')', '')
        tempDict[id] = arrays1[1].replace('\n', '')

    fw = open('test.json', 'w', encoding='utf-8')

    text_error = {}
    for f2 in fr2:
        arrays2 = f2.split(',')
        if arrays2[0] not in text_error:
            text_error[arrays2[0]] = []
        if len(arrays2) == 2:
            text_error[arrays2[0]].append('correct')
        if len(arrays2) == 4:
            text_error[arrays2[0]].append((arrays2[1].replace(' ', ''), arrays2[2].replace(' ', ''), arrays2[3].replace(' ', '').replace('\n', '')))
    print(len(text_error))
    for k, v in text_error.items():
        tempDict1 = {}
        tempList1 = []
        tempDict1["id"] = k
        tempDict1["raw_text"] = tempDict[k]
        if v == ['correct']:
            tempList1.append({"label": "correct"})
        else:
            for error in v:
                tempList1.append({"label": error[2], "entity": tempDict[k][int(error[0])-1:int(error[1])], "start": int(error[0]), "end": int(error[1])})
        tempDict1["label"] = tempList1
        l1 = json.dumps(tempDict1, ensure_ascii=False)
        fw.write(l1)
        fw.write('\n')

def read_xml():
    """处理训练数据
    """
    count = 0
    tree = ET.parse("CGED-DATA-2018/train2018.release.xml")
    # tree = ET.parse("CGED-Test-2016/CGED16_HSK_TrainingSet.xml")
    root = tree.getroot()

    # 标签名
    # print('root_tag:',root.tag)
    fw = open('train.json', 'a', encoding='utf-8')
    for child in root:
        count += 1
        tempDict = dict()
        tempList = list()
        for elem in child:
            if elem.tag == "TEXT":
                tempDict['id'] = elem.attrib["id"]
                tempDict['raw_text'] = elem.text.replace('\n', '')
                Flag_text = elem.text.replace('\n', '')
            if elem.tag == "CORRECTION":
                if elem.text is not None:
                    tempDict['correction_text'] = elem.text.replace('\n', '')
            if elem.tag == "ERROR":
                # print(elem.attrib["start_off"])
                # print(Flag_text)
                tempList.append({"label": elem.attrib["type"], "entity": Flag_text[int(elem.attrib["start_off"])-1:int(elem.attrib["end_off"])], "start": int(elem.attrib["start_off"]), "end": int(elem.attrib["end_off"])})
        tempDict['label'] = tempList
        l1 = json.dumps(tempDict, ensure_ascii=False)
        fw.write(l1)
        fw.write('\n')

if __name__ == '__main__':
    pass


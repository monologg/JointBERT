# JointBERT

中文 | [原始简介](readme_en.md)

扩展JointBERT支持中文训练, 提供从数据合成到意图和槽位联合训练, 测试完整流程.



## 环境

- python>=3.8
- torch==2.0.1
- transformers==3.0.2
- seqeval==1.2.2
- pytorch-crf==0.7.2



## 数据集

|           | Train  | Dev  | Test | Intent Labels | Slot Labels |
| --------- | ------ | ---- | ---- | ------------- | ----------- |
| ATIS      | 4,478  | 500  | 893  | 21            | 120         |
| Snips     | 13,084 | 700  | 700  | 7             | 72          |
| generalQA | 46050  | 5754 | 5754 | 7             | 15          |



- The number of labels are based on the _train_ dataset.
- Add `UNK` for labels (For intent and slot labels which are only shown in _dev_ and _test_ dataset)
- Add `PAD` for slot label



## Prediction

这里提供训练好的模型[52AI/generalQA_intent_slotFilling](https://huggingface.co/52AI/generalQA_intent_slotFilling/tree/main) 供测试. 国内下载容易中断,多运行两次.

```bash
$ python3 predict.py --task generalQA \
                     --input_file data/testcase/generalQAtest.txt \
                     --output_file local/generalQAtest_predict.txt \
                     --model_dir out/generalQA
```

> <TranslationZhEn> -> 请 问 [你:B-TransEnZhSentence] [几:I-TransEnZhSentence] [岁:I-TransEnZhSentence] [了:I-TransEnZhSentence] 用 英 语 怎 么 说 ？
>
> <TranslationEnZh> -> 翻 译 ： [i:B-TransEnZhSentence] [love:I-TransEnZhSentence] [you:I-TransEnZhSentence]
>
> <CreateSentence> -> 用 [美:B-CreateSenEntity] [好:I-CreateSenEntity] 写 一 个 句 子
>
> <Antonym> -> [明:B-AntonymEntity] [天:I-AntonymEntity] 的 反 义 词
>
> <Synonym> -> [后:B-SynonymEntity] [天:I-SynonymEntity] 的 同 义 词
>
> 测试结果: local/generalQAtest_predict.txt



## 合成数据

利用预先收集的源数据(中英文句子, 词语, 成语, 常用中文字符),通过预定义模板合成训练数据.

```bash
python3 dataGenerate/dataSynthesis.py
```

数据目录data/generalQA包含训练集,验证集,测试集, 每个集中包含文本, 意图标签, 槽位标签.

label: 对应意图标签, seq.out对应Slot标签,  seq.in对应输入文本.

data/generalQA/intent_label.txt : 为意图类别

data/generalQA/slot_label.txt: 为slot位类别



## Training & Evaluation

```bash
# 中文QA,
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --task generalQA --model_type bertzh \
                  --model_dir out/generalQA \
                  --do_train --do_eval
PS: 使用缓存好的数据添加参数 --use_cache
CUDA_VISIBLE_DEVICES=0 python3 main.py --task generalQA --model_type bertzh \
                  --model_dir out/generalQA_crf \
                  --do_train --do_eval  --use_cache --use_crf
```

其他数据的训练参考



## Results

- Run 5 ~ 10 epochs (Record the best result)
- Only test with `uncased` model
- ALBERT xxlarge sometimes can't converge well for slot prediction.

|           |                         | Intent acc (%) | Slot F1 (%) | Sentence acc (%) | batchsize | loss  | 训练日志                                                     |
| --------- | ----------------------- | -------------- | ----------- | ---------------- | --------- | ----- | ------------------------------------------------------------ |
| **Snips** | BERT                    | **99.14**      | 96.90       | 93.00            |           |       |                                                              |
|           | BERT + CRF              | 98.57          | **97.24**   | **93.57**        |           |       |                                                              |
|           | DistilBERT              | 98.00          | 96.10       | 91.00            |           |       |                                                              |
|           | DistilBERT + CRF        | 98.57          | 96.46       | 91.85            |           |       |                                                              |
|           | ALBERT                  | 98.43          | 97.16       | 93.29            |           |       |                                                              |
|           | ALBERT + CRF            | 99.00          | 96.55       | 92.57            |           |       |                                                              |
| **ATIS**  | BERT                    | 97.87          | 95.59       | 88.24            |           |       |                                                              |
|           | BERT + CRF              | **97.98**      | 95.93       | 88.58            |           |       |                                                              |
|           | DistilBERT              | 97.76          | 95.50       | 87.68            |           |       |                                                              |
|           | DistilBERT + CRF        | 97.65          | 95.89       | 88.24            |           |       |                                                              |
|           | ALBERT                  | 97.64          | 95.78       | 88.13            |           |       |                                                              |
|           | ALBERT + CRF            | 97.42          | **96.32**   | **88.69**        |           |       |                                                              |
| generalQA | bert-base-chinese       | 96.81          | 96.29       |                  | 64        | 0.05  | [log](https://pan.baidu.com/s/1OwnSiADqZw4LTTuNu86McA?pwd=d3uq) |
|           | bert-base-chinese + CRF | 96.43          | 96.37       |                  | 64        | 0.253 |                                                              |



## Updates

- 2023.09. 04
  - 源数据合成语料dataGenerate/corpus(数据来自互联网)
  - 支持合成中文意图识别和槽位填充的微调训练数据generalQA
  - 支持基于中文BERT进行训练
  - 开源训练模型到[HuggingFace](https://huggingface.co/52AI/generalQA_intent_slotFilling)
- [之前的更新记录](./readme_en.md)



## References

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- https://github.com/monologg/JointBERT/
- [joint_intent_slot](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/joint_intent_slot.html)
# JointBERT

(Unofficial) Pytorch implementation of `JointBERT`: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)

## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68875755-b2f92900-0746-11ea-8819-401d60e4185f.png" />  
</p>

- Predict `intent` and `slot` at the same time from **one BERT model** (=Joint model)
- total_loss = intent_loss + coef \* slot_loss (Change coef with `--slot_loss_coef` option)
- **If you want to use CRF layer, give `--use_crf` option**

## Dependencies

- python>=3.5
- torch==1.4.0
- transformers==2.7.0
- seqeval==0.0.12
- pytorch-crf==0.7.2

## Dataset

|       | Train  | Dev | Test | Intent Labels | Slot Labels |
| ----- | ------ | --- | ---- | ------------- | ----------- |
| ATIS  | 4,478  | 500 | 893  | 21            | 120         |
| Snips | 13,084 | 700 | 700  | 7             | 72          |

- The number of labels are based on the _train_ dataset.
- Add `UNK` for labels (For intent and slot labels which are only shown in _dev_ and _test_ dataset)
- Add `PAD` for slot label

## Training & Evaluation

```bash
$ python3 main.py --task {task_name} \
                  --model_type {model_type} \
                  --model_dir {model_dir_name} \
                  --do_train --do_eval \
                  --use_crf

# For ATIS
$ python3 main.py --task atis \
                  --model_type bert \
                  --model_dir atis_model \
                  --do_train --do_eval
# For Snips
$ python3 main.py --task snips \
                  --model_type bert \
                  --model_dir snips_model \
                  --do_train --do_eval
```

## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

## Results

- Run 5 ~ 10 epochs (Record the best result)
- Only test with `uncased` model
- ALBERT xxlarge sometimes can't converge well for slot prediction.

|           |                  | Intent acc (%) | Slot F1 (%) | Sentence acc (%) |
| --------- | ---------------- | -------------- | ----------- | ---------------- |
| **Snips** | BERT             | **99.14**      | 96.90       | 93.00            |
|           | BERT + CRF       | 98.57          | **97.24**   | **93.57**        |
|           | DistilBERT       | 98.00          | 96.10       | 91.00            |
|           | DistilBERT + CRF | 98.57          | 96.46       | 91.85            |
|           | ALBERT           | 98.43          | 97.16       | 93.29            |
|           | ALBERT + CRF     | 99.00          | 96.55       | 92.57            |
| **ATIS**  | BERT             | 97.87          | 95.59       | 88.24            |
|           | BERT + CRF       | **97.98**      | 95.93       | 88.58            |
|           | DistilBERT       | 97.76          | 95.50       | 87.68            |
|           | DistilBERT + CRF | 97.65          | 95.89       | 88.24            |
|           | ALBERT           | 97.64          | 95.78       | 88.13            |
|           | ALBERT + CRF     | 97.42          | **96.32**   | **88.69**        |

## Updates

- 2019/12/03: Add DistilBert and RoBERTa result
- 2019/12/14: Add Albert (large v1) result
- 2019/12/22: Available to predict sentences
- 2019/12/26: Add Albert (xxlarge v1) result
- 2019/12/29: Add CRF option
- 2019/12/30: Available to check `sentence-level semantic frame accuracy`
- 2019/01/23: Only show the result related with uncased model
- 2019/04/03: Update with new prediction code

## References

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)

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
- torch>=1.1.0
- transformers>=2.2.2
- seqeval>=0.0.12
- pytorch-crf==0.7.2

## Dataset

|       | Train  | Dev | Test | Intent Labels | Slot Labels |
| ----- | ------ | --- | ---- | ------------- | ----------- |
| ATIS  | 4,478  | 500 | 893  | 21            | 120         |
| Snips | 13,084 | 700 | 700  | 7             | 72          |

- The number of labels are based on the _train_ dataset.
- Add `UNK` for labels (For intent and slot labels which are only shown in _dev_ and _test_ dataset)

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

- There should be a trained model before running prediction.
- You should write sentences in `preds.txt` in `preds` directory.
- **If your model is trained using CRF, you must give `--use_crf` option when running prediction.**

```bash
$ python3 main.py --task snips \
                  --model_type bert \
                  --model_dir snips_model \
                  --do_pred \
                  --pred_dir preds \
                  --pred_input_file preds.txt
```

## Results

- Run 5 epochs each (No hyperparameter tuning)

|           |            | Intent acc (%) | Slot F1 (%) |
| --------- | ---------- | -------------- | ----------- |
| **ATIS**  | BERT       | 97.87          | 95.46       |
|           | DistilBERT | 97.54          | 94.89       |
|           | RoBERTa    | 97.64          | 94.94       |
|           | ALBERT     | 98.20          | 95.59       |
| **Snips** | BERT       | 98.29          | 96.05       |
|           | DistilBERT | 98.42          | 94.10       |
|           | RoBERTa    | 98.14          | 94.60       |
|           | ALBERT     | 98.28          | 95.50       |

## Updates

- 2019/12/03: Add DistilBert and RoBERTa result
- 2019/12/14: Add Albert (large v1) result
- 2019/12/22: Available to predict sentences
- 2019/12/26: Add Albert (xxlarge v1) result
- 2019/12/29: Add CRF option

## References

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)

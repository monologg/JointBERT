# JointBERT

(Unofficial) Pytorch implementation of `JointBERT`: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)

## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68875755-b2f92900-0746-11ea-8819-401d60e4185f.png" />  
</p>

- Predict `intent` and `slot` at the same time from **one BERT model** (=Joint model)
- total_loss = intent_loss + slot_loss

## Dependencies

- python>=3.5
- torch==1.1.0
- transformers==2.1.1
- scikit-learn>=0.20.0
- seqeval>=0.0.12

## Dataset

|       | Train  | Dev | Test | Intent Labels | Slot Labels |
| ----- | ------ | --- | ---- | ------------- | ----------- |
| ATIS  | 4,478  | 500 | 893  | 21            | 120         |
| Snips | 13,084 | 700 | 700  | 7             | 72          |

- The number of labels are based on the _train_ dataset.
- Add `UNK` for labels (For intent and slot labels which are only shown in _dev_ and _test_ dataset)

## Usage

```bash
$ python3 main.py --task {task_name} \
                  --model_dir {model_dir_name} \
                  --do_train --do_eval

# For ATIS
$ python3 main.py --task atis \
                  --model_dir atis_model \
                  --do_train --do_eval
# For Snips
$ python3 main.py --task snips \
                  --model_dir snips_model \
                  --do_train --do_eval
```

## References

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Keras Implementation](https://github.com/lytum/joint-intent-classification-and-slot-filling-based-on-BERT)

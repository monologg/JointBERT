# JointBERT

(Unofficial) Pytorch implementation of `JointBERT`: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)

## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68875755-b2f92900-0746-11ea-8819-401d60e4185f.png" />  
</p>

## Dependencies

- python>=3.5
- torch==1.1.0
- transformers==2.1.1
- scikit-learn>=0.20.0
- seqeval>=0.0.12

## Dataset

1. ATIS
2. Snips

## Usage

```bash
$ python3 main.py --do_train --do_eval --dataset {dataset_name}
```

## References

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Keras Implementation](https://github.com/lytum/joint-intent-classification-and-slot-filling-based-on-BERT)
- [A Joint Model of Intent Determination and Slot Filling for Spoken Language Understanding](https://www.ijcai.org/Proceedings/16/Papers/425.pdf)

## TODO

- [x] Model (IntentClassifier, SlotClassifier, JointBERT)
- [ ] Find Dataset Paper
- [ ] Transformers NER analysis (Data loading by word level, BertForTokenClassification)
- [ ] CRF Layer

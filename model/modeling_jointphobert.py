import torch
import torch.nn as nn
from transformers.modeling_roberta import RobertaPreTrainedModel,RobertaModel,RobertaConfig
from torchcrf import CRF
from .module import IntentClassifier, SlotClassifier


class JointPhoBERT(RobertaPreTrainedModel):
    def __init__(self, config, args,  slot_label_lst):
        super(JointPhoBERT, self).__init__(config)
        self.args = args

        self.num_slot_labels = len(slot_label_lst)
        self.bert = RobertaModel(config=config)  # Load pretrained bert
        self.bilstm = nn.LSTM(bidirectional=True,
                              input_size=config.hidden_size * 4 ,
                              hidden_size=config.hidden_dim // 2, num_layers=2, batch_first=True)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids,  slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,output_hidden_states=True,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = torch.cat((outputs[2][-1],outputs[2][-2],outputs[2][-3],outputs[2][-4]),dim=-1)
        sequence_output,_=self.bilstm(sequence_output)

        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0



        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ( slot_logits) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
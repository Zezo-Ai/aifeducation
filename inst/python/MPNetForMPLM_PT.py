# This file is part of the R package "aifeducation".
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as published by
# the Free Software Foundation.
#
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

"""
BERT            MPNet
0   - [PAD];    1     - <pad>
100 - [UNK];    3     - <unk>
101 - [CLS];    0     - <s>
102 - [SEP];    2     - </s>
103 - [MASK];   30526 - <mask>
"""

# ======================================================================================
# Imports
# ======================================================================================

from torch import nn
from transformers import MPNetForMaskedLM

# ======================================================================================
# LossFunction for MLM and PLM tasks
# ======================================================================================


class MPLMLoss_PT(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        """
        The combination of nn.LogSoftmax and nn.NLLLoss is equivalent to using nn.CrossEntropyLoss
        """
        self.criterion_mlm = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.criterion_plm = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, mlm_logits, plm_logits, mlm_labels, plm_labels):
        # forward(y_pred, y_true)
        mlm_loss = self.criterion_mlm(mlm_logits, mlm_labels)
        plm_loss = self.criterion_plm(plm_logits, plm_labels)

        return mlm_loss + plm_loss


# ======================================================================================
# Head of model for MLM and PLM tasks
# ======================================================================================


class MPNetForMPLM_PT(MPNetForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.plm_head = nn.Linear(
            config.hidden_size, config.vocab_size
        )  # Adding PLM output layer

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mlm_labels=None,
        plm_labels=None,
        return_dict=None,
        output_hidden_states=None,
    ):
        """
        outputs is BaseModelOutputWithPooling with args:
        - last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        - pooler_output
        - hidden_states
        - attentions

        hidden_size = 768
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        """
        mlm_logits, plm_logits (torch.FloatTensor of shape (batch_size, sequence_length, vocab_size))

        vocab_size = 30527
        """

        sequence_output = outputs[0]

        mlm_logits = self.lm_head(sequence_output)
        plm_logits = self.plm_head(sequence_output)  # Computing PLM logits

        """
        Logits (torch.FloatTensor of shape (batch_size x sequence_length, vocab_size)):
            mlm_logits.view(-1, mlm_logits.size(-1))
            plm_logits.view(-1, plm_logits.size(-1))
        Targets (torch.FloatTensor of shape (batch_size x sequence_length)):
            mlm_labels.view(-1)
            plm_labels.view(-1)

            [0 <= targets[i] < vocab_size]
        """

        loss = None
        if mlm_labels is not None and plm_labels is not None:
            loss_fct = MPLMLoss_PT()
            loss = loss_fct(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                plm_logits.view(-1, plm_logits.size(-1)),
                mlm_labels.view(-1),
                plm_labels.view(-1),
            )

        if not return_dict:
            output = (mlm_logits, plm_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": mlm_logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }

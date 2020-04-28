from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, RobertaConfig, RobertaModel
import torch


class HierarchicalClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.map_u = nn.Linear(config.hidden_size, config.u_dim)
        self.map_d = nn.Linear(config.u_dim, config.d_dim)

        self.map_relatedness = nn.Linear(config.u_dim, 2)
        self.map_related_stances = nn.Linear(config.u_dim+1, 4)


    def forward(self, input_rep):
        hidden_layer = F.relu(self.map_u(input_rep))
        unrelated_prob = F.softmax(self.map_relatedness(hidden_layer), dim=1)
        mid = torch.cat((unrelated_prob[:, 0].view(-1, 1), hidden_layer), dim=1)
        stance_prob = F.softmax(self.map_related_stances(mid), dim=1)
        return unrelated_prob, stance_prob, hidden_layer

    def mmd_regularization(self, hidden_layer, labels, unrelated_index):
        mmd_layer = F.relu(self.map_d(hidden_layer))
        n1_samples = (labels == unrelated_index)
        n2_samples = (labels != unrelated_index)
        n1_sum = torch.matmul(n1_samples.type(torch.float), mmd_layer)
        n2_sum = torch.matmul(n2_samples.type(torch.float), mmd_layer) / torch.sum(n1_samples)
        return torch.sum(n1_samples), n1_sum, torch.sum(n2_samples), n2_sum


class HierarchicalRoberta(BertPreTrainedModel):
    config_class = RobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = HierarchicalClassifier(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            unrelated_index=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[1]
        return self.classifier(sequence_output)

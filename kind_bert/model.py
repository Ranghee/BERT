import torch
import torch.nn as nn
import torch.nn.functional as F


def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.hidden_size, out_channels=self.config.intermediate_size, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.intermediate_size, out_channels=self.config.hidden_size, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, inputs):
        # (batch_size, intermediate_size, n_seq)
        output = self.active(self.conv1(inputs.transpose(1,2)))
        # (batch_size, n_seq, hidden_size)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        # (batch_size, n_seq, hidden_size)
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.attention_head_size = int(self.config.hidden_size / self.config.num_attention_heads)
        self.scale = 1 / (self.attention_head_size ** 0.5)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, V)

        return context, attn_prob


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = self.config.num_attention_heads
        self.attention_head_size = int(self.config.hidden_size / self.config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.W_Q = nn.Linear(self.config.hidden_size, self.all_head_size)
        self.W_K = nn.Linear(self.config.hidden_size, self.all_head_size)
        self.W_V = nn.Linear(self.config.hidden_size, self.all_head_size)
        self.scaled_dot_attention = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.num_attention_heads * self.attention_head_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)           # 행렬의 크기를 구함
        # (batch_size, num_attention_heads, n_q_seq, attention_head_size)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.num_attention_heads, self.attention_head_size).transpose(1,2)      # 행렬의 모양을 바꿔줌 https://tutorials.pytorch.kr/beginner/blitz/tensor_tutorial.html
        k_s = self.W_K(K).view(batch_size, -1, self.config.num_attention_heads, self.attention_head_size).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.config.num_attention_heads, self.attention_head_size).transpose(1,2)
        # (batch_size, num_attention_heads, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.num_attention_heads, 1, 1)              # 두번째 차원에 1을 추가, dim= 1로 n_head만큼 반복추가 https://seducinghyeok.tistory.com/9
        # (batch_size, num_attention_heads, n_q_seq, attention_head_size), (batch_size, num_attention_heads, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attention(q_s, k_s, v_s, attn_mask)
        # (batch_size, num_attention_heads, n_q_seq, hidden_size * attention_head_size)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.num_attention_heads)
        # (batch_size, num_attention_heads, n_q_seq, e_embd)
        output = self.linear(context)
        output = self.dropout(output)

        return output, attn_prob


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attention = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
    
    def forward(self, inputs, attn_mask):
        # (batch_size, n_enc_seq, hidden_size), (batch_size, num_attention_heads, n_enc_seq, n_enc_seq)
        att_outputs, attn_prob = self.self_attention(inputs, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        # (batch_size, n_enc_seq, hidden_size)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        # (batch_size, n_enc_seq, hidden_size), (batch_size, num_attention_heads, n_enc_seq, n_enc_seq)
        return ffn_outputs, attn_prob


class Encoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.word_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size, padding_idx=self.config.pad_token_id)
        self.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)
        self.token_type_embeddings = nn.Embedding(self.config.type_vocab_size, self.config.hidden_size)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.num_hidden_layers)])
    
    def forward(self, inputs, segments):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        # input size 크기 만큼 0 1 2 3 4 5 6 ... input size로 초기화
        pos_mask = inputs.eq(self.config.pad_token_id) # 두 개를 비교해 같으면 True 다르면 False https://runebook.dev/ko/docs/pytorch/generated/torch.eq
        positions.masked_fill_(pos_mask, 0)

        outputs = self.word_embeddings(inputs) + self.position_embeddings(positions) + self.token_type_embeddings(segments)

        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.pad_token_id)

        attn_probs = []
        for layer in self.layers:
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)

        return outputs, attn_probs



class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoder = Encoder(self.config)

        self.linear = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.activation = torch.tanh

    def forward(self, inputs, segments):

        outputs, self_attn_probs = self.encoder(inputs, segments)

        outputs_cls = outputs[:, 0].contiguous()
        outputs_cls = self.linear(outputs_cls)
        outputs_cls = self.activation(outputs_cls)

        return outputs, outputs_cls, self_attn_probs

    def save(self, epoch, loss, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "state_dict": self.state_dict()
        }, path)

    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"]

class BERTPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BERT(self.config)

        # classifier
        self.projection_cls = nn.Linear(self.config.hidden_size, 2, bias=False)
        # lm
        self.projection_lm = nn.Linear(self.config.hidden_size, self.confing.vocab_size, bias=False)
        self.projection_lm.weight = self.bert.encoder.word_embeddings.weight
    
    def forward(self, inputs, segments):
        outputs, outputs_cls, attn_probs = self.bert(inputs, segments)
        logits_cls = self.projection_cls(outputs_cls)
        logits_lm = self.projection_lm(outputs)
        return logits_cls, logits_lm, attn_probs

















class BERTPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BERT(self.config)

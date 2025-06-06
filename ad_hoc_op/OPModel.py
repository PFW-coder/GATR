import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '-1'

class OPModel(nn.Module):

    def __init__(self, env_params, **model_params):
        super().__init__()
        self.model_params = model_params
        self.env_params = env_params

        self.encoder = OP_Encoder(**model_params)
        self.decoder = OP_Decoder(self.env_params, **model_params)
        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_prize = reset_state.node_prize
        # shape: (batch, problem)
        node_xy_prize = torch.cat((node_xy, node_prize[:, :, None]), dim=2)
        # shape: (batch, problem, 3)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_prize)
        # shape: (batch, 1, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        agent_num = state.BATCH_IDX.size(2)
        graph_size = state.graph_size

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size, agent_num), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))


        elif state.selected_count == 1 and not self.model_params['random_choice']:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            selected_ = selected % graph_size
            # shape: (batch, pomo)
            selected_index = selected // graph_size
            # shape: (batch, pomo)
            selected = state.current_node.clone()
            # shape: (batch, pomo, agent)
            selected.scatter_(-1, selected_index[:, :, None], selected_[:, :, None])

        elif state.selected_count == 1 and self.model_params['random_choice']:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            mask_prob = torch.ones((batch_size * pomo_size, agent_num))
            no_mask_index = (torch.multinomial(mask_prob, 1)[:, :, None].reshape(batch_size, pomo_size, -1))
            mask_new = torch.ones((batch_size, pomo_size, agent_num)) * float('-inf')
            mask_new = mask_new.scatter(-1, no_mask_index, 0)
            ninf_mask = state.ninf_mask + mask_new[:, :, :, None].repeat(1, 1, 1, graph_size)
            probs = self.decoder(encoded_last_node, state.endurance[:, None, :] - state.used_time, state.agent_speed, ninf_mask=ninf_mask)

            while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                with torch.no_grad():
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                # shape: (batch, pomo)
                prob = probs[state.BATCH_IDX[:, :, 0], state.POMO_IDX[:, :, 0], selected].reshape(batch_size, pomo_size)
                # shape: (batch, pomo)
                if (prob != 0).all():
                    break
            selected_ = selected % graph_size
            # shape: (batch, pomo)
            selected_index = selected // graph_size
            # shape: (batch, pomo)
            selected = state.current_node.clone()
            # shape: (batch, pomo, agent)
            selected.scatter_(-1, selected_index[:, :, None], selected_[:, :, None])

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, agent, embedding)
            probs = self.decoder(encoded_last_node, state.endurance[:, None, :] - state.used_time, state.agent_speed, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, agent * (problem+1))

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX[:, :, 0], state.POMO_IDX[:, :, 0], selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.
            selected_ = selected % graph_size
            # shape: (batch, pomo)
            selected_index = selected // graph_size
            # shape: (batch, pomo)
            selected = state.current_node.clone()
            # shape: (batch, pomo, agent)
            selected.scatter_(-1, selected_index[:, :, None], selected_[:, :, None])

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, 1, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo, agent)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    agent_num = node_index_to_pick.size(2)
    embedding_dim = encoded_nodes.size(3)

    gathering_index = node_index_to_pick[:, :, :, None].expand(batch_size, pomo_size, agent_num, embedding_dim)
    # shape: (batch, pomo, agent, embedding)

    picked_nodes = encoded_nodes.repeat(1, pomo_size, 1, 1).gather(dim=2, index=gathering_index)
    # shape: (batch, pomo, agent, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class OP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand)
        # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out = layer(out)

        return out
        # shape: (batch, 1, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding) or (batch, 1, problem+1, embedding)
        head_num = self.model_params['head_num']
        if len(input1.size()) == 3:
            input1 = input1[:, None, :, :]
        #shape: (batch, 1, problem+1, embedding)

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, 1, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, 1, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, 1, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, 1, problem, embedding)


class Context_EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.context_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization2(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization2(**model_params)

    def forward(self, context_input):
        # shape: (batch, pomo, agent, EMBEDDING_DIM+2)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(context_input), head_num=head_num)
        k = reshape_by_heads(self.Wk(context_input), head_num=head_num)
        v = reshape_by_heads(self.Wv(context_input), head_num=head_num)
        # qkv shape: (batch, pomo, head_num, agent, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, pomo, agent, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, agent, embedding)

        return multi_head_out


########################################
# DECODER
########################################

class OP_Decoder(nn.Module):
    def __init__(self, env_params, **model_params):
        super().__init__()
        self.model_params = model_params
        self.env_params = env_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        self.pomo_size = self.env_params['pomo_size']
        context_layer_num = self.model_params['context_layer_num']

        self.context_linear = nn.Linear(embedding_dim+2, embedding_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim+2, head_num * qkv_dim, bias=False)
        self.Wq_context = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.layers = nn.ModuleList([Context_EncoderLayer(**model_params) for _ in range(context_layer_num)])
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, 1, problem+1, embedding)

        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, 1, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(2, 3)
        # shape: (batch, 1, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, left_time, agent_speed, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, agent, embedding)
        # load.shape: (batch, pomo, agent)
        # ninf_mask.shape: (batch, pomo, agent, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, left_time[:, :, :, None], agent_speed[:, None, :, None].repeat(1, self.pomo_size, 1, 1)), dim=3)
        # shape = (batch, pomo, agent, EMBEDDING_DIM+2)

        if self.model_params['context_decoder']:
            out = self.context_linear(input_cat)
            for layer in self.layers:
                out = layer(out)
            # shape: (batch, pomo, agent, embedding)

            q_last = reshape_by_heads(self.Wq_context(out), head_num=head_num)
            # shape: (batch, pomo, head_num, agent, qkv_dim)

            q = q_last
            # shape: (batch, pomo, head_num, agent, qkv_dim)

        else:
            q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
            # shape: (batch, pomo, head_num, agent, qkv_dim)

            q = q_last
            # shape: (batch, pomo, head_num, agent, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, agent, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, agent, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key.repeat(1, self.pomo_size, 1, 1))
        # shape: (batch, pomo, agent, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, agent, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked.reshape(score.size(0), score.size(1), -1), dim=2)
        # shape: (batch, pomo, agent * problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, m, n, head_num*key_dim)   : n can be either agent or PROBLEM_SIZE, m can be either 1 or pomo

    batch_s = qkv.size(0)
    m = qkv.size(1)
    n = qkv.size(2)

    q_reshaped = qkv.reshape(batch_s, m, n, head_num, -1)
    # shape: (batch, m, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(2, 3)
    # shape: (batch, m, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, m, head_num, n, key_dim)   : n can be either agent or PROBLEM_SIZE, m can be either 1 or pomo
    # k,v shape: (batch, 1, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, m, problem)
    # rank3_ninf_mask.shape: (batch, group, m, problem)

    batch_s = q.size(0)
    m = q.size(1)
    head_num = q.size(2)
    n = q.size(3)
    key_dim = q.size(4)

    input_s = k.size(3)

    k = k.expand(k.size(0), m, k.size(2), k.size(3), k.size(4))
    v = v.expand(v.size(0), m, v.size(2), v.size(3), v.size(4))

    score = torch.matmul(q, k.transpose(3, 4))
    # shape: (batch, m, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, :, None, None, :].expand(batch_s, m, head_num, n, input_s)

    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, :, None, :, :].expand(batch_s, m, head_num, n, input_s)


    weights = nn.Softmax(dim=4)(score_scaled.reshape(batch_s, m, head_num, 1, -1))
    # shape: (batch, m, head_num, 1, n * problem)

    weights = weights.reshape(batch_s, m, head_num, n, -1)
    # shape: (batch, m, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, m, head_num, n, key_dim)

    out_transposed = out.transpose(2, 3)
    # shape: (batch, m, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, m, n, head_num * key_dim)
    # shape: (batch, m, n, head_num*key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, 1, problem, embedding)

        added = input1 + input2
        # shape: (batch, 1, problem, embedding)

        transposed = added.transpose(2, 3)
        # shape: (batch, 1, embedding, problem)

        normalized = self.norm(transposed.squeeze(1)).unsqueeze(1)
        # shape: (batch, 1, embedding, problem)

        back_trans = normalized.transpose(2, 3)
        # shape: (batch, 1, problem, embedding)

        return back_trans

class AddAndInstanceNormalization2(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm2d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, pomo, agent, embedding)

        added = input1 + input2
        # shape: (batch, pomo, agent, embedding)

        transposed = added.transpose(2, 3)
        # shape: (batch, pomo, embedding, agent)

        transposed = transposed.transpose(1, 2)
        # shape: (batch, embedding, pomo, agent)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, pomo, agent)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, pomo, embedding, agent)

        back_trans = back_trans.transpose(2, 3)
        # shape: (batch, pomo, agent, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, 1, problem, embedding) or (batch, pomo, agent, embedding)

        return self.W2(F.relu(self.W1(input1)))
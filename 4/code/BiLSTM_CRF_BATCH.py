import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence,pad_sequence,pack_padded_sequence
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

use_gpu = torch.cuda.is_available()
torch.manual_seed(1)
EMBEDDING_DIM = 64
HIDDEN_DIM = 32
START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNK = "<UNK>"
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seqs, word_to_ix):
    idxs = [torch.tensor([word_to_ix.get(w,word_to_ix.get(UNK)) for w in seq],dtype=torch.long) for seq in seqs]
    lengths_ = list(map(len,idxs))
    _,idx_sort = torch.sort(torch.tensor(lengths_),descending=True)
    idxs = sorted(idxs,key=lambda x:len(x),reverse=True)
    lengths = list(map(len,idxs))
    if len(seqs):
        idxs = pad_sequence(idxs,batch_first=True)

    return idxs,lengths,idx_sort


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_ids = torch.argmax(vec, dim=1).view(-1,1)
    max_score = torch.gather(vec, 1, max_ids)
    max_score_broadcast = max_score.view(vec.size()[0], -1).repeat(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1).unsqueeze(1))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size-1)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size-1, self.tagset_size-1))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 2, self.hidden_dim // 2,device=device),
                torch.randn(2, 2, self.hidden_dim // 2,device=device))

    def _forward_alg(self, feats,lengths):
        if len(feats.size()) == 2:
            feats = feats.unsqueeze(0)
        # Initialize the viterbi variables in log space

        batch_size = feats.size()[0]

        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((batch_size, self.tagset_size-1), -10000.).to(device)
        # START_TAG has all of the score.

        init_alphas[:, self.tag_to_ix[START_TAG]] = 0
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas


        forward_var_table = torch.zeros((len(lengths), lengths[0], self.tagset_size - 1)).to(device)
        # Iterate through the sentence
        for i in range(lengths[0]):
            feat = feats[:, i, :]
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size-1):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                temp = feat[:, next_tag]
                temp = temp.view(batch_size,-1)
                temp = temp.repeat(1,self.tagset_size-1)
                emit_score = temp
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag]

                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                result = log_sum_exp(next_tag_var)

                alphas_t.append(result)
            forward_var = torch.cat(alphas_t,1)

            forward_var_table[:, i, :] = forward_var
        index = torch.tensor(lengths).reshape(len(lengths), 1, 1).repeat(1, 1, self.tagset_size - 1) - 1
        final_forward_var = torch.gather(forward_var_table, dim=1, index=index.to(device))
        terminal_var = final_forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var = terminal_var.squeeze()
        alpha = log_sum_exp(terminal_var).squeeze(1)
        return alpha

    def _get_lstm_features(self, sentence,lengths):
        sentence = sentence.to(device)
        self.hidden = self.init_hidden()
        if len(sentence)>1:
            embeds = self.word_embeds(sentence)
            lstm_out, self.hidden = self.lstm(embeds)
            embeds = pack_padded_sequence(embeds, lengths=lengths, batch_first=True)
            lstm_out, self.hidden = self.lstm(embeds)
            lstm_out = pad_packed_sequence(lstm_out,batch_first=True)[0]
            lstm_feats = self.hidden2tag(lstm_out)
        else:
            embeds = self.word_embeds(sentence).view(sentence.size()[1], 1, -1)
            lstm_out, self.hidden = self.lstm(embeds)
            lstm_out = lstm_out.view(sentence.size()[1], self.hidden_dim)
            lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags, lengths):
        # Gives the score of a provided tag sequence
        cre, cre_maxtrix = self.mask_maxtric(lengths)
        score = torch.zeros((feats.size()[0], 1)).to(device)
        start = torch.ones((feats.size()[0], 1), dtype=torch.long).to(device)*self.tag_to_ix[START_TAG]
        tags = torch.cat([start, tags], dim=1).to(device)
        for i in range(lengths[0]):

            feat = feats[:,i, :]
            trans_ = self.transitions[tags[:,i + 1]]
            trans_ = torch.gather(trans_,1,tags[:,i].unsqueeze(1))
            trans = cre[:,i].unsqueeze(1) * trans_
            idxs = tags[:, i + 1].unsqueeze(1)
            emit_ = torch.gather(feat,1,idxs)
            emit = cre[:, i].unsqueeze(1) * emit_
            score = score + trans +emit
        last_tags = torch.gather(tags, 1, torch.tensor(lengths,device=device).unsqueeze(1))
        last_score = self.transitions[self.tag_to_ix[STOP_TAG], last_tags]
        score = score + last_score
        return score.squeeze(1)

    def mask_maxtric(self,lengths):
        """
        创建一个矩阵，矩阵维度batch_size*max_len
        1的个数为该句子的长度
        [[1,1,1,1,1],
        [1,1,1,0,0],
        [1,1,0,0,0],
        ...]
        :param lengths:
        :return:
        """
        cre = torch.zeros((len(lengths), lengths[0]))
        cre_maxtrix = torch.ones((len(lengths), lengths[0],len(self.tag_to_ix)))
        for i, lens in enumerate(lengths):
            one = torch.zeros(len(self.tag_to_ix))
            one[-1] = 1
            cre[i][:lens] = 1
            cre_maxtrix[i][lens-1:] = one

        return cre.to(device),cre_maxtrix.to(device)



    def _viterbi_decode(self, feats, lengths, mode=None):

        if len(feats.size()) == 2:
            feats = feats.unsqueeze(0)
        # Initialize the viterbi variables in log space

        batch_size = feats.size()[0]
        init_vvars = torch.full((batch_size, self.tagset_size-1), -10000.).to(device)
        init_vvars[:, self.tag_to_ix[START_TAG]] = 0
        backpointers = []

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        forward_var_table = torch.zeros((len(lengths),lengths[0],self.tagset_size-1)).to(device)
        for i in range(lengths[0]):
            feat = feats[:,i,:]
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            for next_tag in range(self.tagset_size-1):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)

                next_tag_var = forward_var +self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var,dim=1).unsqueeze(1)
                bptrs_t.append(best_tag_id)
                best_node_score = torch.gather(next_tag_var, dim=1, index=best_tag_id)

                viterbivars_t.append(best_node_score)

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # result = cre[:, i].unsqueeze(1) * feat
            forward_var = torch.cat(viterbivars_t, 1) + feat
            forward_var_table[:, i, :] = forward_var
            backpointers.append(torch.cat(bptrs_t, 1))
        index = torch.tensor(lengths).reshape(len(lengths),1,1).repeat(1,1,self.tagset_size-1)-1
        final_forward_var = torch.gather(forward_var_table,dim=1,index=index.to(device))

        # Transition to STOP_TAG
        terminal_var = final_forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var = terminal_var.squeeze(1)
        best_tag_id = torch.argmax(terminal_var, dim=1).unsqueeze(1)
        path_score = torch.gather(terminal_var, dim=1,index=best_tag_id)
        # Follow the back pointers to decode the best path.
        # best_path = [best_tag_id]
        if not mode: #如果mode是训练的话那么只用返回分数
            return path_score,[]
        best_path = [best_tag_id]

        backpointers_mat = torch.cat(backpointers,1).to(device).reshape(feats.size()[0],-1,self.tagset_size-1) #
        backpointers_mat = torch.cat([backpointers_mat,torch.ones(len(lengths),lengths[0],1,device=device).long()*(self.tagset_size-1)],-1) #给每一个backpointers加一个padding

        for i,length in enumerate(lengths):
            backpointers_mat[i][lengths[0]-length:]=backpointers_mat.clone()[i][:length]
            backpointers_mat[i][:lengths[0] - length]=self.tagset_size-1
        for i in range(backpointers_mat.size()[1]-1,-1,-1):
            bptrs_t = backpointers_mat[:,i,:]
            best_tag_id = torch.gather(bptrs_t, dim=1, index=best_tag_id)
            best_path.append(best_tag_id)

        best_path.reverse()
        best_path = torch.cat(best_path, 1)
        last_paths = []

        for i in range(len(lengths)):

            path = best_path[i][lengths[0]-lengths[i]:]
            if path[0] == self.tag_to_ix[START_TAG]:
                last_paths.append(path[1:])
            else:
                last_paths.append(None)

        return path_score, last_paths



    def neg_log_likelihood(self, sentence, tags,lengths):
        feats = self._get_lstm_features(sentence,lengths)
        forward_score = self._forward_alg(feats,lengths)
        gold_score = self._score_sentence(feats, tags,lengths)
        return torch.mean(forward_score - gold_score)



    def forward(self, sentence,lengths,mode=None):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence,lengths)
        # Find the best path, given the features.
        # score, tag_seq = self._viterbi_decode(lstm_feats)
        score, tag_seq = self._viterbi_decode(lstm_feats,lengths,mode=mode)
        return score, tag_seq

if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    PAD = "<PAD>"
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 32

    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )
    ]


    training_datas=[]


    for _ in range(8):
        training_datas.append(training_data.copy())

    x_trains =[]
    tag_trains = []
    for train_data in training_datas:

        x_trains=x_trains+[train_data[0][0],train_data[1][0]]
        tag_trains = tag_trains+[train_data[0][1],train_data[1][1]]

    word_to_ix = {}

    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4, "pad":5}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        precheck_sent,lengths,idx_sort = prepare_sequence([training_data[0][0],training_data[1][0]], word_to_ix)
        # precheck_sent,lengths,idx_sort = prepare_sequence(training_data[0][0], word_to_ix)
        precheck_tags = [torch.tensor([tag_to_ix[t] for t in tags],dtype=torch.long,device=device) for tags in [training_data[0][1],training_data[1][1]]]
        if len(precheck_sent)>1:
            precheck_tags = pad_sequence(precheck_tags,batch_first=True)
        print(model(precheck_sent.to(device),lengths,mode="train"))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded

    import time
    start_time = time.time()
    for epoch in range(
            500):  # again, normally you would NOT do 300 epochs, it is toy data
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        # sentence_in, lengths,idx_sort = prepare_sequence([training_data[0][0],training_data[1][0]], word_to_ix)
        # targets = [torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long) for tags in [training_data[0][1],training_data[1][1]]]
        # targets = pad_sequence(targets,batch_first=True)[idx_sort]
        sentence_in, lengths,idx_sort = prepare_sequence(x_trains, word_to_ix)
        targets = [torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long) for tags in tag_trains]
        targets = pad_sequence(targets,batch_first=True)[idx_sort]

        # Step 3. Run our forward pass.
        # if use_gpu:
        #     sentence_in = sentence_in.cuda()
        #     targets = targets.cuda()
        loss = model.neg_log_likelihood(sentence_in.to(device), targets.to(device),lengths)
        # if use_gpu:
        #
        #     loss=loss.cuda()

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
    # torch.save(model.state_dict(),"./bi_crf.pt")
    # Check predictions after training
    # model.load_state_dict(torch.load("./bi_crf.pt"))
    with torch.no_grad():
        precheck_sent,lenghts,_ = prepare_sequence([training_data[0][0]], word_to_ix)
        print(model(precheck_sent.to(device),lenghts,mode="dev"))
    print("耗时:",time.time()-start_time)
    # We got it!
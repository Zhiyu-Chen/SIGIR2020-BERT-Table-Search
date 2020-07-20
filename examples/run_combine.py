# coding=utf-8
from __future__ import absolute_import, division, print_function

import csv
import os
import sys
import json
from io import open
import argparse
import glob
import random
import shutil
from glob import glob
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
sys.path.append('../')
from pytorch_transformers import (WEIGHTS_NAME,
                                  BertConfig,
                                  BertTokenizer,
                                  BertForSequenceClassification,BertForHybridTable, # BERT Model
                                  )


from pytorch_transformers import AdamW, WarmupLinearSchedule
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from trec import TREC_evaluator
import pandas as pd
import datetime
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor,ElasticNet,LogisticRegression
from sklearn.svm import SVR
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch import nn
from torch.nn import MSELoss
from torch import optim
import fasttext
from scipy.spatial.distance import cosine
import re



MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def select_content(args,query,cols):
    pattern = re.compile('[\W_]+')
    query = query.lower().split()
    col_scores = [0] * len(cols)
    if args.selector == 'SUM':
        for q_token in query:
            q_token_vec = args.ft_model.get_word_vector(q_token)
            for idx,col in enumerate(cols):
                col_score = 0
                for col_token in col:
                    col_token = pattern.sub(' ', col_token.lower())
                    col_token = col_token.split()
                    col_token_vec = np.mean([args.ft_model.get_word_vector(each) for each in col_token],axis=0)
                    col_score = col_score + 1 - cosine(col_token_vec,q_token_vec)
                    col_scores[idx] += col_score

    elif args.selector == 'AVG':
        q_vec = np.mean([args.ft_model.get_word_vector(each) for each in query],axis=0)
        for idx, col in enumerate(cols):
            col_vecs = []
            for cell in col:
                cell_token = pattern.sub(' ', cell.lower())
                cell_token = cell_token.split()
                col_token_vec =np.mean([args.ft_model.get_word_vector(each) for each in cell_token],axis=0)
                col_vecs.append(col_token_vec)
            col_vec = np.mean(col_vecs,axis=0)
            col_scores[idx] = 1 - cosine(q_vec,col_vec)
    elif args.selector == 'MAX':
        for idx, col in enumerate(cols):
            all_sims = []
            for q_token in query:
                for cell in col:
                    cell_token = pattern.sub(' ', cell.lower())
                    cell_token = cell_token.split()
                    cell_vec = np.mean([args.ft_model.get_word_vector(each) for each in cell_token], axis=0)
                    all_sims.append(1 - cosine(args.ft_model.get_word_vector(q_token),cell_vec))
            col_scores[idx] = np.max(all_sims)

    #find the target
    col_scores = np.array(col_scores)
    #top_k = 5 if 5 < len(cols) else len(cols)
    top_k = len(cols)
    indices = (-col_scores).argsort()[:top_k]
    return [cols[each] for each in indices]


class Regressor(nn.Module):
    def __init__(self,hidden_size):
        super(Regressor,self).__init__()
        self.classifier = nn.Linear(hidden_size,1)
        self.apply(self.init_weights)


    def forward(self,features,labels):
        logits = self.classifier(features)
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
        outputs = loss,logits
        return outputs


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, qid,docid, query, feature,table=None, label=None):
        self.guid = guid
        self.query = query
        self.table = table # sentence pair
        self.label = label
        self.feature = feature
        self.qid = qid
        self.docid = docid


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, features,input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.features = features
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def cross_split_pairs(i_k,k,X):
    num_samples = X.shape[0]
    test_start_idx = int(num_samples * i_k / k)
    test_end_idx = int(num_samples * (i_k + 1) / k)
    test_idx = list(range(test_start_idx, test_end_idx))
    train_idx = [each for each in range(num_samples) if each not in test_idx]
    train_df = X[train_idx]
    test_df = X[test_idx]
    return train_df,test_df


class TableProcessor(object):
    """
        Processor for the WWW Tables
    """
    def __init__(self, data_dir='../data/'):
        """
        if resplit == True, create new split for train/test/dev
        """
        self.data_dir = data_dir
        self.all_df = self.get_www_all_features()

    def get_www_all_features(self,feature_file='../data/features.csv'):
        ids_left = []
        ids_right = []
        features = []
        labels = []
        f_f = open(feature_file, 'r')
        line = f_f.readline()
        for line in f_f:
            seps = line.strip().split(',')
            qid = seps[0]
            tid = seps[2]
            ids_left.append(qid)
            ids_right.append(tid)
            rel = seps[-1]
            labels.append(int(rel))
            '''
            if int(rel) > 0:
                labels.append(1)
            else:
                labels.append(0)
            '''
            q_doc_f = np.array([float(each) for each in seps[3:-1]])
            features.append(q_doc_f)

        df = pd.DataFrame({
            'qid': ids_left,
            'docid': ids_right,
            'features': features,
            'label': labels
        })
        return df


    def _read_jsonl(self, input_file):
        """Reads jsonl file."""
        data = []
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                line = json.loads(line.strip())
                lines.append(line)
                data.append([line[each] for each in ['rel', 'qid', 'docid', 'query', 'table']])

        df = pd.DataFrame(data, columns=['rel', 'qid', 'docid', 'query', 'table'])
        final_df = pd.merge(df, self.all_df, on=['qid', 'docid'])
        return final_df

    def get_fold_train_examples(self,data_dir,k):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, str(k) + "_train.jsonl")), 'fold'+str(k) + "_train")

    def get_fold_test_examples(self,data_dir,k):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, str(k) + "_test.jsonl")), 'fold'+str(k) + "_test")


    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "final_test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in df.iterrows():
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid,qid = line['qid'],docid=line['docid'],query=line['query'],feature=line['features'], table=line['table'], label=line['rel']))
        return examples


def convert_examples_to_features(args,examples,  max_seq_length,tokenizer,cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        """
            Logic in here:

            [CLS] query + [SEP] + caption + [SEP] + pg_title + [SEP] + sec_title + [SEP] + schema + [SEP]
            |   segment_a  |                                segment_b                                   |

        """
        tokens_query = tokenizer.tokenize(example.query)
        #add query
        tokens = tokens_query + [sep_token] #query + [SEP]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]  #query + [SEP][SEP]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        #add caption
        if args.use_caption and example.table['caption']:
            tokens_caption = tokenizer.tokenize(example.table['caption'])
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens_caption += [sep_token]
            tokens += tokens_caption + [sep_token] #[CLS] query + [SEP] + caption + [SEP]
            segment_ids += [sequence_b_segment_id] * (len(tokens_caption) + 1)  # query + [SEP] + caption + [SEP]

        ## add pg_title
        if args.use_pg_title and example.table['pg_title']:
            tokens_pg_title = tokenizer.tokenize(example.table['pg_title'])
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens_pg_title += [sep_token]
            tokens += tokens_pg_title + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_pg_title) + 1)

        ## add sec_title
        if args.use_sec_title and example.table['sec_title']:
            tokens_sec_title = tokenizer.tokenize(example.table['sec_title'])
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens_sec_title += [sep_token]
            tokens += tokens_sec_title + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_sec_title) + 1)

        schema_token_lens = 0
        rest_space = max_seq_length - len(tokens)
        ## add schema
        ## add schema
        if args.use_schema and example.table['raw_json']:
            schemas = json.loads(example.table['raw_json'])['title']
            if args.schema == 'DEL':
                for schema in schemas:
                    token_schema = tokenizer.tokenize(schema)
                    if len(token_schema) + 2 < rest_space:
                        if sep_token_extra:
                            # roberta uses an extra separator b/w pairs of sentences
                            token_schema += [sep_token]
                        tokens += token_schema + [sep_token]
                        schema_token_lens += len(token_schema)
                        segment_ids += [sequence_b_segment_id] * (len(token_schema) + 1)
                        rest_space -= (len(token_schema) + 2)
                    else:
                        break
            elif args.schema == 'SEP':
                schemas = ' '.join(schemas)
                token_schema = tokenizer.tokenize(schemas)
                #truncate schemas
                while len(token_schema) > rest_space-2:
                    token_schema.pop()
                tokens += token_schema + [sep_token]
                schema_token_lens += len(token_schema)
                segment_ids += [sequence_b_segment_id] * (len(token_schema) + 1)

        if args.content:
            rest_space = max_seq_length - len(tokens)-1
            table_data = json.loads(example.table['raw_json'])['data']
            headers = json.loads(example.table['raw_json'])['title']
            query = example.query
            if args.content == 'COL':
                cols = [headers] + table_data
                cols = list(map(list, zip(*cols)))
                #find the most relevant columns
                similar_items = select_content(args,query,cols)
            elif args.content == 'ROW':
                similar_items = select_content(args, query, table_data)
            elif args.content == 'RAND_COL':
                cols = [headers] + table_data
                cols = list(map(list, zip(*cols)))
                random.shuffle(cols)
                # top_k = 5 if 5 < len(cols) else len(cols)
                top_k = len(cols)
                similar_items = cols[:top_k]
            elif args.content == 'RAND_ROW':
                random.shuffle(table_data)
                # top_k = 5 if 5 < len(table_data) else len(table_data)
                top_k = len(table_data)
                similar_items = table_data[:top_k]
            for item in similar_items:
                col = ' '.join(item)
                col_token = tokenizer.tokenize(col)
                if len(col_token) + 1 < rest_space:
                    if sep_token_extra:
                        # roberta uses an extra separator b/w pairs of sentences
                        col_token += [sep_token]
                    tokens += col_token + [sep_token]
                    segment_ids += [sequence_b_segment_id] * (len(col_token) + 1)
                    rest_space -= (len(col_token) + 1)

        #### handle different conventions in xlnet and bert.
        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # print("original input ids length {0}".format(len(input_ids)))
        # print(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        # print("input ids length {0}".format(len(input_ids)))
        # print("token length {0}".format(len(tokens)))
        # print("padding  length {0}".format(padding_length))
        # print("caption len {0}".format(len(tokens_caption)))
        # print("schema len {0}".format(schema_token_lens))
        # print("tokens_pg_title len {0}".format(len(tokens_pg_title)))
        # print("tokens_sec_title len {0}".format(len(tokens_sec_title)))
        # print("tokens_query len {0}".format(len(tokens_query)))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if args.mode =='regression':
            label_id = float(example.label)
        else:
            label_id = int(example.label)
            if label_id > 0:
                label_id = 1

            #label_id = int(example.label)


        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def load_and_cache_examples(args,tokenizer, split="train"):

    if args.local_rank not in [-1, 0] and split != "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache


    processor = TableProcessor(args.data_dir)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
        split,
        args.exp_name,
        str(args.max_seq_length)))
    #### UPDATE: we load examples in any case
    if split == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif split == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    elif split == "test":
        examples = processor.get_test_examples(args.data_dir)
    elif split.split('_')[1] == "train":
        examples = processor.get_fold_train_examples(args.data_dir,split.split('_')[0])
    elif split.split('_')[1] == "test":
        examples = processor.get_fold_test_examples(args.data_dir, split.split('_')[0])


    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        print("error")
        exit()
        # features = convert_examples_to_features(args,examples, args.max_seq_length, tokenizer,
        #     cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        #     cls_token=tokenizer.cls_token,
        #     cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
        #     sep_token=tokenizer.sep_token,
        #     sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        #     pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        #     pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        #     pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        # )
        # if args.local_rank in [-1, 0]:
        #     torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_features = torch.tensor([f.feature for f in examples],dtype=torch.float)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if args.mode =='regression':
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    else:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids,all_features, all_input_mask, all_segment_ids, all_label_ids)
    return dataset, examples # also return examples for post-processing


def train(args,regr,train_dataset,classifier,tokenizer):
    if args.local_rank in [-1, 0]:
        if args.k_fold != 0:
            tb_fname = os.path.join('./runs',args.exp_name + '_fold_' + str(args.fold) + '_' +classifier)
        else:
            tb_fname = os.path.join('./runs', args.exp_name+ '_' +classifier)
        if os.path.exists(tb_fname):
            shutil.rmtree(tb_fname)
        tb_writer = SummaryWriter(logdir=tb_fname)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    optimizer = optim.Adam(regr.parameters(), lr=args.learning_rate)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    regr.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    epoch_num = 0
    min_loss = float('inf')
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            regr.train()
            ## model inputs
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'features':batch[0],'labels':batch[1]}
            outputs =regr(**inputs)
            #regr(features=batch[0],labels=batch[1])
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(regr.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            optimizer.step()
            regr.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # train_results = evaluate(args, model, tokenizer, prefix='train: ' + str(global_step) + ' :epoch ' + str(epoch_num),split='train')
                    # for key, value in train_results.items():
                    #     tb_writer.add_scalar('train_eval_{}'.format(key), value, global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss



        epoch_num += 1


    if args.local_rank in [-1, 0]:
        tb_writer.close()
    #del loss
    return global_step, tr_loss / global_step





def get_BERT_features(args, model, tokenizer, prefix="",split="test",ltr=False,bert_layers = [-1],bert_mode = 'cls',pool = 'avg',normalize=False,scaler=None):
    eval_dataset, eval_examples = load_and_cache_examples(args, tokenizer, split)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    all_features = []
    all_labels = []
    all_cls =[]

    for batch in tqdm(eval_dataloader, desc="Extracting features"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[2],
                      'token_type_ids': batch[3] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[4]}
            tmp_eval_loss, logits, hidden_states,attentions = model(**inputs)
            #(loss), logits, (hidden_states), (attentions) every hidden_state has size [16, 128, 1024]
            if len(bert_layers) == 1:
                if bert_mode == 'cls':
                    bert_feature = hidden_states[bert_layers[0]][:, 0, :].detach().cpu().numpy()
                elif bert_mode == 'hidden':
                    bert_feature = torch.mean(hidden_states[bert_layers[0]],dim=1).detach().cpu().numpy()
                elif bert_mode == 'sep':
                    sentence_b_start = inputs['token_type_ids'][0].tolist().index(1)
                    len_a = sentence_b_start
                    token_ids = [int(each) for each in inputs['input_ids'][0]]
                    tokens = list(map(tokenizer._convert_id_to_token, token_ids))
                    tokens = list(filter(lambda x: x != '[PAD]', tokens))
                    for idx, token in enumerate(tokens):
                        if token == '[SEP]':
                            break
                    sep_idx = idx
                    bert_feature = hidden_states[bert_layers[0]][:, sep_idx, :].detach().cpu().numpy()
            else:
                if pool == 'cat' and bert_mode == 'cls':
                    bert_feature = torch.cat(tuple([hidden_states[i] for i in bert_layers]),dim=-1)
                    bert_feature = bert_feature[:, 0, :].detach().cpu().numpy()
                elif pool == 'avg' and bert_mode == 'cls':
                    bert_feature = torch.mean(torch.stack([hidden_states[i][:, 0, :] for i in bert_layers]),dim=0).detach().cpu().numpy()
                else:
                    print("wrong way to calculate bert features !")
                    exit()
            labels = inputs['labels'].detach().cpu().numpy()
            all_cls.append(bert_feature)
            all_features.append(batch[1].detach().cpu().numpy())
            all_labels.append(labels)

    #normalize STR features
    packed_features = [each for batch in all_features for each in batch]
    if normalize and split.split('_')[1] == 'train':
        scaler = preprocessing.StandardScaler()
        packed_features = scaler.fit_transform(packed_features)
    elif normalize and split.split('_')[1] == 'test':
        packed_features = scaler.transform(packed_features)


    packed_cls = [each for batch in all_cls for each in batch]
    packed_labels = [l for batch in all_labels for l in batch]
    final_features = []
    for i in range(len(packed_features)):
        if ltr:
            final_features.append(np.concatenate((packed_features[i], packed_cls[i]), axis=0))
        else:
            final_features.append(packed_cls[i])


    if split.split('_')[1] == 'test':
        # get scores or results depending on the task
        qids = [eg.qid for eg in eval_examples]
        docids = [eg.docid for eg in eval_examples]
        return qids,docids,final_features,packed_labels
    elif split.split('_')[1] == 'train':
        return final_features,packed_labels,scaler



def select_fold_checkpoint(args,model_type,best_checkpoint=False):
    '''
    select the checkpoint of the best fold for certain method type
    '''
    method = ''
    if args.use_caption:
        method = method + 'caption'
    if args.use_pg_title:
        method = method + '_pgTitle'
    if args.use_sec_title:
        method = method + '_secTitle'
    if args.use_schema:
        method = method + '_schema'
    ndcg5s = []
    for fold in range(1, args.k_fold + 1):
        rs_path = os.path.join(args.original_output_dir, model_type, model_type + '_' + method, str(fold) + '_fold',
                               'cache_rs_fold_' + str(fold) + '.bin')
        rs = torch.load(rs_path)
        ndcg5s.append(rs['ndcg_cut_5'])
    best_fold = np.argmax(ndcg5s) + 1
    return os.path.join(args.original_output_dir,model_type,model_type + '_' + method,str(best_fold) + '_fold')


def evaluate(args, regr, eval_dataset,qids,docids, prefix="",split="test"):
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    scores = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        regr.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'features':      batch[0],'labels':         batch[1]}
            outputs = regr(**inputs)
            #outputs = model(features=batch[0],labels=batch[1])
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if scores is None:
            scores = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            scores = np.append(scores, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    # softmax score, for further processing (ensemble)

    if args.mode == 'regression':
        preds = scores.squeeze()
    else:
        scores = np.exp(scores)
        scores = scores / scores.sum(-1, keepdims=True)
        preds = np.argmax(scores, axis=1) # only get logits for true class
    #print(preds)
    #print(preds.shape)
    eval_df = pd.DataFrame(data={
        'id_left': qids,
        'id_right': docids,
        'true': out_label_ids,
        'pred': preds
    })
    ltr_metric_scores = defaultdict(list)

    #### Run evaluation
    trec_eval = TREC_evaluator(run_id='combine_' + args.exp_name + '_' + split, base_path=args.output_dir)
    trec_eval.write_trec_result(eval_df)
    ndcgs = trec_eval.get_ndcgs('all_trec')
    for metric in ndcgs:
        ltr_metric_scores[metric].append(ndcgs[metric])
    ltr_metric_scores["eval_loss"] = eval_loss / nb_eval_steps
    return ltr_metric_scores

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--fasttext_dir", default=None, type=str, required=True,
                        help="path to the fasttext embedding.")
    parser.add_argument("--k_fold", default=5, type=int, required=False,
                        help="k fold cross validation.")
    parser.add_argument("--feature_len", default=39, type=int, required=False,
                        help="number of additional features.")
    parser.add_argument("--trans_len", default=39, type=int, required=False,
                        help="number of additional features.")
    parser.add_argument("--resplit",action='store_true')
    parser.add_argument("--mode", default='regression', type=str, required=False,
                        help="regression or classification.")
    parser.add_argument("--schema", default='SEP', type=str, required=True,
                        help="how to concatenate the schema labels.")
    parser.add_argument("--content", default=None, type=str, required=False,
                        help="how to concatenate the table content:ORDER, ROW,COL,RAND_ROW,RAND_COL")
    parser.add_argument("--selector", default=None, type=str, required=False,
                        help="how to select the table content:SUM,AVG,MAX. Valid when not randomly selecting content")
    parser.add_argument("--trunc", action='store_true', help="truncate all fields ")

    ## field options
    parser.add_argument("--use_caption",action='store_true',help="whether use table cation or not ")
    parser.add_argument("--use_sec_title",action='store_true',help="whether use table section title or not ")
    parser.add_argument("--use_pg_title",action='store_true',help="whether use table page title or not ")
    parser.add_argument("--use_schema",action='store_true',help="whether use table schemas or not ")

    parser.add_argument("--ignore_logits_layer", action='store_true',
                        help="whether to skip initialization of logits layers.")
    parser.add_argument("--ignore_sequence_summary_layer", action='store_true',
                        help="whether to skip initialization of sequence summary layers.")
    ## single task
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task")


    ## define model type
    parser.add_argument("--load_averaged_checkpoint", action='store_true',
                        help="Wether to average a directory of checkpoints to evaluate.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the test set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    #### learning rate difference between original BertAdam and now paramters.
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--lr_layer_decay", default=1.0, type=float,
                        help="layer learning rate decay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    #### model hyperparameters
    parser.add_argument('--alpha', type=float, default=5, help="pairwise maring ranking loss alpha")
    parser.add_argument('--beta', type=float, default=0.1, help="pairwise maring ranking loss beta")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()
    exp_name = args.model_name_or_path.split('/')[-1]
    if args.use_caption:
        exp_name = exp_name + '_caption'
    if args.use_pg_title:
        exp_name = exp_name + '_pgTitle'
    if args.use_sec_title:
        exp_name = exp_name + '_secTitle'
    if args.use_schema and args.schema:
        exp_name = exp_name + '_schema_' + args.schema
    if args.content:
        exp_name = exp_name + '_' + args.content
        if args.selector:
            exp_name = exp_name + '_' + args.selector
        args.ft_model = fasttext.FastText.load_model(args.fasttext_dir)

    if args.trunc:
        exp_name = exp_name + '_' + 'cut'

    if args.max_seq_length > 128:
        exp_name = exp_name + '_' + str(args.max_seq_length)

    if args.mode == 'classification':
        exp_name = exp_name + '_c'
    #exp_name = exp_name + str(datetime.datetime.now())
    args.exp_name =  exp_name
    args.original_output_dir = args.output_dir

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))


    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device


    if not os.path.isdir(args.output_dir): os.mkdir(args.output_dir)
    set_seed(args)

    processor = TableProcessor(args.data_dir)
    label_list = processor.get_labels()
    #regression
    #regression
    if args.mode == 'regression':
        args.num_labels = num_labels = 1
    elif args.mode == 'classification':
        args.num_labels = num_labels = 2
    else:
        print('number label error !!')
        exit()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    """
        add param: 
            ignore_logits_layer:            whether to ignore the logits layer
            ignore_sequence_summary_layer:  whether to ignore the sequence_summary_layer
            load_averaged_checkpoint:       whether load a model with averaged weights
            ...
    """

    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)


    #feature set up
    # each tuple -> LTR, (BERT layers, mode, pool)
    all_settings = []
    #bert_settings = [ [[-1],'cls',''],[[-1],'hidden',''],[[-1,-2,-3,-4],'cls','cat'],[[-1,-2,-3,-4],'cls','avg'] ]
   #bert_settings= [[[-1],'sep',''],[[-1],'cls','']]
    bert_settings = [[[-1], 'sep', '']]
    for ltr_flag in [False]:
        for each_bert in bert_settings:
            all_settings.append([ltr_flag,each_bert[0],each_bert[1],each_bert[2]])


    validation = False
    normalize = True
    classifier = 'mlp'
    f = open('./runs/'+ classifier +'_combine_content.csv', 'a')
    if not validation:
        for each_setting in all_settings:
            ltr_metric_scores = defaultdict(list)
            for fold in range(1,6):
                config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                                      num_labels=num_labels, finetuning_task=args.task_name)
                tokenizer = tokenizer_class.from_pretrained(
                    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                    do_lower_case=args.do_lower_case)

                args.fold = fold
                args.output_dir = os.path.join(args.original_output_dir, args.exp_name, str(fold) + '_fold')
                args.model_name_or_path = args.output_dir #os.path.join(args.output_dir,'best_checkpoint')
                # load model
                config.output_hidden_states = True
                config.output_attentions = True
                print("load model from {}".format(args.model_name_or_path))
                model = model_class.from_pretrained(args.model_name_or_path,
                                                    from_tf=bool('.ckpt' in args.model_name_or_path),
                                                    config=config,
                                                    ignore_logits_layer=args.ignore_logits_layer,
                                                    ignore_sequence_summary_layer=args.ignore_sequence_summary_layer,
                                                    load_averaged_checkpoint=args.load_averaged_checkpoint,
                                                    feature_len=args.feature_len, trans_len=args.trans_len)

                if args.local_rank == 0:
                    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

                model.to(args.device)

                final_features,all_labels,scaler = get_BERT_features(args, model, tokenizer, prefix="",split=str(fold) + "_train",
                                                              ltr=each_setting[0],bert_layers=each_setting[1],bert_mode=each_setting[2],pool=each_setting[3],
                                                              normalize=normalize)
                # training
                if classifier == 'rf':
                    #regr = RandomForestRegressor(random_state=0, n_estimators=1000, n_jobs=-1)
                    regr = RandomForestRegressor(random_state=0, n_estimators=1000,max_depth =3, n_jobs=-1)
                    regr.fit(final_features, all_labels)

                    #evaluate
                    qids, docids, final_features, all_labels = get_BERT_features(args, model, tokenizer, prefix="", split=str(fold) + "_test",
                                                                                 ltr=each_setting[0],
                                                                                 bert_layers=each_setting[1],
                                                                                 bert_mode=each_setting[2], pool=each_setting[3],
                                                                                 normalize=normalize,scaler=scaler)


                    y_preds = regr.predict(final_features)
                    # evaluation
                    eval_df = pd.DataFrame(data={
                        'id_left': qids,
                        'id_right': docids,
                        'true': all_labels,
                        'pred': y_preds.squeeze()
                    })
                    trec_eval = TREC_evaluator(run_id='combine_' + args.exp_name + '_test', base_path=args.output_dir)
                    trec_eval.write_trec_result(eval_df)
                    ndcgs = trec_eval.get_ndcgs(metrics='all_trec')
                    for metric in ndcgs:
                        ltr_metric_scores[metric].append(ndcgs[metric])
                        print("fold {0} - metric {1} - score {2}".format(fold,metric,ndcgs[metric]))
                elif classifier == 'mlp':
                    regr = Regressor(final_features[0].shape[0])
                    regr.to(args.device)
                    all_labels = torch.tensor([l for l in all_labels], dtype=torch.float)
                    all_features = torch.tensor([each for each in final_features],dtype=torch.float)
                    train_dataset = TensorDataset(all_features,all_labels)
                    global_step, tr_loss = train(args,regr,train_dataset,classifier,tokenizer)
                    #evaluate
                    qids, docids, final_features, all_labels = get_BERT_features(args, model, tokenizer, prefix="", split=str(fold) + "_test",
                                                                                 ltr=each_setting[0],
                                                                                 bert_layers=each_setting[1],
                                                                                 bert_mode=each_setting[2], pool=each_setting[3],normalize=normalize,scaler=scaler)
                    # print(sum(final_features[0]))
                    # print(final_features[0][:39])
                    all_labels = torch.tensor([l for l in all_labels], dtype=torch.float)
                    all_features = torch.tensor([each for each in final_features],dtype=torch.float)
                    test_dataset = TensorDataset(all_features,all_labels)
                    ltr_metric_scores = evaluate(args, regr, test_dataset, qids,docids,prefix='fold ' + str(fold), split=str(fold) + '_test')
            scores = ''
            for metric in ltr_metric_scores:
                print("{0}:{1}".format(metric, np.mean(ltr_metric_scores[metric])))
                scores = scores  + str(np.mean(ltr_metric_scores[metric])) + ','

            #writing results
            #args.exp_name , ltr , layers, mode, pool, ndcg5, ...
            line = classifier + ':'+args.exp_name + ',' + str(each_setting[0]) + ',' +  str(each_setting[1]).replace(',',':') + ',' \
                   + each_setting[2] + ',' + each_setting[3] + ',' + \
                   scores + '\n'
            f.write(line)
        f.close()


if __name__ == '__main__':
    main()
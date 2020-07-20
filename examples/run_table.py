# coding=utf-8
from __future__ import absolute_import, division, print_function

import csv
import logging
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
from pytorch_transformers import (WEIGHTS_NAME,
                                  BertConfig,
                                  BertTokenizer,
                                  BertForSequenceClassification, # BERT Model
                                  )

sys.path.append('../')
from pytorch_transformers import AdamW, WarmupLinearSchedule
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from trec import TREC_evaluator
import pandas as pd
import datetime
import fasttext
from scipy.spatial.distance import cosine
import re

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, qid,docid, query, table=None, label=None):
        self.guid = guid
        self.query = query
        self.table = table # sentence pair
        self.label = label
        self.qid = qid
        self.docid = docid


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
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
    def __init__(self, data_dir='../data'):
        """
        if resplit == True, create new split for train/test/dev
        """
        self.data_dir = data_dir

    @staticmethod
    def create_k_fold(data_dir='../data',k=5):
        f = open(data_dir + '/all.json', 'r')
        X = []
        y = []
        for line in f:
            X.append(line)
            y.append(json.loads(line)['rel'])
        f.close()
        X = np.array(X)
        X = shuffle(X)
        # kf = KFold(n_splits=k,shuffle=False)
        # fold = 1
        # for train_index, test_index in kf.split(X):
        #     X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        #     f = open(os.path.join(data_dir,str(fold) + '_train.jsonl'),'w')
        #     for each in X_train:
        #         f.write(each)
        #     f.close()
        #     f = open(os.path.join(data_dir, str(fold) + '_test.jsonl'), 'w')
        #     for each in X_test:
        #         f.write(each)
        #     f.close()
        #     fold += 1
        # X, X_final = train_test_split(X, test_size = 0.2, shuffle=False)
        # f = open(os.path.join(data_dir,'final_test.jsonl'),'w')
        # for each in X_final:
        #     f.write(each)
        # f.close()
        for i_k in range(k):
            X_train, X_test = cross_split_pairs(i_k, k, X)
            f = open(os.path.join(data_dir,str(i_k+1) + '_train.jsonl'),'w')
            for each in X_train:
                f.write(each)
            f.close()
            f = open(os.path.join(data_dir, str(i_k+1) + '_test.jsonl'), 'w')
            for each in X_test:
                f.write(each)
            f.close()

    def _read_jsonl(self, input_file):
        """Reads jsonl file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(json.loads(line.strip()))
            return lines

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
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid,qid = line['qid'],docid=line['docid'],query=line['query'], table=line['table'], label=line['rel']))
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
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
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
            if args.trunc:
                tokens_caption = tokens_caption[:20]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens_caption += [sep_token]
            tokens += tokens_caption + [sep_token] #[CLS] query + [SEP] + caption + [SEP]
            segment_ids += [sequence_b_segment_id] * (len(tokens_caption) + 1)  # query + [SEP] + caption + [SEP]

        ## add pg_title
        if args.use_pg_title and example.table['pg_title']:
            tokens_pg_title = tokenizer.tokenize(example.table['pg_title'])
            if args.trunc:
                tokens_pg_title = tokens_pg_title[:10]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens_pg_title += [sep_token]
            tokens += tokens_pg_title + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_pg_title) + 1)

        ## add sec_title
        if args.use_sec_title and example.table['sec_title']:
            tokens_sec_title = tokenizer.tokenize(example.table['sec_title'])
            if args.trunc:
                tokens_sec_title = tokens_sec_title[:10]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens_sec_title += [sep_token]
            tokens += tokens_sec_title + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_sec_title) + 1)

        schema_token_lens = 0
        rest_space = max_seq_length - len(tokens)
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
                if args.trunc:
                    token_schema = token_schema[:20]
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
            elif args.content == 'RAND_CELL':
                if example.docid not in args.rand_content:
                    cols = [headers] + table_data
                    cells = []
                    for item in cols:
                        cells.extend(item)
                    #cells = list(set(cells))
                    # find the most relevant
                    random.shuffle(cells)
                    # top_k = 5 if 5 < len(cols) else len(cols)
                    top_k = len(cells)
                    similar_items = cells[:top_k]
                    args.rand_content[example.docid] = similar_items
                else:
                    similar_items = args.rand_content[example.docid]
            elif args.content == 'CELL':
                cols = [headers] + table_data
                cells = []
                for item in cols:
                    cells.extend(item)
                #cells = list(set(cells))
                cells = select_cells(args,query,cells)
                top_k = len(cells)
                similar_items = cells[:top_k]
            elif args.content == 'RAND_COL':
                if example.docid not in args.rand_content:
                    cols = [headers] + table_data
                    cols = list(map(list, zip(*cols)))
                    random.shuffle(cols)
                    # top_k = 5 if 5 < len(cols) else len(cols)
                    top_k = len(cols)
                    similar_items = cols[:top_k]
                    args.rand_content[example.docid] = similar_items
                else:
                    similar_items = args.rand_content[example.docid]
            elif args.content == 'RAND_ROW':
                if example.docid not in args.rand_content:
                    random.shuffle(table_data)
                    # top_k = 5 if 5 < len(table_data) else len(table_data)
                    top_k = len(table_data)
                    similar_items = table_data[:top_k]
                    args.rand_content[example.docid] = similar_items
                else:
                    similar_items = args.rand_content[example.docid]
            if 'CELL' not in args.content:
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
            else:
                for item in similar_items:
                    col = item
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

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def select_cells(args,query,cells):
    pattern = re.compile('[\W_]+')
    query = query.lower().split()
    cell_scores = [0] * len(cells)
    if args.selector == 'SUM':
        for q_token in query:
            q_token_vec = args.ft_model.get_word_vector(q_token)
            for idx, cell in enumerate(cells):
                cell_token = pattern.sub(' ', cell.lower())
                cell_token = cell_token.split()
                cell_vec = np.mean([args.ft_model.get_word_vector(each) for each in cell_token], axis=0)
                cell_score = 1 - cosine(cell_vec, q_token_vec)
                cell_scores[idx] += cell_score
    elif args.selector == 'AVG':
        q_vec = np.mean([args.ft_model.get_word_vector(each) for each in query], axis=0)
        for idx, cell in enumerate(cells):
            cell_token = pattern.sub(' ', cell.lower())
            cell_token = cell_token.split()
            cell_vec = np.mean([args.ft_model.get_word_vector(each) for each in cell_token], axis=0)
            cell_scores[idx] = 1 - cosine(q_vec, cell_vec)
    elif args.selector == 'MAX':
        for idx, cell in enumerate(cells):
            all_sims = []
            for q_token in query:
                cell_token = pattern.sub(' ', cell.lower())
                cell_token = cell_token.split()
                cell_vec = np.mean([args.ft_model.get_word_vector(each) for each in cell_token], axis=0)
                all_sims.append(1 - cosine(args.ft_model.get_word_vector(q_token),cell_vec))
            cell_scores[idx] = np.max(all_sims)
    # find the target
    top_k = len(cells)
    cell_scores = np.array(cell_scores)
    indices = (-cell_scores).argsort()[:top_k]
    return [cells[each] for each in indices]

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


    logger.info("load dataset %s",split)
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
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        features = convert_examples_to_features(args,examples, args.max_seq_length, tokenizer,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if args.mode =='regression':
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    else:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset, examples # also return examples for post-processing


def train(args, train_dataset, model, tokenizer):
    if args.local_rank in [-1, 0]:
        if args.k_fold != 0:
            tb_fname = os.path.join('./runs',args.exp_name + '_fold_' + str(args.fold))
        else:
            tb_fname = os.path.join('./runs', args.exp_name)
        if os.path.exists(tb_fname):
            shutil.rmtree(tb_fname)
        tb_writer = SummaryWriter(logdir=tb_fname)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    ##
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    """
        learning rate layer decay, hard coding for BERT, XLNet and RoBERTa.
    """

    def extract_n_layer(n, max_n_layer=-1):
        n = n.split('.')
        try:
            idx = n.index("layer")
            n_layer = int(n[idx + 1]) + 1
        except:
            if any(nd in n for nd in ["embeddings", "word_embedding", "mask_emb"]):
                n_layer = 0
            else:
                n_layer = max_n_layer
        return n_layer

    # we acquire the max_n_layer from inference,
    # we leave the sequence_summary layer and logits layer own same learning rate scale 1.
    # the lower 24 encoder layers shave decaying learning rate scare decay_scale ** (24-layer), layer ~ (0,23)
    max_n_layer = max([extract_n_layer(n) for n, p in model.named_parameters()]) + 1
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []  # group params by layers and weight_decay params.
    for n_layer in range(max_n_layer + 1):
        #### n_layer and decay
        optimizer_grouped_parameters.append({
            'params': [p for n, p in model.named_parameters() if (
                    extract_n_layer(n, max_n_layer=max_n_layer) == n_layer and not any(nd in n for nd in no_decay))],
            'weight_decay': args.weight_decay,
            'lr_decay': args.lr_layer_decay ** (max_n_layer - n_layer)
        })
        #### n_layer and no_decay
        optimizer_grouped_parameters.append({
            'params': [p for n, p in model.named_parameters() if (
                    extract_n_layer(n, max_n_layer=max_n_layer) == n_layer and any(nd in n for nd in no_decay))],
            'weight_decay': 0.0,
            'lr_decay': args.lr_layer_decay ** (max_n_layer - n_layer)
        })
        # #### debug info
        # ns = [n for n, _ in model.named_parameters() if (
        #     extract_n_layer(n, max_n_layer=max_n_layer) == n_layer and not any(nd in n for nd in no_decay))]
        # lr_decay = args.lr_layer_decay ** (max_n_layer-n_layer)
        # print(ns)
        # print(lr_decay)
        # print('\n\n')
    ## setting optimizer, plan to add RADAM & LookAhead
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if 0. < args.warmup_proportion < 1.0:
        warmup_steps = t_total * args.warmup_proportion
    else:
        warmup_steps = args.warmup_steps
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    """
        The following settings follow apex's instruction
        ``
        model.cuda() # Cuda-ing your model should occur before the call to amp.initialize
        model, optimizer = amp.initialize(model, optimizer)
        model = nn.DataParallel(model)
        ``
        https://github.com/NVIDIA/apex/issues/227
    """
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Fold  = %d", args.fold)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d, warmup steps = %d", t_total, warmup_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    epoch_num = 0
    min_loss = float('inf')

    for _ in train_iterator:


        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            ## model inputs
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            #(loss), logits, (hidden_states), (attentions)
            #first_token_tensor = hidden_states[:, 0]



            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        if args.k_fold == 0:
                            results = evaluate(args, model, tokenizer, prefix='test '+ str(global_step) + ' :epoch ' + str(epoch_num))
                        else:
                            results = evaluate(args, model, tokenizer,
                                               prefix='test ' + str(global_step) + ' :epoch ' + str(epoch_num) + ' fold ' + str(args.fold),split = str(args.fold) + '_test')

                        for key, value in results.items():
                            tb_writer.add_scalar('test_eval_{}'.format(key), value, global_step)
                        # train_results = evaluate(args, model, tokenizer, prefix='train: ' + str(global_step) + ' :epoch ' + str(epoch_num),split='train')
                        # for key, value in train_results.items():
                        #     tb_writer.add_scalar('train_eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                #if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                # if min_loss > loss.item():
                #     min_loss = loss.item()
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'best_checkpoint')
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model,
                #                                             'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     #torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        epoch_num += 1
        logger.info("epoch %s",epoch_num)
        if args.k_fold != 0:
            results = evaluate(args, model, tokenizer, prefix='epoch '+ str(epoch_num),split=str(args.fold) + '_test')
        else:
            results = evaluate(args, model, tokenizer, prefix='epoch ' + str(epoch_num)+ ' fold ' + str(args.fold), split= 'test')
        for key, value in results.items():
            tb_writer.add_scalar('test_eval_epoch_{}'.format(key), value, epoch_num)
        if args.k_fold != 0:
            train_results = evaluate(args, model, tokenizer, prefix='epoch '+ str(epoch_num)+ ' fold ' + str(args.fold), split=str(args.fold) + '_train')
        else:
            train_results = evaluate(args, model, tokenizer, prefix='epoch ' + str(epoch_num),split='train')
        for key, value in train_results.items():
            tb_writer.add_scalar('train_eval_epoch_{}'.format(key), value, epoch_num)
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    #del loss
    return global_step, tr_loss / global_step



def evaluate(args, model, tokenizer, prefix="",split="test"):
    eval_output_dir = args.output_dir
    eval_dataset, eval_examples = load_and_cache_examples(args, tokenizer, split)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix + '\t' + split))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    scores = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
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

    # get scores or results depending on the task
    qids = [eg.qid for eg in eval_examples]
    docids = [eg.docid for eg in eval_examples]
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
    trec_eval = TREC_evaluator(run_id=args.exp_name + '_' + split, base_path=args.output_dir)
    trec_eval.write_trec_result(eval_df)
    ndcgs = trec_eval.get_ndcgs()
    for metric in ndcgs:
        ltr_metric_scores[metric].append(ndcgs[metric])
    #
    ltr_metric_scores["eval_loss"] = eval_loss / nb_eval_steps
    # report resutls
    logger.info("***** Eval results {} *****".format(prefix + ' '+split))
    for key in sorted(ltr_metric_scores.keys()):
        logger.info("  %s = %s", key, str(ltr_metric_scores[key]))

    return ltr_metric_scores


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--fasttext_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--k_fold", default=5, type=int, required=True,
                        help="k fold cross validation.")
    parser.add_argument("--mode", default='regression', type=str, required=False,
                        help="regression or classification.")
    parser.add_argument("--schema", default='SEP', type=str, required=True,
                        help="how to concatenate the schema labels.")
    parser.add_argument("--content", default=None, type=str, required=False,
                        help="how to concatenate the table content:ORDER, ROW,COL,RAND_ROW,RAND_COL")
    parser.add_argument("--selector", default=None, type=str, required=False,
                        help="how to select the table content:SUM,AVG,MAX. Valid when not randomly selecting content")


    ## field options
    parser.add_argument("--use_caption",action='store_true',help="whether use table cation or not ")
    parser.add_argument("--use_sec_title",action='store_true',help="whether use table section title or not ")
    parser.add_argument("--use_pg_title",action='store_true',help="whether use table page title or not ")
    parser.add_argument("--use_schema",action='store_true',help="whether use table schemas or not ")
    parser.add_argument("--resplit", action='store_true')
    parser.add_argument("--no_save", action='store_true')
    parser.add_argument("--trunc", action='store_true', help="truncate all fields ")

    parser.add_argument("--ignore_logits_layer", action='store_true',
                        help="whether to skip initialization of logits layers.")
    parser.add_argument("--ignore_sequence_summary_layer", action='store_true',
                        help="whether to skip initialization of sequence summary layers.")
    ## single task
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task")
    parser.add_argument("--fold", default=0, type=int, required=False,
                        help="The fold of the dataset")

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
    args.rand_content = dict()
    if args.use_caption:
        exp_name = exp_name + '_caption'
    if args.use_pg_title:
        exp_name = exp_name + '_pgTitle'
    if args.use_sec_title:
        exp_name = exp_name + '_secTitle'

    if args.mode =='classification':
        exp_name = exp_name + '_c'
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

    #exp_name = exp_name + str(datetime.datetime.now())
    args.exp_name = exp_name
    args.original_output_dir = args.output_dir
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    #clear cache
    cached_features_file = os.path.join(args.data_dir, 'cached_*_{}_{}'.format(
        args.exp_name,
        str(args.max_seq_length)))
    if args.resplit:
        for each_cache in glob(cached_features_file):
            os.remove(each_cache)
    #preprare for k-fold data
    if args.k_fold == 0:
        args.fold = 0
    if args.k_fold != 0 and args.resplit:
        TableProcessor.create_k_fold(args.data_dir,args.k_fold)


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

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        filemode = 'w')
    # add file handler to log training info
    if not os.path.isdir(args.output_dir): os.mkdir(args.output_dir)
    log_file_name = str(args.k_fold) + '_fold_training.log'

    fh = logging.FileHandler(os.path.join(args.output_dir, str(args.fold) + '_fold_'+"training.log"))
    logger.addHandler(fh)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    processor = TableProcessor(args.data_dir)
    label_list = processor.get_labels()
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
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
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

    if args.do_train and args.k_fold == 0:
        # training without cross validation
        # load model
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            ignore_logits_layer=args.ignore_logits_layer,
                                            ignore_sequence_summary_layer=args.ignore_sequence_summary_layer,
                                            load_averaged_checkpoint=args.load_averaged_checkpoint)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        train_dataset, _ = load_and_cache_examples(args, tokenizer, split="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not args.no_save:
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            #torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    elif args.do_train  and args.k_fold  > 0 and args.fold == 0:
        metric_scores = defaultdict(list)
        for fold in range(1,6):
            args.fold = fold
            logger.info("beginning fold %s",args.fold)
            logger.info("memory_cached %s", torch.cuda.memory_cached(device=args.device)/1024/1024)
            logger.info("memory_allocated %s", torch.cuda.memory_allocated(device=args.device)/1024/1024)
            args.output_dir = os.path.join(args.original_output_dir, args.exp_name,str(args.fold)+'_fold')
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            # load model
            model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                                config=config,
                                                ignore_logits_layer=args.ignore_logits_layer,
                                                ignore_sequence_summary_layer=args.ignore_sequence_summary_layer,
                                                load_averaged_checkpoint=args.load_averaged_checkpoint)

            if args.local_rank == 0:
                torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

            model.to(args.device)

            train_dataset, _ = load_and_cache_examples(args, tokenizer, split=str(args.fold) + "_train")
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
            logger.info(" global_step = %s, average loss = %s, fold = %s", global_step, tr_loss,fold)

            # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("Saving model checkpoint to %s", args.output_dir)
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)

                # Good practice: save your training arguments together with the trained model
                #torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

            #evaluate
            results = evaluate(args, model, tokenizer, prefix='fold ' + str(fold), split=str(args.fold) + '_test')
            torch.save(results, os.path.join(args.output_dir, 'cache_rs_fold_{0}.bin'.format(str(fold))))
            del model
            del model_to_save
            del tr_loss
            torch.cuda.empty_cache()
            for metric in results:
                metric_scores[metric].append(results[metric])
            logger.info("ending fold %s",fold)
            logger.info("memory_cached %s", torch.cuda.memory_cached(device=args.device)/1024/1024)
            logger.info("memory_allocated %s", torch.cuda.memory_allocated(device=args.device)/1024/1024)

        for metric in metric_scores:
            print("{0}:{1}".format(metric, np.mean(metric_scores[metric])))
            logger.info(" Metric = %s, score = %s", metric, np.mean(metric_scores[metric]))


    elif args.do_train  and args.k_fold  > 0 and args.fold > 0:
        fold = args.fold
        logger.info("beginning fold %s",args.fold)
        logger.info("memory_cached %s", torch.cuda.memory_cached(device=args.device)/1024/1024)
        logger.info("memory_allocated %s", torch.cuda.memory_allocated(device=args.device)/1024/1024)
        args.output_dir = os.path.join(args.original_output_dir, args.exp_name,str(args.fold)+'_fold')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        # load model
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            ignore_logits_layer=args.ignore_logits_layer,
                                            ignore_sequence_summary_layer=args.ignore_sequence_summary_layer,
                                            load_averaged_checkpoint=args.load_averaged_checkpoint)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)
        #print(model)
        #exit()

        train_dataset, _ = load_and_cache_examples(args, tokenizer, split=str(args.fold) + "_train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s, fold = %s", global_step, tr_loss,fold)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            #torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        #evaluate
        results = evaluate(args, model, tokenizer, prefix='fold ' + str(fold), split=str(args.fold) + '_test')
        torch.save(results, os.path.join(args.output_dir, 'cache_rs_fold_{0}.bin'.format(str(fold))))



if __name__ == '__main__':
    main()
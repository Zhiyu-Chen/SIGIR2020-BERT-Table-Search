import subprocess
import os

class TREC_evaluator(object):
    def __init__(self,run_id,base_path = "/root/data/table",trec_cmd = "../trec_eval"):
        self.run_id = run_id
        self.base_path = base_path
        self.rank_path = os.path.join(self.base_path ,self.run_id + ".result")
        self.qrel_path = os.path.join(self.base_path ,self.run_id + ".qrels")
        #self.qrel_path = "/home/zhc415/Dropbox/repo/research/table_ir/www2018-table/data/qrels.txt"
        self.trec_cmd = trec_cmd


    def write_trec_result(self,eval_df):
        '''
        eval_df = pd.DataFrame(data={
        'id_left': test_df['id_left'],
        'id_right':  test_df['id_right'],
        'true': test_y.squeeze(),
        'pred': y_preds.squeeze()
    })
        :param eval_df:
        :return:
        '''
        # write result files
        f_rank = open(self.rank_path, 'w')
        f_rel = open(self.qrel_path, 'w')
        qids = [str(each) for each in sorted(list(set([int(each) for each in eval_df['id_left'].unique()])))]
        for qid in qids:
            qid_docs = eval_df[eval_df['id_left'] == qid]
            qid_docs = qid_docs.sort_values(by=['pred'], ascending=False)
            rank = 1
            for each in qid_docs.values:
                f_rank.write(each[0] + "\tQ0\t" + each[1] + "\t" + str(rank) + "\t" + str(each[3]) + "\t" + self.run_id + "\n")
                f_rel.write(each[0] + "\t0\t" + each[1] + "\t" + str(each[2]) + '\n')
                rank += 1
        f_rank.close()
        f_rel.close()

    def get_ndcgs(self,metric='ndcg_cut',qrel_path=None,rank_path=None,all_queries = False ):
        if qrel_path is None:
            qrel_path = self.qrel_path
        if rank_path is None:
            rank_path = self.rank_path
        metrics = ['ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_15', 'ndcg_cut_20',
                   #'map_cut_5', 'map_cut_10', 'map_cut_15', 'map_cut_20',
                   'map', 'recip_rank']  # 'relstring'
        if all_queries:
            results = subprocess.run([self.trec_cmd, '-c', '-m', metric,'-q', qrel_path, rank_path],
                                     stdout=subprocess.PIPE).stdout.decode('utf-8')
            q_metric_dict = dict()
            for line in results.strip().split("\n"):
                seps = line.split('\t')
                metric = seps[0].strip()
                qid = seps[1].strip()
                if metric not in metrics:
                    continue
                if metric != 'relstring':
                    score = float(seps[2].strip())
                else:
                    score = seps[2].strip()
                if qid not in q_metric_dict:
                    q_metric_dict[qid] = dict()
                q_metric_dict[qid][metric] = score
            return q_metric_dict
        else:
            results = subprocess.run([self.trec_cmd,'-c','-m',metric, qrel_path,rank_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
            ndcg_scores = dict()
            for line in results.strip().split("\n"):
                seps = line.split('\t')
                metric = seps[0].strip()
                qid = seps[1].strip()
                if metric not in metrics or qid != 'all':
                    continue
                ndcg_scores[seps[0].strip()] = float(seps[2])
            return ndcg_scores





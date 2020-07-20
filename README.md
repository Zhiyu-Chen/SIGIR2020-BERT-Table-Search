# SIGIR2020-BERT-Table-Search

## Data

1. WikiTables dataset is from https://github.com/iai-group/www2018-table. The json version is under './data' and named as "all.json". We also created the splitted version which used for 5-fold cross validation in our paper. For example, '1_train.jsonl' is the 1st fold for training.
2. To run all the experiments, please download the weights of bert-large-cased model and fasttext embedding. 


## Running Examples

1. run_table.py: without additional features. Only BERT is used. "table_content.sh" is an example to run it.
2. run_hybrid_table.py: jointly training BERT with additional features. "run_hybrid_table.sh" is an example to run it.
3. run_combine.py: feature-based approach of BERT. "run_combine.sh" is an example to run it. Since this method uses the fine-tuned BERT model, the corresponding method should be run by "run_table.py" 1st. The script will read the saved BERT weights by "run_table.py" from the output directory of corresponding fold.

There are a lot of options in the arguments. Among all of them, "--content" selects the item type which could be ROW/COL/CELL. " --selector" choses the selector strategy which could be MAX/SUM/AVG. In our experiments, we find out that the combination of ROW and MAX performs the best.



# BOW_NLI
## 3_1
### model_py_file
contains all models trained on SNLI, model name follows '{vocab size}_{embed dimension}_{interaction}_{model type}.py'
### results_pickle_file
contains validation loss and accuracy computed by all models in model_py_file
## 3_2
contains jupyter notebooks that use the best logistic and nn models tuned in 3_1 to evaluate on MNLI
contains validation loss and accuracy computed by the best two models on MNLI
contains saved best logistic and nn model from 3_1
## 3_3
contains the best finetuned model further trained and evaluated on each genre in MNLI

## 3_4
used pre-trained word embedding from FastText to train and evaluate the SNLI and MNLI
FastText data set can be found at https://fasttext.cc/
Please download 'wiki-news-300d-1M.vec.zip' to folder 3_4

## 3_5
find similar words in our self-trained word embedding in 2-norm 
import argparse
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from transformers.data.processors.utils import InputExample
from transformers.data.processors.glue import glue_convert_examples_to_features
import logging
import torch
import random
import numpy as np
import pydevd_pycharm
import pandas as pd
import os
from dataset import DataSet
import _pickle as pickle
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score


label_list = ['unrelated', 'discuss', 'agree', 'disagree']
label_map = dict(zip(range(4), label_list))
related_labels = set(label_list) - {'unrelated'}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging(log_file=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
def official_metric(test_labels, gold_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g_stance, t_stance) in enumerate(zip(gold_labels, test_labels)):
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in related_labels and t_stance in related_labels:
            score += 0.25

        cm[label_list.index(g_stance)][label_list.index(t_stance)] += 1

    return score, cm


def evaluate(model, dataset, batch_size, device):
    data_sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, batch_size, sampler=data_sampler)

    model.to(device)
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    for step, batch in enumerate(tqdm(data_loader)):
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        loss, logits = model(**inputs)[:2]

        running_loss += loss.item()
        predictions.extend(logits.max(dim=1).indices.tolist())
        true_labels.extend(inputs['labels'].tolist())

    running_loss /= len(data_loader)
    predictions = [label_map[i] for i in predictions]
    true_labels = [label_map[i] for i in true_labels]
    rel_score, conf_matr = official_metric(predictions, true_labels)
    best_score, _ = official_metric(true_labels, true_labels)
    score = rel_score*100/best_score
    report = classification_report(true_labels, predictions, labels=label_list)

    return score, report, running_loss, conf_matr


def train(model, train_dataset, val_dataset, batch_size, num_epochs, tb_log_path, device, learning_rate,
          early_stopping_threshold, out_path):
    """Defines the training loop for the model."""
    tb_logger = SummaryWriter(tb_log_path)
    logger = logging.getLogger(__name__)

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    rel_score, report, val_loss, conf_matr = evaluate(model, val_dataset, batch_size, device)
    logger.info('Epoch -1 Validation Stats')
    logger.info(f'Rel Score {rel_score}, Val Loss {val_loss}')

    best_score = 0.0
    score_list = []
    model.to(device)
    for epoch in range(num_epochs):
        logger.info(f"Training Epoch: {epoch}")
        running_loss = 0.0
        model.zero_grad()
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            loss, logits = model(**inputs)[:2]

            loss.backward()
            optimizer.step()
            model.zero_grad()

            running_loss += loss.item()

        running_loss /= len(train_loader)
        rel_acc, report, val_loss, conf_matr = evaluate(model, val_dataset, batch_size, device)

        tb_logger.add_scalars('Running Losses', {'train_loss': running_loss,
                                                 'val_loss': val_loss}, epoch)
        tb_logger.add_scalar('Validation Scores', rel_acc, epoch)

        logger.info(f"Epoch: {epoch}")
        logger.info(f"Training Loss: {running_loss}")
        logger.info(f"Validation Loss: {val_loss}")
        logger.info(f"Relative Accurary {rel_acc}")
        logger.info(f"Report: {report}")

        if rel_acc > best_score:
            model.save_pretrained(out_path)
            logger.info(f"Model saved at Epoch {epoch}.")
            best_score = rel_acc
            score_list = []
        else:
            score_list.append(rel_acc)

        if len(score_list) >= early_stopping_threshold:
            logger.info(f"Early Stopping after {epoch} iteration")
            logger.info(f"the highest F-score achieved on the validation data: {best_score}")
            logger.info(f"The model is saved to {out_path}")
            logger.info("--------------------")
            break


def process_data(datapath, tokenizer, test_portion=False, split_ratio=0.8, load_processed=False):
    """Reads datafiles, computes features, splits into train/val, generate tensor datasets."""

    def generate_inputexamples(dataframe):
        examples = []
        for irow, row in dataframe.iterrows():
            examples.append(InputExample(irow, row.Headline, row.articleBody, row.Stance))
        return examples

    def balanced_split(features, split_ratio):
        """Split train/val such that each class proportionately represented in splits"""

        logger.info(f"Splitting data into train/val split, maintaining class compositions.")
        train_feats = []
        val_feats = []
        for cur_label in range(4):
            featlist = [feature for feature in features if feature.label == cur_label]
            random.shuffle(featlist)
            train, val = featlist[:int(split_ratio * len(featlist))], featlist[int(split_ratio * len(featlist)):]
            train_feats.extend(train)
            val_feats.extend(val)
        return train_feats, val_feats

    def gen_dataset(features):
        """Generate tensor dataset for a split"""

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        tensor_dataset = TensorDataset(input_ids, all_attention_mask, all_labels)
        logger.info(f"Processed Tensor data from feature objects")
        return tensor_dataset

    logger = logging.getLogger(__name__)
    mode = 'competition_test' if test_portion else 'train'
    logger.info(f"Processing dataset from {datapath} in mode {mode}.")
    dumppath = os.path.join(datapath, f'processed_{mode}_data.pkl')

    if load_processed and os.path.isfile(dumppath):
        logger.info(f"Loading the preprocessed data from {dumppath}")
        with open(dumppath, 'rb') as f:
            features = pickle.load(f)
    else:
        df_labels = pd.read_csv(os.path.join(datapath, f'{mode}_stances.csv'))
        df_body = pd.read_csv(os.path.join(datapath, f'{mode}_bodies.csv'))
        df_dataset = pd.merge(df_labels, df_body, left_on='Body ID', right_on='Body ID')
        df_dataset['Headline'] = df_dataset['Headline'].apply(DataSet.clean_article)
        df_dataset['articleBody'] = df_dataset['articleBody'].apply(DataSet.clean_article)

        logger.info(f"Preparing data from dataset files")
        examples = generate_inputexamples(df_dataset)
        features = glue_convert_examples_to_features(examples, tokenizer,
                                                     label_list=label_list,
                                                     output_mode='classification')
        with open(dumppath, 'wb') as f:
            pickle.dump(features, f)
        logger.info(f"Dumped processed data pickle at {dumppath}")

    if mode == 'train':
        train_feats, val_feats = balanced_split(features, split_ratio)
        train_dataset = gen_dataset(train_feats)
        val_dataset = gen_dataset(val_feats)

        return train_dataset, val_dataset
    else:
        return gen_dataset(features)


def main():
    parser = argparse.ArgumentParser(prog='Roberta factchecker', description='An experiment with Roberta for '
                                                                             'factchecking.')

    parser.add_argument('-remote_debug', default=False, action='store_true', help='True for pycharm remote debug')
    parser.add_argument('--mode', default='test', type=str, choices=['train', 'test', 'infer'])
    parser.add_argument('-seed', default=123, type=int, help="Fix a random seed here.")
    parser.add_argument('--load_processed', default=True, type=bool,
                        help="True if would like to load processed data pickle")
    parser.add_argument('--split_ratio', default=0.85, type=float, help="Split portion for train/val split.")

    # Path args
    parser.add_argument('--data_path', default='data/fnc-1', type=str, help='Path to '
                                                                            'the data '
                                                                            'directory.')
    parser.add_argument('--save_path', default='models', type=str, help="Path to save models to.")
    parser.add_argument('--pretrained_model_path', default='models/roberta', type=str, help="Path to pretrained model "
                                                                                            "config and checkpoint.")

    # Logging args
    parser.add_argument('--log_dir', default='logs', type=str, help='path to log file.')
    parser.add_argument('--tb_log_dir', default='tb_runs', type=str, help="Path to tensorboard logs.")
    parser.add_argument('--experiment_label', default='stance_detection', type=str, help='Label for logs.')

    # Training args
    parser.add_argument('--cuda', default=True, type=bool, help="False if cpu mode needed.")
    parser.add_argument('--batch_size', default=6, type=int, help="Training batch size.")
    parser.add_argument('--num_epochs', default=10, type=int, help="Number of training epochs")
    parser.add_argument('--learning_rate', default=2e-5, type=float, help="Optimizer(Adam) Learning Rate.")
    parser.add_argument('--early_stopping_threshold', default=4, type=int, help="Epoch Limit for Early Stopping")

    args = parser.parse_args()

    if args.remote_debug:
        pydevd_pycharm.settrace('10.1.65.133', port=4328, stdoutToServer=True, stderrToServer=True)

    # Logging Setup
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    setup_logging(os.path.join(args.log_dir, args.experiment_label)+'.log')
    logger = logging.getLogger(__name__)
    tb_path = os.path.join(args.tb_log_dir, args.experiment_label)

    argstring = 'Arguments parsed:\n' + '\n'.join([arg+' '+str(getattr(args, arg)) for arg in vars(args)])
    logger.info(argstring)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='models/roberta/')
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        train_dataset, val_dataset = process_data(args.data_path, tokenizer, test_portion=(args.mode == 'test'),
                                                  split_ratio=args.split_ratio,
                                                  load_processed=args.load_processed)

        config = RobertaConfig.from_pretrained(args.pretrained_model_path, num_labels=4)
        config.hidden_dropout_prob = 0.32
        model = RobertaForSequenceClassification.from_pretrained(args.pretrained_model_path, config=config)

        out_path = os.path.join(args.save_path, args.experiment_label)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        train(model, train_dataset, val_dataset,
              batch_size=args.batch_size,
              num_epochs=10,
              tb_log_path=tb_path,
              device=device,
              learning_rate=args.learning_rate,
              early_stopping_threshold=args.early_stopping_threshold,
              out_path=out_path)

    elif args.mode == 'test':
        logger.info(f"Testing model saved at {args.save_path}")

        test_dataset = process_data(args.data_path, tokenizer, test_portion=(args.mode == 'test'),
                                    load_processed=args.load_processed)

        model_path = os.path.join(args.save_path, args.experiment_label)
        config = RobertaConfig.from_pretrained(model_path, num_labels=4)
        model = RobertaForSequenceClassification.from_pretrained(model_path, config=config)

        rel_score, report, running_loss, conf_matr = evaluate(model, test_dataset, args.batch_size, device)

        logger.info(f"Evaluation results")
        logger.info(f"Relative Accuracy: {rel_score}")
        logger.info(f"Running Loss: {running_loss}")
        logger.info(f"Report: {report}")

    elif args.mode == 'infer':
        logging.info(f"Loading model saved at {args.save_path}")

        model_path = os.path.join(args.save_path, args.experiment_label)
        config = RobertaConfig.from_pretrained(model_path, num_labels=4)
        model = RobertaForSequenceClassification.from_pretrained(model_path, config=config)

        headline = 'Man arrested in connection to Brooklyn Murders'
        article = 'A 28 year old male was arrested by the NYPD relating to the murder on Saturday. Two children were found dead in a truck in Brooklyn, and police had been looking for leads regarding the case.'

        headline1 = 'Trump orders investigation into the voters scandal.'
        article1 = 'The president has ordered to close the investigation into the scandal.'
        batch_inp = [(headline, article), (headline1, article1), (headline, article1), (headline1, article)]
        bout = tokenizer.batch_encode_plus(batch_inp, pad_to_max_length=True, max_len=512, return_tensors='pt')
        mout = model(**bout)
        predictions = [label_map[key] for key in mout[0].argmax(dim=1).tolist()]
        pass


if __name__ == "__main__":
    main()

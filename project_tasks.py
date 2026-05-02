# coding: UTF-8
import argparse
import random
import time
from importlib import import_module
from pathlib import Path

import numpy as np


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def safe_label(config, label_id):
    if 0 <= label_id < len(config.class_list):
        return config.class_list[label_id]
    return str(label_id)


def clean_field(value):
    return str(value).replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')


def write_misclassified(config, model, data_iter, output_path):
    import torch

    model.eval()
    records = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for start in range(0, len(data_iter.batches), data_iter.batch_size):
            batch = data_iter.batches[start:start + data_iter.batch_size]
            texts, labels = data_iter._to_tensor(batch)
            outputs = model(texts)
            preds = torch.max(outputs.data, 1)[1].cpu().numpy()
            golds = labels.data.cpu().numpy()

            for offset, (gold, pred) in enumerate(zip(golds, preds)):
                if int(gold) == int(pred):
                    continue
                sample = batch[offset]
                original = sample[3] if len(sample) > 3 else ''
                processed = sample[4] if len(sample) > 4 else original
                record = {
                    'index': start + offset,
                    'true_id': int(gold),
                    'pred_id': int(pred),
                    'true_label': safe_label(config, int(gold)),
                    'pred_label': safe_label(config, int(pred)),
                    'text': original,
                    'processed_text': processed,
                }
                records.append(record)

    with output_path.open('w', encoding='UTF-8') as f:
        f.write(
            'index\ttrue_id\tpred_id\ttrue_label\tpred_label\t'
            'text\tprocessed_text\n'
        )
        for record in records:
            f.write(
                '{index}\t{true_id}\t{pred_id}\t{true_label}\t'
                '{pred_label}\t{text}\t{processed_text}\n'.format(
                    index=record['index'],
                    true_id=record['true_id'],
                    pred_id=record['pred_id'],
                    true_label=clean_field(record['true_label']),
                    pred_label=clean_field(record['pred_label']),
                    text=clean_field(record['text']),
                    processed_text=clean_field(record['processed_text']),
                )
            )
    return records


def make_explanation(record, direction):
    if direction == 'a_wrong_b_right':
        return (
            'Task B removes punctuation/symbol noise, so the model can focus '
            'more on category words in the sentence.'
        )
    return (
        'Task A keeps punctuation, symbols, or numeric cues that may help this '
        'case; Task B may remove part of that useful signal.'
    )


def write_comparison(task_a_records, task_b_records, output_path):
    a_wrong = {record['index']: record for record in task_a_records}
    b_wrong = {record['index']: record for record in task_b_records}
    a_wrong_b_right = [
        record for index, record in a_wrong.items()
        if index not in b_wrong
    ][:3]
    b_wrong_a_right = [
        record for index, record in b_wrong.items()
        if index not in a_wrong
    ][:3]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='UTF-8') as f:
        f.write('Task A misclassified but Task B correctly classified\n')
        for i, record in enumerate(a_wrong_b_right, 1):
            f.write(
                '\n{}. index: {}\ntrue: {} ({})\nTask A pred: {} ({})\n'
                'text: {}\nprocessed_text: {}\nexplanation: {}\n'.format(
                    i,
                    record['index'],
                    clean_field(record['true_label']),
                    record['true_id'],
                    clean_field(record['pred_label']),
                    record['pred_id'],
                    clean_field(record['text']),
                    clean_field(record['processed_text']),
                    make_explanation(record, 'a_wrong_b_right'),
                )
            )

        f.write('\n\nTask B misclassified but Task A correctly classified\n')
        for i, record in enumerate(b_wrong_a_right, 1):
            f.write(
                '\n{}. index: {}\ntrue: {} ({})\nTask B pred: {} ({})\n'
                'text: {}\nprocessed_text: {}\nexplanation: {}\n'.format(
                    i,
                    record['index'],
                    clean_field(record['true_label']),
                    record['true_id'],
                    clean_field(record['pred_label']),
                    record['pred_id'],
                    clean_field(record['text']),
                    clean_field(record['processed_text']),
                    make_explanation(record, 'b_wrong_a_right'),
                )
            )

    return a_wrong_b_right, b_wrong_a_right


def run_experiment(task_name, preprocess, args):
    from train import train
    from utils import build_dataset, build_iterator, get_time_dif

    print('\n========== {} =========='.format(task_name))
    set_seed(args.seed)
    x = import_module('models.Transformer')
    config = x.Config(args.dataset, args.embedding)
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    config.classifier_head = args.classifier_head

    start_time = time.time()
    print('Loading data...')
    vocab, train_data, dev_data, test_data = build_dataset(
        config, False, preprocess=preprocess
    )
    config.n_vocab = len(vocab)
    train_iter = build_iterator(train_data, config, False)
    dev_iter = build_iterator(dev_data, config, False)
    test_iter = build_iterator(test_data, config, False)
    print('Time usage:', get_time_dif(start_time))

    model = x.Model(config).to(config.device)
    test_acc, test_loss, _, _ = train(
        config, model, train_iter, dev_iter, test_iter
    )
    loss_value = test_loss.item() if hasattr(test_loss, 'item') else test_loss

    output_path = args.output_dir / '{}.misclassified'.format(task_name)
    misclassified = write_misclassified(config, model, test_iter, output_path)
    print(
        '{} test loss: {:.6f}, test accuracy: {:.4%}'.format(
            task_name, float(loss_value), test_acc
        )
    )
    print('{} misclassified: {}'.format(task_name, len(misclassified)))
    print('Wrote {}'.format(output_path))
    return {
        'name': task_name,
        'accuracy': test_acc,
        'loss': float(loss_value),
        'misclassified': misclassified,
        'classifier_head': config.classifier_head,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run Project 3 Task A/Task B experiments.'
    )
    parser.add_argument('--dataset', default='THUCNews')
    parser.add_argument('--embedding', default='random')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument(
        '--classifier-head',
        choices=['flatten', 'cls', 'mean'],
        default='flatten',
        help='flatten keeps the original head; cls enables the optional '
             'Dropout + Linear first-token pooling head; mean enables '
             'mean pooling over all token embeddings.',
    )
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    if args.classifier_head != 'flatten' and args.output_dir == Path('outputs'):
        args.output_dir = Path('outputs_{}'.format(args.classifier_head))

    task_a = run_experiment('taskA', preprocess=False, args=args)
    task_b = run_experiment('taskB', preprocess=True, args=args)

    comparison_path = args.output_dir / 'taskA_taskB_comparison.txt'
    a_examples, b_examples = write_comparison(
        task_a['misclassified'], task_b['misclassified'], comparison_path
    )

    summary_path = args.output_dir / 'summary.txt'
    with summary_path.open('w', encoding='UTF-8') as f:
        f.write('Classifier head: {}\n'.format(args.classifier_head))
        f.write('Task A test loss: {:.6f}\n'.format(task_a['loss']))
        f.write('Task A test accuracy: {:.4%}\n'.format(task_a['accuracy']))
        f.write('Task A misclassified: {}\n'.format(
            len(task_a['misclassified'])
        ))
        f.write('Task B test loss: {:.6f}\n'.format(task_b['loss']))
        f.write('Task B test accuracy: {:.4%}\n'.format(task_b['accuracy']))
        f.write('Task B misclassified: {}\n'.format(
            len(task_b['misclassified'])
        ))
        f.write('A wrong, B right examples: {}\n'.format(len(a_examples)))
        f.write('B wrong, A right examples: {}\n'.format(len(b_examples)))

    print('\n========== Summary ==========')
    print('Task A accuracy: {:.4%}'.format(task_a['accuracy']))
    print('Task B accuracy: {:.4%}'.format(task_b['accuracy']))
    print('Wrote {}'.format(summary_path))
    print('Wrote {}'.format(comparison_path))


if __name__ == '__main__':
    main()

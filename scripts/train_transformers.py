"""Fine-tune MuRIL or XLM-R, optionally with the logit-normalised objective.

Every hyperparameter other than the loss is held fixed, so that the two arms
differ only in the quantity under study. Validation loss is plain cross-entropy
in both arms: if the arms stopped on different criteria the comparison would not
isolate the loss.

Use --fixed_epochs to disable early stopping and train every arm for the same
number of epochs. Without it a cross-entropy arm that stops at three epochs is
compared against an arm that trains for six, and the difference in training
length is attributed to the objective.

    python scripts/train_transformers.py --model muril --arm baseline --fixed_epochs 6
    python scripts/train_transformers.py --model muril --arm logitnorm --kappa 0.04 \
        --fixed_epochs 6 --seed 1
"""
import argparse
import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlfnd.data import assert_aligned, build_split
from mlfnd.losses import logit_norm_loss
from mlfnd.models import BATCH_SIZE, HF_NAMES, LR, MAX_EPOCHS, MAX_LEN, PATIENCE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HeadlineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts, self.labels, self.tokenizer = list(texts), list(labels), tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(str(self.texts[idx]), truncation=True,
                                 padding='max_length', max_length=MAX_LEN,
                                 return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


@torch.no_grad()
def predict(model, loader, device):
    """Ordinary softmax of the raw logits, for both arms."""
    model.eval()
    out = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        out.append(F.softmax(model(**inputs).logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=list(HF_NAMES))
    parser.add_argument('--arm', default='baseline', choices=['baseline', 'logitnorm'])
    parser.add_argument('--kappa', type=float, default=0.04,
                        help='logit normalisation scale; ignored for the baseline arm')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--bundle', default='probs/probs_bundle.npz')
    parser.add_argument('--out_dir', default='probs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fixed_epochs', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    tag = args.model + '_' + args.arm
    if args.arm == 'logitnorm':
        tag += f'_kappa{args.kappa}'
    if args.fixed_epochs:
        tag += f'_ep{args.fixed_epochs}'
    if args.seed != 42:
        tag += f'_seed{args.seed}'
    out_path = os.path.join(args.out_dir, f'probs_{tag}.npz')
    ckpt_path = os.path.join(args.out_dir, f'checkpoint_{tag}.pt')
    if os.path.exists(out_path) and not args.overwrite:
        print(f"{out_path} exists; pass --overwrite to redo")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu')
    train, val, test = build_split(args.data_dir)
    print(f"train {len(train):,}  val {len(val):,}  test {len(test):,}")
    assert_aligned(args.bundle, val, test)

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    name = HF_NAMES[args.model]
    tokenizer = AutoTokenizer.from_pretrained(name)
    # use_safetensors avoids the .bin checkpoint, which recent transformers
    # releases refuse to load under torch below 2.6
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=2, use_safetensors=True).to(device)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(HeadlineDataset(train['headline'], train['label_bin'], tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    val_loader = DataLoader(HeadlineDataset(val['headline'], val['label_bin'], tokenizer),
                            batch_size=BATCH_SIZE * 2)
    test_loader = DataLoader(HeadlineDataset(test['headline'], test['label_bin'], tokenizer),
                             batch_size=BATCH_SIZE * 2)

    n_epochs = args.fixed_epochs or MAX_EPOCHS
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * n_epochs)

    start_epoch, best_val, best_state, no_improve = 0, float('inf'), None, 0
    if args.resume and os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            start_epoch = state['epoch'] + 1
            best_val, best_state, no_improve = (state['best_val'], state['best_state'],
                                                state['no_improve'])
            print(f"resumed at epoch {start_epoch + 1}")
        except Exception as error:
            print(f"checkpoint unreadable ({error.__class__.__name__}); starting fresh")

    epochs_run = start_epoch
    for epoch in range(start_epoch, n_epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            inputs = {k: v for k, v in batch.items() if k != 'labels'}
            logits = model(**inputs).logits
            loss = (F.cross_entropy(logits, batch['labels']) if args.arm == 'baseline'
                    else logit_norm_loss(logits, batch['labels'], args.kappa))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()

        model.eval()
        val_loss, preds, trues = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**{k: v for k, v in batch.items() if k != 'labels'}).logits
                val_loss += F.cross_entropy(logits, batch['labels']).item()
                preds.extend(logits.argmax(-1).cpu().numpy())
                trues.extend(batch['labels'].cpu().numpy())
        val_loss /= len(val_loader)
        epochs_run = epoch + 1
        print(f"epoch {epochs_run}: train {running / len(train_loader):.4f}  "
              f"val_ce {val_loss:.4f}  val_f1 {f1_score(trues, preds, average='macro') * 100:.2f}")

        if val_loss < best_val:
            best_val, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1

        # write to a temporary file and rename: an interrupted write can then
        # only lose the new checkpoint, never corrupt the previous one
        tmp_path = ckpt_path + '.tmp'
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'best_val': best_val,
                    'best_state': best_state, 'no_improve': no_improve}, tmp_path)
        os.replace(tmp_path, ckpt_path)

        if not args.fixed_epochs and no_improve >= PATIENCE:
            print("early stopping")
            break

    # with a fixed epoch budget the final weights are used, so that every arm
    # sees exactly the same amount of training
    if not args.fixed_epochs:
        model.load_state_dict(best_state)

    probs_val, probs_test = predict(model, val_loader, device), predict(model, test_loader, device)
    val_acc = (probs_val.argmax(1) == val['label_bin'].values).mean()
    test_acc = (probs_test.argmax(1) == test['label_bin'].values).mean()
    print(f"val {val_acc * 100:.2f}   test {test_acc * 100:.2f}")

    np.savez(out_path, probs_val=probs_val, probs_test=probs_test,
             y_val=val['label_bin'].values, lang_val=val['language'].values,
             y_test=test['label_bin'].values, lang_test=test['language'].values,
             val_acc=np.array(val_acc), test_acc=np.array(test_acc),
             epochs_run=np.array(epochs_run), seed=np.array(args.seed),
             kappa=np.array(args.kappa if args.arm == 'logitnorm' else 0.0))
    print(f"saved {out_path}")


if __name__ == '__main__':
    main()

"""Train the replication corpus tiers under the primary-corpus protocol.

Every hyperparameter matches the primary corpus, so that any difference in the
results is attributable to the corpus rather than the setup. The one necessary
change is stratifying on label alone, the corpus being monolingual.

    python scripts/train_hookbait.py --model gate
    python scripts/train_hookbait.py --model muril
    python scripts/train_hookbait.py --model xlmr
    python scripts/train_hookbait.py --bundle
"""
import argparse
import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlfnd.data import SEED, build_hookbait_split
from mlfnd.models import (BATCH_SIZE, HF_NAMES, LOGREG_KWARGS, LR, MAX_EPOCHS,
                          MAX_LEN, PATIENCE, TFIDF_KWARGS)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_probs(path, probs_val, probs_test, val, test, extra=None):
    payload = dict(
        probs_val=probs_val, probs_test=probs_test,
        y_val=val['label_bin'].values, lang_val=val['language'].values,
        y_test=test['label_bin'].values, lang_test=test['language'].values,
        val_acc=np.array((probs_val.argmax(1) == val['label_bin'].values).mean()),
        test_acc=np.array((probs_test.argmax(1) == test['label_bin'].values).mean()))
    if extra:
        payload.update(extra)
    np.savez(path, **payload)
    print(f"  saved {path}   val {float(payload['val_acc']) * 100:.2f}   "
          f"test {float(payload['test_acc']) * 100:.2f}")


def run_gate(train, val, test, args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    print("=== tier-1 gate ===")
    vectoriser = TfidfVectorizer(**TFIDF_KWARGS)
    features = vectoriser.fit_transform(train['headline'])
    classifier = LogisticRegression(**LOGREG_KWARGS).fit(features, train['label_bin'])
    print(f"  features {features.shape[1]:,}")
    save_probs(os.path.join(args.out_dir, 'probs_hb_tfidf.npz'),
               classifier.predict_proba(vectoriser.transform(val['headline'])).astype(np.float32),
               classifier.predict_proba(vectoriser.transform(test['headline'])).astype(np.float32),
               val, test)


def run_transformer(key, train, val, test, args, device):
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    name = HF_NAMES[key]
    print(f"=== {name} ===")
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=2, use_safetensors=True).to(device)

    class HeadlineDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts, self.labels = list(texts), list(labels)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoded = tokenizer(str(self.texts[idx]), truncation=True,
                                padding='max_length', max_length=MAX_LEN,
                                return_tensors='pt')
            item = {k: v.squeeze(0) for k, v in encoded.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(HeadlineDataset(train['headline'], train['label_bin']),
                              batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    val_loader = DataLoader(HeadlineDataset(val['headline'], val['label_bin']),
                            batch_size=BATCH_SIZE * 2)
    test_loader = DataLoader(HeadlineDataset(test['headline'], test['label_bin']),
                             batch_size=BATCH_SIZE * 2)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 0, len(train_loader) * MAX_EPOCHS)
    ckpt_path = os.path.join(args.out_dir, f'checkpoint_hb_{key}.pt')
    start, best_val, best_state, no_improve = 0, float('inf'), None, 0
    if args.resume and os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler']); start = state['epoch'] + 1
            best_val, best_state, no_improve = (state['best_val'], state['best_state'],
                                                state['no_improve'])
            print(f"  resumed at epoch {start + 1}")
        except Exception as error:
            print(f"  checkpoint unreadable ({error.__class__.__name__}); starting fresh")

    for epoch in range(start, MAX_EPOCHS):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); running += loss.item()
        model.eval()
        val_loss, correct, seen = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                val_loss += out.loss.item()
                correct += (out.logits.argmax(-1) == batch['labels']).sum().item()
                seen += len(batch['labels'])
        val_loss /= len(val_loader)
        print(f"  epoch {epoch + 1}: train {running / len(train_loader):.4f}  "
              f"val {val_loss:.4f}  val_acc {correct / seen * 100:.2f}")
        if val_loss < best_val:
            best_val, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        tmp_path = ckpt_path + '.tmp'
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'best_val': best_val, 'best_state': best_state,
                    'no_improve': no_improve}, tmp_path)
        os.replace(tmp_path, ckpt_path)
        if no_improve >= PATIENCE:
            print("  early stopping")
            break

    model.load_state_dict(best_state)

    @torch.no_grad()
    def predict(loader):
        model.eval()
        out = []
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            out.append(F.softmax(model(**inputs).logits, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)

    save_probs(os.path.join(args.out_dir, f'probs_hb_{key}.npz'),
               predict(val_loader), predict(test_loader), val, test)


def make_bundle(args, val, test):
    muril = np.load(os.path.join(args.out_dir, 'probs_hb_muril.npz'), allow_pickle=True)
    xlmr = np.load(os.path.join(args.out_dir, 'probs_hb_xlmr.npz'), allow_pickle=True)
    path = os.path.join(args.out_dir, 'probs_hb_bundle.npz')
    np.savez(path, pa_val=muril['probs_val'], pb_val=xlmr['probs_val'],
             pa_test=muril['probs_test'], pb_test=xlmr['probs_test'],
             y_val=val['label_bin'].values, lang_val=val['language'].values,
             y_test=test['label_bin'].values, lang_test=test['language'].values)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['gate', 'muril', 'xlmr'])
    parser.add_argument('--bundle', action='store_true')
    parser.add_argument('--csv', default='data_hb/hookbait.csv')
    parser.add_argument('--out_dir', default='probs_hb')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.model and not args.bundle:
        raise SystemExit("pass --model gate|muril|xlmr, or --bundle")

    os.makedirs(args.out_dir, exist_ok=True)
    train, val, test = build_hookbait_split(args.csv)
    print(f"train {len(train):,}  val {len(val):,}  test {len(test):,}   "
          f"fake {train['label_bin'].mean() * 100:.2f}%")

    if args.bundle:
        make_bundle(args, val, test)
    elif args.model == 'gate':
        run_gate(train, val, test, args)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device:", torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu')
        run_transformer(args.model, train, val, test, args, device)


if __name__ == '__main__':
    main()

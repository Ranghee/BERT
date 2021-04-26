import argparse, random, os
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm, trange
from vocab import load_vocab
import config as cfg
import model as bert
import data
import optimization

def destroy_process_group():
    dist.destroy_process_group()

def train_epoch(config, rank, epoch, model, criterion_lm, criterion_cls, optimizer, scheduler, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train({rank}) {epoch}") as pbar:
        for i, value in enumerate(train_loader):
            labels_cls, labels_lm, inputs, segments = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()
            outputs = model(inputs, segments)
            logits_cls, logits_lm = outputs[0], outputs[1]

            loss_cls = criterion_cls(logits_cls, labels_cls)
            loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
            loss = loss_cls + loss_lm

            loss_val = loss_lm.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)

def init_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) ## Reproducible PyTorch를 위한 randomness 올바르게 제어 https://hoya012.github.io/blog/reproducible_pytorch/

def train_model(rank, world_size, args):
    if 1 < args.n_gpu:
        init_process_group(rank, world_size)

    master = (world_size == 0 or rank % world_size == 0)
    vocab = load_vocab(args.vocab)
    config = cfg.Config.load(args.config)
    config.vocab_size
    config.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(config)

    best_epoch, best_loss = 0, 0

    model = bert.BERTPretraining(config)

    if os.path.isfile(args.save):
        best_epoch, best_loss = model.bert.load(args.save)
        print(f"rank: {rank} load pretrain from: {args.save}, epoch={best_epoch}, loss={best_loss}")
        best_epoch += 1
    if 1 < args.n_gpu:
        model.to(config.device)
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model.to(config.device)

    criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    criterion_cls = torch.nn.CrossEntropyLoss()

    train_loader = data.build_pretrain_loader(vocab, args, epoch=best_epoch, shuffle=True)

    t_total = len(train_loader) * args.epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_rate},
        {'params': [p for n, p in model.named_parameters() if any (nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optimization.Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    offset = best_epoch
    losses = []
    for step in trange(args.epoch, desc="Epoch"):
        epoch = step + offset
        if 0 < step:
            del train_loader
            train_loader = data.build_pretrain_loader(vocab, args, epoch=epoch, shuffle=True)

        loss = train_epoch(config, rank, epoch, model, criterion_lm, criterion_cls, optimizer, scheduler, train_loader)
        losses.append(loss)

        if master:
            best_epoch, best_loss = epoch, loss
            if isinstance(model, DistributedDataParallel):
                model.module.bert.save(best_epoch, best_loss, args.save)
            else:
                model.bert.save(best_epoch, best_loss, args.save)
            print(f">>>> rank: {rank} save model to {args.save}, epoch={best_epoch}, loss={best_loss:.3f}")

    print(f">>>> rank: {rank} losses: {losses}")

    if 1 < args.n_gpu:
        destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgmentPaser()
    
    parser.add_argument("--config", default="data/config.json", type=str, required=False, help="config file")
    parser.add_argument("--vocab", default="data/vocab.model", type=str, required=False, help="vocab file")
    parser.add_argument("--input", default="data/ptr_input_data.json", type=str, required=False, help="input processed corpus data")
    parser.add_argument("--save", default="data/save_pretrain.pt", type=str, required=False, help="save file path")
    parser.add_argument("--epoch", default=20, type=int, required=False, help="epoch")
    parser.add_argument("--batch", default=512, type=int, required=False, help="batch size for pretraining")
    parser.add_argument("--seed", default=1, type=int, required=False, help="random seed for initialization")
    parser.add_argument("--gpu", default=None, type=int, required=False, help="number of gpu id to use, default is None = ALL")
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, required=False, help="decay_rate")
    parser.add_argument("--learning_rate", default=5e-5, type=float, required=False, help="learning_rate")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, required=False, help="adam epsilon")
    parser.add_argument("--warmup_proportion", default=0.1, required=False, help="warmup steps proportion")

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count()
    else:
        arg.n_gpu = 0

    set_seed(args)

    if 1 < args.n_gpu:
        mp.spawn(train_model, args=(n_gpu, args), nprocs=n_gpu, join=True) # https://newsight.tistory.com/323 
                                                                       # https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b

    else:
        train_model(0 if args.gpu is None else args.gpu, args.n_gpu, args)
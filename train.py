import os
import argparse
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import timm
from torch.utils.tensorboard import SummaryWriter

# --- 1. Custom Optimizers ---
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]), p=2
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("SAM requires closure, use first_step and second_step")

class Muon(optim.SGD):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=1e-4):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

# --- 2. Setup ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR100', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'OxfordIIITPet'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model', type=str, default='resnest26d', choices=['resnet18', 'resnet50', 'resnest14d', 'resnest26d', 'mlp'])
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW', 'Muon', 'SAM'])
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', choices=['StepLR', 'ReduceLROnPlateau'])
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=2) # Colab usually prefers 2 workers
    parser.add_argument('--exp_name', type=str, default='experiment_1')
    parser.add_argument('--batch_schedule', type=str, default=None)
    return parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# --- 3. Components ---
def get_transforms(args, is_train=True):
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if args.dataset == 'MNIST': mean, std = (0.1307,), (0.3081,)
    elif args.dataset in ['CIFAR100', 'CIFAR10']: mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    ts = [transforms.Resize((args.image_size, args.image_size))]
    if is_train:
        ts.append(transforms.RandomHorizontalFlip())
        if args.dataset != 'MNIST':
            ts.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
            ts.append(transforms.RandomCrop(args.image_size, padding=4))
        ts.append(transforms.ToTensor())
        ts.append(transforms.Normalize(mean, std))
        ts.append(transforms.RandomErasing(p=0.25))
    else:
        ts.append(transforms.ToTensor())
        ts.append(transforms.Normalize(mean, std))
    return transforms.Compose(ts)

def get_dataset(args):
    # This automatically downloads data to ./data folder
    train_transform = get_transforms(args, is_train=True)
    test_transform = get_transforms(args, is_train=False)

    if args.dataset == 'MNIST':
        train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=train_transform)
        test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=test_transform)
        args.num_classes = 10
    elif args.dataset == 'CIFAR10':
        train_ds = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
        test_ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)
        args.num_classes = 10
    elif args.dataset == 'CIFAR100':
        train_ds = datasets.CIFAR100(args.data_dir, train=True, download=True, transform=train_transform)
        test_ds = datasets.CIFAR100(args.data_dir, train=False, download=True, transform=test_transform)
        args.num_classes = 100
    elif args.dataset == 'OxfordIIITPet':
        train_ds = datasets.OxfordIIITPet(args.data_dir, split='trainval', download=True, transform=train_transform)
        test_ds = datasets.OxfordIIITPet(args.data_dir, split='test', download=True, transform=test_transform)
        args.num_classes = 37
    return train_ds, test_ds

class MLPWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x): return self.net(self.flatten(x))

def get_model(args):
    if args.model == 'mlp':
        input_channels = 1 if args.dataset == 'MNIST' else 3
        input_dim = input_channels * args.image_size * args.image_size
        model = MLPWrapper(input_dim, 512, args.num_classes)
    else:
        model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes, in_chans=1 if args.dataset=='MNIST' else 3)
    return model.to(args.device)

def get_optimizer(model, args):
    if args.optimizer == 'SGD': return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'Adam': return optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'AdamW': return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    elif args.optimizer == 'Muon': return Muon(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SAM':
        base_optim = optim.AdamW if args.pretrained else optim.SGD
        return SAM(model.parameters(), base_optim, lr=args.lr, momentum=0.9)
    return optim.AdamW(model.parameters(), lr=args.lr)

class BatchSizeScheduler:
    def __init__(self, schedule_str):
        self.schedule = {}
        if schedule_str:
            for part in schedule_str.split(','):
                ep, sz = part.split(':')
                self.schedule[int(ep)] = int(sz)
    def get_batch_size(self, epoch, current_size):
        if epoch in self.schedule:
            print(f"Batch Scheduler: Changing batch size to {self.schedule[epoch]}")
            return self.schedule[epoch]
        return current_size

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience; self.min_delta = min_delta; self.counter = 0; self.best_loss = None; self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss is None: self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else: self.best_loss = val_loss; self.counter = 0

# --- 4. Main ---
def main():
    args = get_args()
    seed_everything()
    log_dir = os.path.join("runs", args.exp_name)
    writer = SummaryWriter(log_dir=log_dir)

    train_ds, test_ds = get_dataset(args)
    current_batch_size = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=current_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=current_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args) if args.scheduler else None
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    batch_scheduler = BatchSizeScheduler(args.batch_schedule)
    early_stopping = EarlyStopping(patience=args.patience)
    scaler = GradScaler()

    print(f"Starting {args.exp_name} on {args.device} with {args.model}...")
    best_acc = 0.0

    for epoch in range(args.epochs):
        new_bs = batch_scheduler.get_batch_size(epoch, current_batch_size)
        if new_bs != current_batch_size:
            current_batch_size = new_bs
            train_loader = DataLoader(train_ds, batch_size=current_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

        # Train
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(args.device), lbls.to(args.device)
            if args.optimizer == 'SAM':
                with autocast(): criterion(model(imgs), lbls).backward()
                optimizer.first_step(zero_grad=True)
                with autocast(): criterion(model(imgs), lbls).backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                with autocast():
                    out = model(imgs)
                    loss = criterion(out, lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                r_loss += loss.item()
                _, pred = out.max(1)
                total += lbls.size(0)
                correct += pred.eq(lbls).sum().item()
                pbar.set_postfix({'loss': loss.item()})

        train_loss, train_acc = r_loss/len(train_loader), 100.*correct/total

        # Eval
        model.eval()
        r_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(args.device), lbls.to(args.device)
                out = model(imgs)
                r_loss += criterion(out, lbls).item()
                _, pred = out.max(1)
                total += lbls.size(0)
                correct += pred.eq(lbls).sum().item()
        val_loss, val_acc = r_loss/len(test_loader), 100.*correct/total

        # Update Scheduler
        if args.scheduler == 'ReduceLROnPlateau' and scheduler: scheduler.step(val_loss)
        elif args.scheduler == 'StepLR' and scheduler: scheduler.step()

        print(f"Ep {epoch+1} | TrainAcc: {train_acc:.2f}% | ValAcc: {val_acc:.2f}%")
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)

        if val_acc > best_acc: best_acc = val_acc
        early_stopping(val_loss)
        if early_stopping.early_stop: break

    print(f"Best Acc: {best_acc:.2f}%")
    writer.close()

def get_scheduler(optimizer, args):
    opt = optimizer.base_optimizer if args.optimizer == 'SAM' else optimizer
    if args.scheduler == 'StepLR': return StepLR(opt, step_size=10, gamma=0.1)
    elif args.scheduler == 'ReduceLROnPlateau': return ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)
    return None

if __name__ == "__main__":
    main()

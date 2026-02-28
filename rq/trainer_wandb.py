import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import ensure_dir, set_color, get_local_time, delete_file
import os

import heapq

try:
    import wandb
except ImportError:
    wandb = None


class Trainer(object):

    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir, saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

        # --- WandB init ---
        self.use_wandb = wandb is not None
        if self.use_wandb:
            wandb_project = getattr(args, 'wandb_project', 'rqvae-replication')
            wandb_run_name = getattr(args, 'wandb_run_name', None)
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=vars(args),
            )
            wandb.watch(self.model, log='gradients', log_freq=100)

    def _build_optimizer(self):

        params = self.model.parameters()
        learner = self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}", "pink"),
                    )

        for batch_idx, data in enumerate(iter_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            self._check_nan(loss)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()

        # Log training metrics to wandb
        if self.use_wandb:
            current_lr = self.scheduler.get_last_lr()[0]
            quant_loss = total_loss - total_recon_loss
            wandb.log({
                'train/total_loss': total_loss,
                'train/recon_loss': total_recon_loss,
                'train/quant_loss': quant_loss,
                'train/lr': current_lr,
                'train/grad_norm': grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                'epoch': epoch_idx,
            })

        return total_loss, total_recon_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data = tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )

        indices_set = set()
        # Per-level sets for hierarchical collision rates
        indices_L1 = set()
        indices_L1L2 = set()
        # Per-level code counters
        all_codes_per_level = []  # list of lists, one per VQ level
        num_levels = len(self.model.rq.vq_layers)
        for _ in range(num_levels):
            all_codes_per_level.append([])

        num_sample = 0
        for batch_idx, data in enumerate(iter_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

                # Hierarchical collision tracking
                if num_levels >= 1:
                    indices_L1.add(str(int(index[0])))
                if num_levels >= 2:
                    indices_L1L2.add(f"{int(index[0])}-{int(index[1])}")

                # Per-level code collection
                for lvl in range(num_levels):
                    all_codes_per_level[lvl].append(int(index[lvl]))

        collision_rate = (num_sample - len(list(indices_set))) / num_sample

        # --- Compute per-level metrics ---
        collision_rate_L1 = (num_sample - len(indices_L1)) / num_sample
        collision_rate_L1L2 = (num_sample - len(indices_L1L2)) / num_sample

        # Per-level quantization losses via extra forward pass through each VQ layer
        per_level_quant_losses = []
        codebook_utilizations = []
        code_freq_histograms = []

        # Extra forward: encode all data once
        all_encoded = []
        for batch_idx, data in enumerate(valid_data):
            data = data.to(self.device)
            x_e = self.model.encoder(data)
            all_encoded.append(x_e)
        all_encoded = torch.cat(all_encoded, dim=0)

        # Walk through VQ layers to get per-level losses
        residual = all_encoded
        for lvl, vq_layer in enumerate(self.model.rq.vq_layers):
            x_res, loss, _ = vq_layer(residual, use_sk=False)
            residual = residual - x_res
            per_level_quant_losses.append(loss.item())

            # Codebook utilization
            codes = np.array(all_codes_per_level[lvl])
            unique_codes = len(np.unique(codes))
            total_codes = vq_layer.n_e
            codebook_utilizations.append(unique_codes / total_codes)

            # Code frequency histogram
            freq = np.bincount(codes, minlength=total_codes)
            code_freq_histograms.append(freq)

        # --- Log to wandb ---
        if self.use_wandb:
            log_dict = {
                'eval/collision_rate': collision_rate,
                'eval/collision_rate_L1': collision_rate_L1,
                'eval/collision_rate_L1L2': collision_rate_L1L2,
                'eval/collision_rate_L1L2L3': collision_rate,
                'eval/best_collision_rate': min(self.best_collision_rate, collision_rate),
            }

            level_names = ['L1', 'L2', 'L3']
            for lvl in range(min(num_levels, 3)):
                ln = level_names[lvl]
                log_dict[f'train/quant_loss_{ln}'] = per_level_quant_losses[lvl]
                log_dict[f'eval/codebook_utilization_{ln}'] = codebook_utilizations[lvl]
                log_dict[f'eval/code_freq_{ln}'] = wandb.Histogram(
                    code_freq_histograms[lvl].tolist()
                )

            wandb.log(log_dict)

        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

        return ckpt_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output += ", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        return train_loss_output + "]"

    def fit(self, data):

        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
            )
            self.logger.info(train_loss_output)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)
                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])

        if self.use_wandb:
            wandb.finish()

        return self.best_loss, self.best_collision_rate

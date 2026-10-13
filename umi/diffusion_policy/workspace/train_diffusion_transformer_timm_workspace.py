if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import math
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
import pickle
import dill

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_timm_policy import DiffusionTransformerTimmPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainDiffusionTransformerTimmWorkspace(BaseWorkspace):
    include_keys = [
        'global_step', 'epoch', 'next_global_step', 'next_epoch',
        'ema_optimization_step', 'scaler_state', 'torch_rng_state',
        'cuda_rng_state', 'numpy_rng_state', 'python_rng_state'
    ]

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerTimmPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerTimmPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        self.global_step = 0
        self.epoch = 0
        self.next_global_step = 0
        self.next_epoch = 0
        self.ema_optimization_step = 0
        self.scaler_state = None
        self.torch_rng_state = None
        self.cuda_rng_state = None
        self.numpy_rng_state = None
        self.python_rng_state = None
        
        # Existing experiments keep their lightweight checkpoints. New runs
        # opt into complete resume state before their first checkpoint.
        if not cfg.training.get('save_resume_state', False):
            self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        init_checkpoint = cfg.training.get('init_checkpoint', None)

        # Keep the Transformer entrypoint safe on the same laptop GPUs as the
        # UNet entrypoint.  This is opt-in, so historical configurations retain
        # their original allocation behaviour.
        gpu_memory_fraction = cfg.training.get('gpu_memory_fraction', None)
        if gpu_memory_fraction is not None:
            device = torch.device(cfg.training.device)
            if device.type != 'cuda':
                raise ValueError(
                    'training.gpu_memory_fraction requires a CUDA device, '
                    f'got {cfg.training.device!r}'
                )
            torch.cuda.set_per_process_memory_fraction(
                float(gpu_memory_fraction), device=device
            )

        accelerator = Accelerator(
            log_with='wandb',
            mixed_precision=cfg.training.get('mixed_precision', 'no'),
        )
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        resume_payload = None
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if not lastest_ckpt_path.is_file():
                raise FileNotFoundError(f'Resume checkpoint not found: {lastest_ckpt_path}')
            print(f"Resuming from checkpoint {lastest_ckpt_path}")
            resume_payload = self.load_checkpoint(
                path=lastest_ckpt_path, exclude_keys=('lr_scheduler',),
                map_location='cpu')
            if cfg.training.get('require_exact_resume', False):
                state_keys = resume_payload['state_dicts']
                pickle_keys = resume_payload['pickles']
                missing = [
                    key for key in ('optimizer', 'lr_scheduler')
                    if key not in state_keys
                ] + [
                    key for key in (
                        'next_epoch', 'next_global_step', 'ema_optimization_step',
                        'scaler_state', 'torch_rng_state', 'numpy_rng_state',
                        'python_rng_state'
                    ) if key not in pickle_keys
                ]
                if missing:
                    raise RuntimeError(
                        'Checkpoint lacks exact resume state: '
                        + ', '.join(missing)
                    )
                saved_cfg = OmegaConf.to_container(resume_payload['cfg'], resolve=True)
                current_cfg = OmegaConf.to_container(cfg, resolve=True)
                for section in (
                    'task', 'shape_meta', 'policy', 'n_emb', 'dataloader',
                    'val_dataloader', 'optimizer', 'ema'
                ):
                    if saved_cfg[section] != current_cfg[section]:
                        raise ValueError(f'Resume config differs in {section}')
                for training_cfg in (saved_cfg['training'], current_cfg['training']):
                    training_cfg.pop('resume', None)
                    training_cfg.pop('require_exact_resume', None)
                if saved_cfg['training'] != current_cfg['training']:
                    raise ValueError('Resume config differs in training')
            if 'next_epoch' in resume_payload['pickles']:
                self.epoch = self.next_epoch
                self.global_step = self.next_global_step
            else:
                # Historical checkpoints were written before incrementing
                # epoch and global_step at the end of each epoch.
                self.epoch += 1
                self.global_step += 1
            if 'optimizer' not in resume_payload['state_dicts']:
                print('WARNING: optimizer state absent; resuming weights '
                      'with a fresh optimizer (warm restart)')

        if init_checkpoint and not cfg.training.resume:
            source_path = pathlib.Path(init_checkpoint).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f'Initialization checkpoint not found: {source_path}')
            source_key = cfg.training.get('init_weights', 'ema_model')
            print(f'Initializing policy weights from {source_path} [{source_key}]')
            payload = torch.load(source_path, map_location='cpu',
                                 pickle_module=dill, mmap=True, weights_only=False)
            if source_key not in payload['state_dicts']:
                raise KeyError(f'{source_key} absent from {source_path}')
            weights = payload['state_dicts'][source_key]
            self.model.load_state_dict(weights, strict=True)
            if self.ema_model is not None:
                self.ema_model.load_state_dict(weights, strict=True)
            if not self.model.normalizer.params_dict:
                raise RuntimeError('Initialization checkpoint has no policy normalizer')
            del weights, payload
            print('Initialized model and EMA; optimizer, scheduler and counters start fresh')

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            # Fine-tuned weights expect the action and observation scales used
            # when they were learned. The new episodes still train normally.
            normalizer = (copy.deepcopy(self.model.normalizer) if init_checkpoint
                          else dataset.get_normalizer())
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler. Respect max_train_steps when the epoch loop
        # is truncated below; otherwise the scheduler is sized for the full
        # dataloader while training performs far fewer optimizer updates.
        steps_per_epoch = len(train_dataloader)
        if cfg.training.max_train_steps is not None:
            steps_per_epoch = min(steps_per_epoch, int(cfg.training.max_train_steps))
        accumulation_steps = int(cfg.training.gradient_accumulate_every)
        if accumulation_steps < 1:
            raise ValueError('training.gradient_accumulate_every must be >= 1')
        total_optimizer_steps = (
            math.ceil(steps_per_epoch / accumulation_steps)
            * int(cfg.training.num_epochs)
        )
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=total_optimizer_steps,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=-1 if resume_payload is not None else self.global_step-1
        )
        if (resume_payload is not None
                and 'lr_scheduler' not in resume_payload['state_dicts']):
            # Warm restart from an older weights-only checkpoint. The exact
            # optimizer history is unavailable, but retain its LR position.
            completed_updates = self.epoch * math.ceil(steps_per_epoch / accumulation_steps)
            lr_scheduler.step(completed_updates)

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            if resume_payload is not None:
                if 'ema_optimization_step' in resume_payload['pickles']:
                    ema.optimization_step = self.ema_optimization_step
                else:
                    ema.optimization_step = self.epoch * math.ceil(steps_per_epoch / accumulation_steps)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        # device = torch.device(cfg.training.device)
        # self.model.to(device)
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        # optimizer_to(self.optimizer, device)

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        self.lr_scheduler = lr_scheduler
        if resume_payload is not None:
            scheduler_state = resume_payload['state_dicts'].get('lr_scheduler')
            if scheduler_state is not None:
                lr_scheduler.load_state_dict(scheduler_state)
            if self.scaler_state is not None and accelerator.scaler is not None:
                accelerator.scaler.load_state_dict(self.scaler_state)
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        if resume_payload is not None and self.torch_rng_state is not None:
            torch.set_rng_state(self.torch_rng_state)
            if self.cuda_rng_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(self.cuda_rng_state)
            np.random.set_state(self.numpy_rng_state)
            random.setstate(self.python_rng_state)
        if resume_payload is not None:
            del resume_payload


        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # always use the latest batch
                        train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        group_start = (batch_idx // accumulation_steps) * accumulation_steps
                        group_size = min(accumulation_steps, steps_per_epoch - group_start)
                        loss = raw_loss / group_size
                        accelerator.backward(loss)

                        # step optimizer
                        if (batch_idx - group_start + 1) == group_size:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            if cfg.training.use_ema:
                                ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                # if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0 and accelerator.is_main_process:
                #     with torch.no_grad():
                #         val_losses = list()
                #         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                #             for batch_idx, batch in enumerate(tepoch):
                #                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                #                 loss = self.model(batch)
                #                 val_losses.append(loss)
                #                 if (cfg.training.max_val_steps is not None) \
                #                     and batch_idx >= (cfg.training.max_val_steps-1):
                #                     break
                #         if len(val_losses) > 0:
                #             val_loss = torch.mean(torch.tensor(val_losses)).item()
                #             # log epoch average validation loss
                #             step_log['val_loss'] = val_loss

                def log_action_mse(step_log, category, pred_action, gt_action):
                    B, T, _ = pred_action.shape
                    pred_action = pred_action.view(B, T, -1, 10)
                    gt_action = gt_action.view(B, T, -1, 10)
                    step_log[f'{category}_action_mse_error'] = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log[f'{category}_action_mse_error_pos'] = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
                    step_log[f'{category}_action_mse_error_rot'] = torch.nn.functional.mse_loss(pred_action[..., 3:9], gt_action[..., 3:9])
                    step_log[f'{category}_action_mse_error_width'] = torch.nn.functional.mse_loss(pred_action[..., 9], gt_action[..., 9])
                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        gt_action = batch['action']
                        pred_action = policy.predict_action(batch['obs'])['action_pred']
                        log_action_mse(step_log, 'train', pred_action, gt_action)

                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            gt_action = batch['action']
                            pred_action = policy.predict_action(batch['obs'])['action_pred']
                            log_action_mse(step_log, 'val', pred_action, gt_action)

                        del batch
                        del gt_action
                        del pred_action
                
                # checkpoint
                if accelerator.is_main_process and (
                    (self.epoch % cfg.training.checkpoint_every) == 0
                    or (self.epoch + 1) == cfg.training.num_epochs
                ):
                    self.next_epoch = self.epoch + 1
                    self.next_global_step = self.global_step + 1
                    if cfg.training.use_ema:
                        self.ema_optimization_step = ema.optimization_step
                    if cfg.training.get('save_resume_state', False):
                        self.scaler_state = (
                            accelerator.scaler.state_dict()
                            if accelerator.scaler is not None else None
                        )
                        self.torch_rng_state = torch.get_rng_state()
                        self.cuda_rng_state = (
                            torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else None
                        )
                        self.numpy_rng_state = np.random.get_state()
                        self.python_rng_state = random.getstate()
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(use_thread=False)
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path,
                                             use_thread=False)

                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerTimmWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

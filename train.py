import argparse
from collections import OrderedDict, defaultdict
import contextlib
import copy
from copy import deepcopy
import enum
import functools
import gc
import importlib.util
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import random
from time import time
from typing import Dict

from PIL import Image
from diffusers.models import AutoencoderKLCogVideoX
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
import wandb

from data import DataNoReportException, ItemProcessor, LenClusteringSampler, MyDataset, read_general
from imgproc import generate_crop_size_list, to_rgb_if_rgba, var_center_crop
import models
from transport import create_transport
from utils.ckpt import remove_wrapper_name_from_state_dict
from utils.misc import SmoothedValue, random_seed, to_item
from utils.parallel import distributed_init, get_intra_node_process_group, set_sequence_parallel

logger = logging.getLogger(__name__)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

#############################################################################
#                            Data item Processor                            #
#############################################################################


class MediaTensorType(enum.Enum):
    VAE_IN = enum.auto()
    LATENT = enum.auto()


class NonRGBError(DataNoReportException):
    pass


SYSTEM_PROMPT = "You are an assistant designed to generate high-quality videos with the highest degree of image-text alignment based on user prompts. <Prompt Start> "  # noqa


class T2IItemProcessor(ItemProcessor):
    def __init__(self, transform):
        self.image_transform = transform

    def process_item(self, data_item, meta, training_mode=False):

        image_path = data_item["path"]
        image = Image.open(read_general(image_path))
        text = data_item["prompt"]
        system_prompt = SYSTEM_PROMPT

        if image.mode.upper() != "RGB":
            mode = image.mode.upper()
            if mode == "RGBA":
                image = to_rgb_if_rgba(image)
            else:
                raise NonRGBError()

        image = self.image_transform(image).unsqueeze(1)  # C, F, H, W

        if text is None or text.strip() == "":
            text = ""

        text = system_prompt + text

        text_cfg_rand_val = random.random()
        if text_cfg_rand_val < 0.8:
            pass
        else:
            text = system_prompt

        if random.random() < 0.4:
            motion_score = 0.0
        else:
            motion_score = -1.0

        return image, text, MediaTensorType.VAE_IN, motion_score

    def extract_quick_info(self, data_item):
        # provide useful information for compute item length and then cluster items with similar lengths into a batch
        return {}

    def len_from_info(self, info):
        return 1


class T2VItemProcessor(ItemProcessor):
    def __init__(self, transform, max_frames, target_frame_rate, max_hidden_frames, patch_size=2):
        self.image_transform = transform
        self.max_frames = max_frames
        self.target_frame_rate = target_frame_rate
        self.max_hidden_frames = max_hidden_frames
        self.patch_size = patch_size
        # self.printed = False

    def select_frames_and_form_tensor(self, data_item):
        # decord has to be imported after torch, bug here: https://github.com/dmlc/decord/issues/293
        import decord

        url = data_item["path"]

        video_data = read_general(url)

        video_reader = decord.VideoReader(video_data)

        if video_reader.get_avg_fps() < self.target_frame_rate / 2:
            raise ValueError(f"fps {video_reader.get_avg_fps():.2f} too low {url}")

        if len(video_reader) / video_reader.get_avg_fps() < self.max_frames / self.target_frame_rate:
            seconds = len(video_reader) / video_reader.get_avg_fps()
            # print(f"item length do not qualify, only {seconds} s.")
            target_frames = int(seconds * self.target_frame_rate) // 8 * 8
        else:
            target_frames = self.max_frames

        assert (
            target_frames > 0
        ), f"Unqualified video: {url} of len {len(video_reader)} and fps {video_reader.get_avg_fps()}"

        start_time = random.uniform(
            0, max(0, len(video_reader) / video_reader.get_avg_fps() - self.max_frames / self.target_frame_rate)
        )

        l_frame_ids = []
        for i in range(target_frames):
            frame_id = round((start_time + i / self.target_frame_rate) * video_reader.get_avg_fps())
            if frame_id >= len(video_reader):
                frame_id = len(video_reader) - 1

            l_frame_ids.append(frame_id)

        frames_npy = video_reader.get_batch(l_frame_ids).asnumpy()
        # if not self.printed:
        #     print("fps: ", video_reader.get_avg_fps())
        #     print("F: ", len(video_reader))
        #     print("size: ", Image.fromarray(frames_npy[0]).size)
        #     self.printed = True

        frames = [self.image_transform(Image.fromarray(frames_npy[i])) for i in range(frames_npy.shape[0])]

        return frames

    def process_item(self, data_item, meta, training_mode=False):

        caption = data_item["prompt"]
        system_prompt = SYSTEM_PROMPT

        caption = system_prompt + caption

        cfg_rand_val = random.random()
        if cfg_rand_val < 0.8:
            pass
        else:
            caption = system_prompt

        motion_score = data_item["unimatch_flow"]
        if random.random() < 0.3:
            motion_score = -1.0

        if "pkl" in data_item:
            latent = pickle.load(read_general(data_item["pkl"]))["latent"]
            latent = latent[:, 1:]  # avoid using the first frame due to potential artifacts
            c, f, h, w = latent.shape

            if f <= self.max_hidden_frames:
                latent = latent
            else:
                start_idx = random.randint(0, max(0, f - self.max_hidden_frames))
                latent = latent[:, start_idx : start_idx + self.max_hidden_frames]

            if h % self.patch_size != 0:
                total_crop = h % self.patch_size
                top_crop = total_crop // 2
                bottom_crop = total_crop - top_crop
                latent = latent[:, :, top_crop:-bottom_crop]

            if w % self.patch_size != 0:
                total_crop = w % self.patch_size
                left_crop = total_crop // 2
                right_crop = total_crop - left_crop
                latent = latent[:, :, :, left_crop:-right_crop]

            return latent, caption, MediaTensorType.LATENT, motion_score
        else:
            vae_in = self.select_frames_and_form_tensor(data_item)
            return vae_in, caption, MediaTensorType.VAE_IN, motion_score

    def extract_quick_info(self, data_item):
        # provide useful information for compute item length and then cluster items with similar lengths into a batch
        return {"seconds": data_item["seconds"]}

    def len_from_info(self, info):
        return min(int(float(info["seconds"]) * self.target_frame_rate), self.max_frames)


#############################################################################
#                           Training Helper Functions                       #
#############################################################################


def dataloader_collate_fn(samples):
    index = [x[0] for x in samples]
    image = [x[1][0] for x in samples]
    caps = [x[1][1] for x in samples]
    media_tensor_types = [x[1][2] for x in samples]
    motion_scores = torch.tensor([x[1][3] for x in samples], dtype=torch.float32)
    return index, image, caps, media_tensor_types, motion_scores


def get_train_sampler(dataset, task_name, rank, world_size, global_batch_size, resume_step, seed):

    # the length clustering sampler is epoch based
    # here we modify it to be iteration based
    len_clustering_sampler = LenClusteringSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        batch_size=global_batch_size // world_size,
        acc_grad=1,
        length_clustering=True,
        allow_mixed_task_among_acc=True,  # faster, no negative effect in our case as accumulate_grad=1
    )

    steps_per_epoch = len(len_clustering_sampler) // (global_batch_size // world_size)

    epoch_id, step_in_epoch = resume_step // steps_per_epoch, resume_step % steps_per_epoch
    len_clustering_sampler.set_epoch(epoch_id, step_in_epoch)
    logger.info(f"{task_name} epoch {epoch_id} step {step_in_epoch}")

    indices_cache = len_clustering_sampler.get_local_indices()

    while True:
        if len(indices_cache) == 0:
            epoch_id += 1
            step_in_epoch = 0
            len_clustering_sampler.set_epoch(epoch_id, step_in_epoch)
            logger.info(f"{task_name} epoch {epoch_id} step {step_in_epoch}")
            indices_cache = len_clustering_sampler.get_local_indices()

        index = indices_cache.pop(0)
        yield index


@torch.no_grad()
def update_ema(ema_model, model, decay=0.99):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.layers),
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model


def setup_mixed_precision(args):
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer):
    captions = []
    for caption in prompt_batch:
        captions.append(caption)

    with torch.no_grad():

        # print(f"rank {dist.get_rank()} start tokenizer", flush=True)
        # in some recent transformers version, the following becomes extremely slow
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        # print(f"rank {dist.get_rank()} end tokenizer", flush=True)

        text_input_ids = text_inputs.input_ids.cuda()
        prompt_masks = text_inputs.attention_mask.cuda()

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks


def wandb_process_main(write_queue, write_barrier, args, task_name):
    wandb_path = os.path.join(args.results_dir, f"wandb/{task_name}")
    os.makedirs(wandb_path, exist_ok=True)
    wandb_logger = wandb.init(
        project="Lumina-video",
        name=args.results_dir.split("/")[-1] + f"--{task_name}",
        config=args.__dict__,  # Use args.__dict__ to pass all arguments
        dir=wandb_path,  # Set the directory for wandb files
        job_type="training",
    )

    write_barrier.wait()
    while True:
        (records, step) = write_queue.get()
        wandb_logger.log(records, step=step)
        write_barrier.wait()


#############################################################################
#                                Training Loop                              #
#############################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    distributed_init(args)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()

    torch.cuda.set_device(device)

    torch.backends.cudnn.benchmark = False
    setup_mixed_precision(args)

    seed = args.global_seed
    random_seed(seed)

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory: {args.results_dir}")
    else:
        logger = create_logger(None)

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))

    logger.info(f"Setting-up language model: google/gemma-2-2b")

    # create tokenizers
    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    tokenizer.padding_side = "right"

    # create text encoders
    # text_encoder
    text_encoder = AutoModel.from_pretrained(
        "google/gemma-2-2b",
        torch_dtype=torch.bfloat16,
    ).cuda()
    text_encoder.eval()
    # text_encoder = setup_lm_fsdp_sync(text_encoder)
    logger.info(f"text encoder: {type(text_encoder)}")
    cap_feat_dim = text_encoder.config.hidden_size
    text_encoder = torch.compile(text_encoder)

    # Create model:
    model = models.__dict__[args.model](
        all_patch_size=args.patch_sizes,
        all_f_patch_size=args.f_patch_sizes,
        in_channels=16,
        qk_norm=args.qk_norm,
        cap_feat_dim=cap_feat_dim,
        rope_theta=args.rope_theta,
        t_scale=args.t_scale,
        motion_scale=args.motion_scale,
    )
    logger.info(f"DiT Parameters: {model.parameter_count():,}")

    if args.auto_resume:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
                args.no_resume_opt = False
                args.no_resume_ema = False
                args.no_resume_step = False
                args.main_load_ema = False
        except Exception:
            pass
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    # Note that parameter initialization is done within the DiT constructor
    model_ema = deepcopy(model)
    if args.resume:
        if rank == 0:  # other ranks receive weights in setup_fsdp_sync
            if args.main_load_ema:
                load_path = os.path.join(args.resume, f"consolidated_ema.{0:02d}-of-{1:02d}.pth")
            else:
                load_path = os.path.join(args.resume, f"consolidated.{0:02d}-of-{1:02d}.pth")
            logger.info(f"Resuming model weights from: {load_path}")
            model.load_state_dict(
                remove_wrapper_name_from_state_dict(
                    torch.load(
                        load_path,
                        map_location="cpu",
                    ),
                ),
                strict=True,
            )
            if args.no_resume_ema:  # use main weights to initialize ema
                load_path = os.path.join(args.resume, f"consolidated.{0:02d}-of-{1:02d}.pth")
            else:
                load_path = os.path.join(args.resume, f"consolidated_ema.{0:02d}-of-{1:02d}.pth")
            logger.info(f"Resuming ema weights from: {load_path}")
            model_ema.load_state_dict(
                remove_wrapper_name_from_state_dict(
                    torch.load(
                        load_path,
                        map_location="cpu",
                    ),
                ),
                strict=True,
            )
    elif args.init_from:
        if rank == 0:
            if args.main_load_ema:
                load_path = os.path.join(args.init_from, f"consolidated_ema.{0:02d}-of-{1:02d}.pth")
            else:
                load_path = os.path.join(args.init_from, f"consolidated.{0:02d}-of-{1:02d}.pth")
            logger.info(f"Initializing model and ema weights from: {load_path}")
            state_dict = remove_wrapper_name_from_state_dict(
                torch.load(
                    load_path,
                    map_location="cpu",
                )
            )

            size_mismatch_keys = []
            model_state_dict = model.state_dict()
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape != v.shape:
                    size_mismatch_keys.append(k)
            for k in size_mismatch_keys:
                del state_dict[k]
            del model_state_dict

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            missing_keys_ema, unexpected_keys_ema = model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            assert set(missing_keys) == set(missing_keys_ema)
            assert set(unexpected_keys) == set(unexpected_keys_ema)
            logger.info("Model initialization result:")
            logger.info(f"  Size mismatch keys: {size_mismatch_keys}")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpeected keys: {unexpected_keys}")
    logger.info(f"rank{rank} barrier")
    dist.barrier()

    # checkpointing (part1, should be called before FSDP wrapping)
    if args.checkpointing:
        model.my_checkpointing()
        model_ema.my_checkpointing()

    if args.compile:
        logger.info("model is compiled")
        model.my_compile()
        model_ema.my_compile()

    model = setup_fsdp_sync(model, args)
    model_ema = setup_fsdp_sync(model_ema, args)

    logger.info(f"model:\n{model}\n")

    vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.bfloat16).to(
        device
    )
    vae = torch.compile(vae)

    logger.info("AdamW eps 1e-15 betas (0.9, 0.95)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=1e-15, betas=(0.9, 0.95))
    if args.resume and not args.no_resume_opt:
        opt_state_world_size = len(
            [x for x in os.listdir(args.resume) if x.startswith("optimizer.") and x.endswith(".pth")]
        )
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt.load_state_dict(
            torch.load(
                os.path.join(
                    args.resume,
                    f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth",
                ),
                map_location="cpu",
            )
        )
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr
            param_group["weight_decay"] = args.wd

    if args.resume and not args.no_resume_step:
        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0

    spec = importlib.util.spec_from_file_location("task_config", args.task_config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    all_task_config: Dict = config.tasks

    logger.info(f"all_task_config:\n{all_task_config}")

    # For task sampling with difference ratios
    # Example:
    #   when task_ratio=7:3:2, the cycle period of task sampling will be 7,
    #   so in every 7 iterations, the 1st task will be trained in every iteration,
    #   the 2nd task will be trained only in the first 3 iterations,
    #   and the 3rd task will be trained only in the first 2 iterations
    task_sample_cycle_period = max([task_config["task_ratio"] for task_config in all_task_config.values()])

    task_collection = {}
    for task_id, (task_name, task_config) in enumerate(all_task_config.items()):
        logger.info(f"Creating suit for task {task_name}")

        logger.info(f"task_config:\n{task_config}")

        sp_size = task_config.get("sp_size", 1)
        global_bsz = task_config["global_bsz"]
        micro_bsz = task_config["micro_bsz"]

        assert world_size % sp_size == 0
        dp_size = world_size // sp_size
        dp_rank = rank // sp_size

        assert global_bsz % dp_size == 0, "Batch size must be divisible by data parallel size."
        local_bsz = global_bsz // dp_size

        logger.info(f"Data Parallel: {dp_size} Sequence Parallel: {sp_size}")
        logger.info(f"Global bsz: {global_bsz} Local bsz: {local_bsz} Micro bsz: {micro_bsz}")

        # Setup data:
        # patch_size = 8 * task_config["patch_size"]
        patch_size = 8 * 4
        logger.info(f"patch size for crop size computation: {patch_size}")
        max_num_patches = round((task_config["image_size"] / patch_size) ** 2)
        logger.info(f"Limiting number of patches to {max_num_patches}.")
        crop_size_list = generate_crop_size_list(max_num_patches, patch_size)
        logger.info("List of crop sizes:")
        for line in range(0, len(crop_size_list), 6):
            logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in crop_size_list[line : line + 6]]))
        transform = transforms.Compose(
            [
                transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list, random_top_k=1)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        if task_config["media_type"] == "video":
            max_hidden_frames = task_config["max_frames"] // 4
            logger.info(
                f"max frames: {task_config['max_frames']} "
                f"max hidden frames: {max_hidden_frames} "
                f"FPS: {task_config['target_fps']} "
                f"Size: {task_config['image_size']}"
            )
            item_processor = T2VItemProcessor(
                transform,
                task_config["max_frames"],
                task_config["target_fps"],
                max_hidden_frames,
                patch_size=4,
                # task_config["patch_size"],
            )
        elif task_config["media_type"] == "image":
            logger.info(f"Size: {task_config['image_size']}")
            item_processor = T2IItemProcessor(transform)
        else:
            raise ValueError(f"Unsupported media type: {task_config['media_type']}")
        dataset = MyDataset(
            task_config["data"],
            item_processor=item_processor,
            cache_on_disk=args.cache_data_on_disk,
        )
        task_sample_prop = task_config["task_ratio"] / task_sample_cycle_period
        task_real_max_steps = math.ceil(args.max_steps * task_sample_prop)
        total_num_samples = global_bsz * task_real_max_steps
        logger.info(f"Dataset contains {len(dataset):,} samples")
        logger.info(
            f"Total # video samples to consume: {total_num_samples:,} "
            f"(global_bsz {global_bsz} * max_steps {args.max_steps} * sample_proportion {task_sample_prop}), "
            f"equivalent to {total_num_samples / len(dataset):.2f} epochs"
        )

        task_real_resume_step = resume_step // task_sample_cycle_period * task_config["task_ratio"] + min(
            resume_step % task_sample_cycle_period, task_config["task_ratio"]
        )
        sampler = get_train_sampler(
            dataset,
            task_name,
            dp_rank,
            dp_size,
            global_bsz,
            task_real_resume_step,
            args.global_seed + task_id * 100,
        )

        loader = DataLoader(
            dataset,
            batch_size=local_bsz,
            sampler=sampler,
            num_workers=task_config["num_workers"],
            pin_memory=True,
            collate_fn=dataloader_collate_fn,
            worker_init_fn=lambda worker_id: random_seed(args.global_seed + task_id * 100 + dp_rank),
        )

        logger.info(f"snr_type {task_config['snr_type']}  time shift {task_config['time_shift']}")
        # default: 1000 steps, linear noise schedule
        transport = create_transport(
            "Linear",
            "velocity",
            None,
            None,
            None,
            snr_type=task_config["snr_type"],
            time_shift=task_config["time_shift"],
            generator=torch.Generator(device="cuda").manual_seed(args.global_seed + task_id * 100 + dp_rank),
        )  # default: velocity;
        task_pack = copy.deepcopy(task_config)
        task_pack.update(
            {
                "loader": loader,
                "loader_iter": iter(loader),
                "metrics": defaultdict(lambda: SmoothedValue(task_config["task_ratio"])),
                "transport": transport,
                "local_bsz": local_bsz,
                "sp_size": sp_size,
            }
        )

        if rank == 0 and args.use_wandb:
            wandb_queue = mp.Queue()
            wandb_barrier = mp.Barrier(2)
            wandb_process = mp.Process(
                target=wandb_process_main,
                args=(wandb_queue, wandb_barrier, args, task_name),
            )
            wandb_process.start()
            task_pack["wandb_queue"] = wandb_queue
            task_pack["wandb_barrier"] = wandb_barrier
            task_pack["wandb_process"] = wandb_process
        else:
            task_pack["wandb_queue"] = None
            task_pack["wandb_barrier"] = None
            task_pack["wandb_process"] = None

        task_collection[task_name] = task_pack

    # Prepare models for training:
    model.train()

    logger.info(f"Training for {args.max_steps:,} steps...")

    peak_memory = 0.0

    for step in range(resume_step, args.max_steps):

        # skip some tasks
        # Example:
        #   task_ratio=4:3:2, so task_ratio_loop_steps=4
        #   when step_in_loop=2 or 3, the 3rd task will be skipped
        step_in_loop = step % task_sample_cycle_period
        sampled_tasks_this_step = {
            task_name: task_pack
            for task_name, task_pack in task_collection.items()
            if step_in_loop < task_pack["task_ratio"]
        }

        for task_name, task_pack in sampled_tasks_this_step.items():
            try:
                start_time = time()

                set_sequence_parallel(task_pack["sp_size"])

                data_start_time = time()
                indices, x, caps, media_tensor_types, motion_scores = next(task_pack["loader_iter"])
                data_end_time = time()

                motion_scores = motion_scores.cuda()
                x = [item.to(device=device, dtype=torch.bfloat16, non_blocking=True) for item in x]

                with torch.no_grad():
                    vae_scale = {
                        "sdxl": 0.13025,
                        "sd3": 1.5305,
                        "ema": 0.18215,
                        "mse": 0.18215,
                        "cogvideox": 1.15258426,
                    }["cogvideox"]
                    vae_shift = {"sdxl": 0.0, "sd3": 0.0609, "ema": 0.0, "mse": 0.0, "cogvideox": 0.0}["cogvideox"]

                    if step == resume_step:
                        logger.info(f"vae scale: {vae_scale}    vae shift: {vae_shift}")

                    loss_mask = []

                    for i, (item, tensor_type) in enumerate(zip(x, media_tensor_types)):
                        if tensor_type == MediaTensorType.LATENT:
                            pass
                        elif tensor_type == MediaTensorType.VAE_IN:
                            vae._clear_fake_context_parallel_cache()
                            x[i] = ((vae.encode(item[None]).latent_dist.mode()[0] - vae_shift) * vae_scale).float()
                            vae._clear_fake_context_parallel_cache()
                        else:
                            raise ValueError(f"Unsupported tensor type: {tensor_type}")

                        if task_pack["media_type"] == "image":
                            if task_pack["f_patch_size"] > 1 and x[i].shape[1] == 1:
                                x[i] = x[i].repeat(1, task_pack["f_patch_size"], 1, 1)
                                assert x[i].shape[1] == task_pack["f_patch_size"]
                                item_loss_mask = torch.ones_like(x[i], dtype=torch.bool)
                                item_loss_mask[:, 1:] = False
                                loss_mask.append(item_loss_mask)
                            else:
                                loss_mask.append(None)
                        elif task_pack["media_type"] == "video":
                            x[i] = x[i][:, : x[i].shape[1] // task_pack["f_patch_size"] * task_pack["f_patch_size"]]
                            loss_mask.append(None)

                        if step - resume_step < 1:
                            print(indices[i], x[i].shape)

                with torch.no_grad():
                    text_encoder_start_time = time()
                    cap_feats, cap_mask = encode_prompt(caps, text_encoder, tokenizer)
                    text_encoder_end_time = time()

                loss_item = 0.0
                # Number of bins, for loss recording
                n_loss_bins = 20
                # Create bins for t
                loss_bins = torch.linspace(0.0, 1.0, n_loss_bins + 1, device="cuda")
                # Initialize occurrence and sum tensors
                bin_occurrence = torch.zeros(n_loss_bins, device="cuda")
                bin_sum_loss = torch.zeros(n_loss_bins, device="cuda")
                bin_sum_vd_loss = torch.zeros(n_loss_bins, device="cuda")
                sub_losses = defaultdict(list)

                accumulate_iters = (task_pack["local_bsz"] - 1) // task_pack["micro_bsz"] + 1
                for mb_idx in range(accumulate_iters):
                    mb_st = mb_idx * task_pack["micro_bsz"]
                    mb_ed = min((mb_idx + 1) * task_pack["micro_bsz"], task_pack["local_bsz"])

                    x_mb = x[mb_st:mb_ed]
                    cap_feats_mb = cap_feats[mb_st:mb_ed]
                    cap_mask_mb = cap_mask[mb_st:mb_ed]
                    loss_mask_mb = loss_mask[mb_st:mb_ed]
                    motion_score_mb = motion_scores[mb_st:mb_ed]

                    model_kwargs = dict(
                        cap_feats=cap_feats_mb,
                        cap_mask=cap_mask_mb,
                        motion_score=motion_score_mb,
                        patch_size=task_pack["patch_size"],
                        f_patch_size=task_pack["f_patch_size"],
                    )
                    with {
                        "bf16": torch.amp.autocast("cuda", dtype=torch.bfloat16),
                        "fp16": torch.amp.autocast("cuda", dtype=torch.float16),
                        "fp32": contextlib.nullcontext(),
                        "tf32": contextlib.nullcontext(),
                    }[args.precision]:
                        loss_dict = task_pack["transport"].training_losses(
                            model, x_mb, loss_mask_mb, model_kwargs, vd_weight=args.vd_weight
                        )
                    loss = loss_dict["loss"].sum() / task_pack["local_bsz"]
                    loss_item += loss.item()

                    last_mb = mb_idx == accumulate_iters - 1
                    if last_mb:
                        assert mb_ed == task_pack["local_bsz"]

                    grad_sync = (not args.data_parallel in ["sdp"]) or (
                        task_name == list(sampled_tasks_this_step.keys())[-1] and last_mb
                    )
                    with contextlib.nullcontext() if grad_sync else model.no_sync():
                        loss.backward()

                    # for bin-wise loss recording
                    # Digitize t values to find which bin they belong to
                    bin_indices = torch.bucketize(loss_dict["t"].cuda(), loss_bins, right=True) - 1
                    task_loss = loss_dict["sub_losses"]["task_loss"]
                    vd_loss = loss_dict["sub_losses"]["vd_loss"]

                    # Iterate through each bin index to update occurrence and sum
                    for i in range(n_loss_bins):
                        mask = bin_indices == i  # Mask for elements in the i-th bin
                        bin_occurrence[i] = bin_occurrence[i] + mask.sum()  # Count occurrences in the i-th bin
                        bin_sum_loss[i] = bin_sum_loss[i] + task_loss[mask].sum()  # Sum loss values in the i-th bin
                        bin_sum_vd_loss[i] = bin_sum_vd_loss[i] + vd_loss[mask].sum()

                    for sub_loss_name, value in loss_dict["sub_losses"].items():
                        sub_losses[sub_loss_name].append(value)

                async_hooks = []
                loss_item = torch.as_tensor(loss_item, device=device, dtype=torch.float32)
                loss_item = loss_item / dist.get_world_size()
                async_hooks.append(dist.all_reduce(loss_item, async_op=True))

                for sub_loss_name, value in sub_losses.items():
                    value = torch.cat(value).mean()
                    value = value / dist.get_world_size()
                    async_hooks.append(dist.all_reduce(value, async_op=True))
                    sub_losses[sub_loss_name] = value

                async_hooks.append(dist.all_reduce(bin_occurrence, async_op=True))
                async_hooks.append(dist.all_reduce(bin_sum_loss, async_op=True))
                async_hooks.append(dist.all_reduce(bin_sum_vd_loss, async_op=True))

                for hook in async_hooks:
                    hook.wait()

                if step == resume_step:
                    logger.info(f"t bin for task {task_name}: {bin_occurrence}")

                records = {
                    "loss": to_item(loss_item),
                    "lr": opt.param_groups[0]["lr"],
                }
                for sub_loss_name, value in sub_losses.items():
                    records[sub_loss_name] = to_item(value)
                for i in range(n_loss_bins):
                    if bin_occurrence[i] > 0:
                        bin_avg_loss = (bin_sum_loss[i] / bin_occurrence[i]).item()
                        bin_avg_vd_loss = (bin_sum_vd_loss[i] / bin_occurrence[i]).item()
                        records[f"loss-bin{i + 1}-{n_loss_bins}"] = to_item(bin_avg_loss)
                        records[f"vd-bin{i + 1}-{n_loss_bins}"] = to_item(bin_avg_vd_loss)

                if task_pack["wandb_queue"] is not None:
                    task_pack["records"] = records
                # gc.collect()
                # torch.cuda.empty_cache()
                end_time = time()

                # Log loss values:
                metrics = task_pack["metrics"]
                metrics["loss"].update(loss_item)
                metrics["Secs/Step"].update(end_time - start_time)
                metrics["t-data"].update(data_end_time - data_start_time)
                metrics["t-text"].update(text_encoder_end_time - text_encoder_start_time)
                metrics["Items/Sec"].update(task_pack["global_bsz"] / (end_time - start_time))

                # record memory status to clarify which task is the bottleneck that costs the most memory
                current_memory = torch.cuda.max_memory_allocated() / (1024**3)
                if current_memory > peak_memory:
                    # the current task has made new peak
                    peak_memory = current_memory
                    logger.info(f"Peak memory raised to {peak_memory:.2f}GB by task {task_name}")
                elif current_memory < peak_memory:
                    # GPU memory has undergone re-allocation, which usually indicates too-high GPU memory load
                    peak_memory = current_memory
                    logger.info(
                        f"Peak memory decreased to {peak_memory:2f}GB by task {task_name}.\n"
                        f"This indicates memory re-allocation"
                    )

                if step_in_loop + 1 == task_pack["task_ratio"]:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    logger.info(
                        f"{task_name}: (step{step + 1:07d}) "
                        + f"lr:{opt.param_groups[0]['lr']:.6f} "
                        + " ".join([f"{key}:{str(val)}" for key, val in metrics.items()])
                        + f" Memory: {current_memory:.2f}GB"
                    )
            except Exception as e:
                print(f"rank {rank} exception when working with task {task_name}", flush=True)
                raise e

        grad_norm = model.clip_grad_norm_(max_norm=args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        for task_name, task_pack in sampled_tasks_this_step.items():
            if task_pack["wandb_queue"] is not None:
                records = task_pack.pop("records")
                records["grad_norm"] = to_item(grad_norm)
                task_pack["wandb_barrier"].wait()
                task_pack["wandb_queue"].put((records, step))

        if (step + 1) % 10 == 0:
            update_ema(model_ema, model)

        # Save DiT checkpoint:
        if step == 0 or (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(checkpoint_path, exist_ok=True)

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = model.state_dict()
                if dp_rank == 0:
                    consolidated_fn = "consolidated." f"{0:02d}-of-" f"{1:02d}" ".pth"
                    torch.save(
                        consolidated_model_state_dict,
                        os.path.join(checkpoint_path, consolidated_fn),
                    )
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved consolidated to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_ema_state_dict = model_ema.state_dict()
                if dp_rank == 0:
                    consolidated_ema_fn = "consolidated_ema." f"{0:02d}-of-" f"{1:02d}" ".pth"
                    torch.save(
                        consolidated_ema_state_dict,
                        os.path.join(checkpoint_path, consolidated_ema_fn),
                    )
            dist.barrier()
            del consolidated_ema_state_dict
            logger.info(f"Saved consolidated_ema to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model,
                StateDictType.LOCAL_STATE_DICT,
            ):
                opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved optimizer to {checkpoint_path}.")

            if dist.get_rank() == 0:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    print(step + 1, file=f)
            dist.barrier()
            logger.info(f"Saved training arguments to {checkpoint_path}.")

            gc.collect()
            torch.cuda.empty_cache()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT_Llama2_7B_patch2 with the
    # hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_data_on_disk", default=False, action="store_true")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--model", type=str, default="DiT_Llama2_7B_patch2")
    parser.add_argument("--patch_sizes", type=int, nargs="+", default=(2,))
    parser.add_argument("--f_patch_sizes", type=int, nargs="+", default=(2,))
    parser.add_argument("--max_steps", type=int, default=100_000, help="Number of training steps.")
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--master_port", type=int, default=18181)
    parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--no_compile", action="store_false", dest="compile")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument("--no_resume_step", action="store_true")
    parser.add_argument("--no_resume_opt", action="store_true")
    parser.add_argument("--no_resume_ema", action="store_true")
    parser.add_argument("--main_load_ema", action="store_true")
    parser.add_argument(
        "--init_from",
        type=str,
        help="Initialize the model weights from a checkpoint folder. "
        "Compared to --resume, this loads neither the optimizer states "
        "nor the data loader states.",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=2.0, help="Clip the L2 norm of the gradients to the given value."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--qk_norm",
        action="store_true",
    )
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--t_scale", type=float, default=1.0)
    parser.add_argument("--motion_scale", type=float, default=1.0)
    parser.add_argument("--vd_weight", type=float, default=0.0)
    parser.add_argument("--task_config", required=True, type=str)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    main(args)

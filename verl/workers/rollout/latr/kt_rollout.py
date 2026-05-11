# LATR KTRollout: Key-Token tree search rollout worker.
# Ported from https://github.com/starreeze/latr/blob/main/verl/workers/rollout/kt_rollout.py
# Adapted for verl 0.7.1's interface.
#
# This is a sync-mode rollout (like HFRollout). It does NOT use the async
# rollout server architecture. It takes the FSDP actor module directly and
# performs HF forward passes for tree search, then optionally hands off to a
# co-located vLLM engine for completion.
#
# Weight sync: Uses convert_weight_keys() + DTensor handling to sync from
# FSDP actor to the co-located vLLM engine. This is the same mechanism used
# by the original LATR code on verl 0.5.0.
#
# COMPATIBILITY NOTES (verl 0.7.1 vs LATR's verl 0.5.0):
# - BaseRollout.__init__ signature changed: now takes (config, model_config, device_mesh).
#   KTRollout bypasses this (like HFRollout does) with its own __init__.
# - convert_weight_keys() still exists in verl 0.7.1 at verl.utils.model.
# - vLLM sleep/wake API (enable_sleep_mode, sleep(level=2), wake_up()) requires
#   vLLM >= 0.10.0. The vllm_engine access path may differ between vLLM versions;
#   see the vllm_model property access below.

import copy
import gc
from typing import cast

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch import nn
from torch.amp.autocast_mode import autocast

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from vllm import LLM, SamplingParams

from verl.workers.rollout.latr.generation import KeyTokenGenConfig, KtModules, generate
from verl.workers.rollout.latr.kt_utils import BranchParamScheduler
from verl.workers.rollout.latr._utils import init_dataclass_from_dict
from verl import DataProto
from verl.utils.device import get_device_name
from verl.utils.model import convert_weight_keys
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _get_vllm_model(vllm_engine: LLM):
    """Extract the underlying model from vLLM engine.

    The internal path varies across vLLM versions. We try the common paths
    and fall back gracefully.
    """
    # vLLM >= 0.8 path
    try:
        return vllm_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
    except AttributeError:
        pass
    # vLLM >= 0.6 path
    try:
        return vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        pass
    raise RuntimeError(
        "Could not locate vLLM model runner. Check your vLLM version compatibility."
    )


class KTRollout(BaseRollout):
    """Key-Token tree search rollout.

    This rollout performs token-by-token tree search using HuggingFace forward
    passes on the FSDP actor module. When the tree search completes early
    (before max_new_tokens), it hands off to a co-located vLLM engine for
    completion of the remaining tokens.

    Args:
        module: The FSDP-wrapped actor module.
        path: Path to the model (for tokenizer and vLLM initialization).
        config: OmegaConf rollout config. Expected to contain:
            - kt: dict of KeyTokenGenConfig parameters
            - kt_mixed_engine: bool (default True), whether to use vLLM for completion
            - n: number of return sequences
            - response_length: max response length
            - temperature, top_k, top_p: sampling parameters
            - val_kwargs: validation sampling parameters
            - micro_batch_size: optional micro batch size
            Plus standard vLLM config fields (dtype, enforce_eager, etc.)
    """

    def __init__(self, module: nn.Module, path: str, config):
        # NOTE: We intentionally do NOT call super().__init__(config, model_config, device_mesh)
        # because KTRollout manages its own module reference and vLLM engine.
        # This matches HFRollout's pattern in verl 0.7.1.
        self.config = config
        self.module = module

        # Initialize KT config from the 'kt' section of rollout config
        kt_dict = dict(config.get("kt", {}))
        self.kt_config = init_dataclass_from_dict(KeyTokenGenConfig, kt_dict)
        self.kt_config.sync_gpus = True
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.kt_modules = KtModules()

        self.eos_token_id = cast(int, self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.eos_token_id
        self.pad_token_id = cast(int, self.tokenizer.pad_token_id)

        if config.get("kt_mixed_engine", True):
            self.kt_config.return_on_full = True
            self.vllm_engine = LLM(
                model=path,
                enable_sleep_mode=True,
                tensor_parallel_size=config.get("tensor_parallel_size", 1),
                distributed_executor_backend="external_launcher",
                dtype=config.get("dtype", "bfloat16"),
                enforce_eager=config.get("enforce_eager", True),
                gpu_memory_utilization=config.get("gpu_memory_utilization", 0.5),
                disable_custom_all_reduce=True,
                skip_tokenizer_init=False,
                disable_log_stats=config.get("disable_log_stats", True),
                enable_chunked_prefill=config.get("enable_chunked_prefill", True),
                enable_prefix_caching=True,
                trust_remote_code=True,
                seed=config.get("seed", 0),
                max_model_len=config.get("response_length", None),
            )
            self.vllm_model = _get_vllm_model(self.vllm_engine)
            self.vllm_engine.sleep(level=2)
        else:
            self.vllm_engine = None

        self.vllm_weights_updated = False

    def update_vllm_weights(self):
        """Sync FSDP actor weights to the co-located vLLM engine."""
        assert self.vllm_engine is not None
        rank = dist.get_rank()
        torch.cuda.empty_cache()
        self.vllm_engine.wake_up()

        params = convert_weight_keys(
            self.module.state_dict(),
            cast(PreTrainedModel, getattr(self.module, "_fsdp_wrapped_module", self.module)),
        )
        loaded_params = self.vllm_model.load_weights(
            (name, (param.to("cuda").full_tensor() if isinstance(param, DTensor) else param))
            for name, param in params.items()
        )
        print(f"rank {rank} vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        if self.kt_modules.sched is None:
            self.kt_modules.sched = init_dataclass_from_dict(BranchParamScheduler, self.kt_config.__dict__)
        self.kt_modules.sched.set_step(prompts.meta_info.get("global_steps", 0))
        validate = prompts.meta_info.get("validate", False)
        assert self.kt_modules.sched.mix_ratio is not None
        self.vllm_weights_updated = False

        if validate and self.vllm_engine is not None:
            batch_prompts = [prompts]
        else:
            batch_size = prompts.batch.batch_size[0]
            mbs = self.config.get("micro_batch_size", batch_size)
            num_chunks = max((batch_size + mbs - 1) // mbs, 1)
            batch_prompts = prompts.chunk(chunks=num_chunks)

        output = [self._generate_minibatch(p) for p in batch_prompts]

        if self.vllm_engine is not None:
            self.vllm_engine.sleep(level=2)

        output = DataProto.concat(output)
        self.module.train()
        gc.collect()
        torch.cuda.empty_cache()
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        self.module.eval()
        idx = prompts.batch["input_ids"]
        batch_size, prompt_length = idx.shape
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        is_validate = prompts.meta_info.get("validate", False)

        config = copy.copy(self.kt_config)
        if is_validate:
            config.fallback = True
            config.num_return_sequences = self.config.val_kwargs.n
            for k in ["temperature", "top_k", "top_p"]:
                setattr(config, k, getattr(self.config.val_kwargs, k))
        else:
            config.num_return_sequences = self.config.n
            for k in ["temperature", "top_k", "top_p"]:
                setattr(config, k, getattr(self.config, k))
        config.max_new_tokens = self.config.response_length

        assert self.kt_modules.sched is not None and self.kt_modules.sched.mix_ratio is not None
        if self.kt_modules.sched.mix_ratio > 0 and not is_validate and self.vllm_engine is not None:
            self.vllm_engine.sleep(level=2)

        with autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = generate(
                self.module,  # type: ignore
                self.kt_modules,
                idx,
                self.tokenizer,
                attention_mask,
                config=config,
            )

        assert self.kt_modules.sched is not None
        if not is_validate:
            self.kt_modules.sched.step_filter_params(output.suppress_ratio, output.empty_branch_ratio)

        kt_seqs = output.sequences
        generated_batch_size = kt_seqs.size(0)
        prompt = kt_seqs[:, :prompt_length]
        response = kt_seqs[:, prompt_length:]

        response_length = response.size(1)
        if self.vllm_engine is None:
            assert response_length == self.config.response_length

        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        output_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        if response_length == self.config.response_length:
            response_attention_mask = get_response_mask(
                response_id=response, eos_token=self.eos_token_id, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
            batch = TensorDict(
                {
                    "prompts": prompt,
                    "responses": response,
                    "input_ids": kt_seqs,
                    "attention_mask": attention_mask,
                    "position_ids": output_position_ids,
                },
                batch_size=generated_batch_size,
            )
            return DataProto(batch=batch)

        # Hand off to vLLM for completion
        assert self.vllm_engine is not None
        gc.collect()
        torch.cuda.empty_cache()

        self.vllm_engine.wake_up()
        if not self.vllm_weights_updated:
            self.update_vllm_weights()
            self.vllm_weights_updated = True

        target_vllm_response_length = self.config.response_length - response_length
        params = SamplingParams(
            n=1,
            logprobs=0,
            max_tokens=target_vllm_response_length,
            detokenize=False,
            top_k=config.top_k if config.top_k and config.top_k > 0 else -1,
            top_p=config.top_p,
            temperature=config.temperature,
        )
        vllm_inputs = [{"prompt_token_ids": _pre_process_inputs(self.pad_token_id, s)} for s in kt_seqs]
        rank = dist.get_rank()
        print(f"rank {rank} start vllm generation")
        outputs = self.vllm_engine.generate(prompts=vllm_inputs, sampling_params=params, use_tqdm=False)

        vllm_response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response_ids = output.outputs[sample_id].token_ids
                vllm_response.append(response_ids)

        vllm_response = pad_2d_list_to_length(
            vllm_response, self.pad_token_id, max_length=target_vllm_response_length
        ).to(idx.device)

        final_seqs = torch.cat([kt_seqs, vllm_response], dim=-1)
        response = final_seqs[:, prompt_length:]
        assert response.size(1) == self.config.response_length

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        output_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=prompts.meta_info.get("eos_token_id", self.eos_token_id),
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": final_seqs,
                "attention_mask": attention_mask,
                "position_ids": output_position_ids,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch)

    # -- Stubs for BaseRollout abstract methods --
    # KTRollout is a sync-mode rollout and does not use the async server
    # architecture. These methods are required by the ABC but are not called
    # in the sync code path.

    async def resume(self, tags: list[str]):
        pass

    async def update_weights(self, weights, **kwargs):
        pass

    async def release(self):
        pass

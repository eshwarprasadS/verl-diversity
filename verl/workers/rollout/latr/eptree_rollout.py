# LATR EptreeRollout: TreeRL's entropy-guided chain rollout worker.
# Ported from https://github.com/starreeze/latr/blob/main/verl/workers/rollout/eptree_rollout.py
# Adapted for verl 0.7.1's interface.
#
# This is a sync-mode rollout (like HFRollout). It uses a co-located vLLM
# engine exclusively (no HF forward passes). The EPTree algorithm generates
# diverse responses via entropy-guided tree expansion within vLLM.
#
# Weight sync: Same convert_weight_keys() + DTensor mechanism as KTRollout.

from typing import cast

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch import nn

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from vllm import LLM, SamplingParams

from verl.workers.rollout.latr.eptree import EPTree
from verl.workers.rollout.latr._utils import get_repeat_interleave
from verl import DataProto
from verl.utils.model import convert_weight_keys
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _get_vllm_model(vllm_engine: LLM):
    """Extract the underlying model from vLLM engine."""
    try:
        return vllm_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
    except AttributeError:
        pass
    try:
        return vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        pass
    raise RuntimeError(
        "Could not locate vLLM model runner. Check your vLLM version compatibility."
    )


class EptreeRollout(BaseRollout):
    """TreeRL's entropy-guided chain rollout.

    Uses a co-located vLLM engine to perform entropy-guided tree expansion,
    generating diverse responses by branching at high-entropy tokens.

    Args:
        module: The FSDP-wrapped actor module (used for weight sync to vLLM).
        path: Path to the model (for tokenizer and vLLM initialization).
        config: OmegaConf rollout config. Expected to contain:
            - n: number of return sequences per prompt
            - response_length: max response length
            - temperature, top_p: sampling parameters
            - val_kwargs: validation sampling parameters
            - mnlt: tuple (m, n, l, t) for EPTree parameters
            Plus standard vLLM config fields.
    """

    def __init__(self, module: nn.Module, path: str, config):
        # NOTE: Bypasses BaseRollout.__init__ like HFRollout/KTRollout.
        self.config = config
        self.module = module

        self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.eos_token_id = cast(int, self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.eos_token_id
        self.pad_token_id = cast(int, self.tokenizer.pad_token_id)

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

        mnlt = config.get("mnlt", (2, 4, 1, 1))
        if hasattr(mnlt, '__iter__') and not isinstance(mnlt, str):
            m, n, l, t = mnlt
        else:
            m, n, l, t = 2, 4, 1, 1
        self.eptree = EPTree(
            n_responses=config.n,
            tokenizer=self.tokenizer,
            vllm_engine=self.vllm_engine,
            max_new_tokens=config.response_length,
            temperature=config.temperature,
            top_p=config.get("top_p", 1.0),
            m=m,
            n=n,
            l=l,
            t=t,
        )

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        rank = dist.get_rank()
        torch.cuda.empty_cache()
        self.vllm_engine.wake_up()
        self.module.eval()

        # Sync FSDP weights to vLLM
        params = convert_weight_keys(
            self.module.state_dict(),
            cast(PreTrainedModel, getattr(self.module, "_fsdp_wrapped_module", self.module)),
        )
        loaded_params = self.vllm_model.load_weights(
            (name, (param.to("cuda").full_tensor() if isinstance(param, DTensor) else param))
            for name, param in params.items()
        )
        print(f"rank {rank} vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}")

        idx = prompts.batch["input_ids"]
        batch_size, prompt_length = idx.shape
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        if prompts.meta_info.get("validate", False):
            params = SamplingParams(
                n=1,
                logprobs=0,
                max_tokens=self.config.response_length,
                detokenize=False,
                top_k=self.config.val_kwargs.top_k if hasattr(self.config.val_kwargs, 'top_k') else -1,
                top_p=self.config.val_kwargs.top_p if hasattr(self.config.val_kwargs, 'top_p') else 1.0,
                temperature=self.config.val_kwargs.temperature if hasattr(self.config.val_kwargs, 'temperature') else 1.0,
            )
            vllm_inputs = [{"prompt_token_ids": _pre_process_inputs(self.pad_token_id, s)} for s in idx]
            print(f"rank {rank} start vllm generation")
            outputs = self.vllm_engine.generate(prompts=vllm_inputs, sampling_params=params, use_tqdm=False)

            vllm_response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    vllm_response.append(response_ids)
        else:
            print(f"rank {rank} start eptree generation")
            unique_idx = idx[:: get_repeat_interleave(idx)]
            vllm_inputs = [_pre_process_inputs(self.pad_token_id, s) for s in unique_idx]
            vllm_response = self.eptree.generate_sequences(vllm_inputs)

        vllm_response = pad_2d_list_to_length(
            vllm_response, self.pad_token_id, max_length=self.config.response_length
        ).to(idx.device)

        final_seqs = torch.cat([idx, vllm_response], dim=-1)
        response = final_seqs[:, prompt_length:]
        assert response.size(1) == self.config.response_length

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        output_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response,
            eos_token=prompts.meta_info.get("eos_token_id", self.eos_token_id),
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

        self.vllm_engine.sleep(level=2)
        self.module.train()
        torch.cuda.empty_cache()
        return DataProto(batch=batch)

    # -- Stubs for BaseRollout abstract methods --
    async def resume(self, tags: list[str]):
        pass

    async def update_weights(self, weights, **kwargs):
        pass

    async def release(self):
        pass

"""LLM backends behind one interface, and the policy that wraps them.

    backend.generate(messages_batch, max_new_tokens, temperature, top_p) -> list[str]

`HFBackend`   — transformers generate; used for GRPO rollouts because the
                training forward needs the same weights in the same process.
`VLLMBackend` — batched offline inference (eval, baselines); loads a LoRA
                adapter through vLLM's LoRA support.
Both are lazy: importing this module needs neither torch nor vllm.
"""

from __future__ import annotations

from .env import Observation
from .schema import Instance


class HFBackend:
    name = "hf"

    def __init__(self, model, tokenizer, *, max_prompt_tokens: int = 12000):
        self.model = model
        self.tok = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.last_batch: list[dict] = []       # token bookkeeping for the trainer

    @property
    def device(self):
        return next(self.model.parameters()).device

    def generate(self, messages_batch: list[list[dict]], *, max_new_tokens: int = 160,
                 temperature: float = 0.8, top_p: float = 0.95, sample: bool = True) -> list[str]:
        import torch

        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in messages_batch]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_prompt_tokens).to(self.device)
        kw = dict(max_new_tokens=max_new_tokens, do_sample=sample, pad_token_id=self.tok.pad_token_id)
        if sample:
            kw.update(temperature=temperature, top_p=top_p)
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        prompt_len = enc["input_ids"].shape[1]
        self.last_batch = []
        results = []
        for i in range(out.shape[0]):
            comp = out[i, prompt_len:]
            # strip padding / everything after the first eos
            ids = comp.tolist()
            if self.tok.eos_token_id in ids:
                ids = ids[: ids.index(self.tok.eos_token_id) + 1]
            ids = [t for t in ids if t != self.tok.pad_token_id or t == self.tok.eos_token_id]
            mask = enc["attention_mask"][i].bool()
            self.last_batch.append({"prompt_ids": enc["input_ids"][i][mask].tolist(), "completion_ids": ids})
            results.append(self.tok.decode(ids, skip_special_tokens=True))
        return results


class VLLMBackend:
    name = "vllm"

    def __init__(self, model: str, *, adapter: str | None = None, tensor_parallel_size: int = 1,
                 max_model_len: int = 16384, gpu_memory_utilization: float = 0.9, max_lora_rank: int = 64,
                 seed: int = 0):
        from vllm import LLM

        self.llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, dtype="bfloat16",
                       enable_lora=bool(adapter), max_lora_rank=max_lora_rank, seed=seed)
        self.tok = self.llm.get_tokenizer()
        self.lora = None
        if adapter:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest("policy", 1, adapter)

    def generate(self, messages_batch, *, max_new_tokens=160, temperature=0.8, top_p=0.95, sample=True):
        from vllm import SamplingParams

        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in messages_batch]
        sp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature if sample else 0.0,
                            top_p=top_p if sample else 1.0)
        outs = self.llm.generate(texts, sp, lora_request=self.lora, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


class LLMPolicy:
    """Stateless policy: each observation is rendered fresh and answered in one
    generate call. `collect` keeps the (prompt_ids, completion_ids) of the last
    call for GRPO."""

    privileged = False

    def __init__(self, backend, *, max_new_tokens: int = 160, temperature: float = 0.8, top_p: float = 0.95,
                 sample: bool = True, name: str | None = None):
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.sample = sample
        self.name = name or f"llm-{backend.name}"

    def act_batch(self, obs_list: list[Observation], insts: list[Instance]) -> list[str]:
        msgs = [o.to_messages() for o in obs_list]
        out = self.backend.generate(msgs, max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                                    top_p=self.top_p, sample=self.sample)
        self.last_step_extras = list(getattr(self.backend, "last_batch", []) or [])
        return out


def build_backend(kind: str, model: str, adapter: str | None = None, **kw):
    if kind == "vllm":
        return VLLMBackend(model, adapter=adapter, **{k: v for k, v in kw.items()
                                                     if k in ("tensor_parallel_size", "max_model_len",
                                                              "gpu_memory_utilization", "seed")})
    if kind == "hf":
        from .common import load_policy, load_tokenizer

        return HFBackend(load_policy(model, adapter), load_tokenizer(model))
    raise ValueError(f"unknown backend {kind!r}")

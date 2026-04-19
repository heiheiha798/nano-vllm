import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.llm_engine import LLM
from engine.sequence import SamplingParams


MODEL_PATH = "/data/pretrained_models/Qwen3-0.6B"
PROMPT_TEXT = (
    "Prefill decode paged attention runtime scheduler kv cache graph batching "
    "throughput comparison benchmark."
)
INPUT_TOKENS = 10
DECODE_TOKENS = 100
TEMPERATURE = 0.8
WARMUP_STEPS = 5
SEED = 0


def sync_cuda() -> None:
    torch.cuda.synchronize()


def sample_next_token(logits: torch.Tensor, temperature: float) -> int:
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def build_prompt_token_ids(tokenizer: AutoTokenizer) -> list[int]:
    token_ids = tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    if len(token_ids) < INPUT_TOKENS:
        raise RuntimeError(
            f"Prompt only produced {len(token_ids)} tokens, but benchmark requires {INPUT_TOKENS}."
        )
    return token_ids[:INPUT_TOKENS]


def run_hf_eager(
    model: AutoModelForCausalLM,
    prompt_token_ids: list[int],
) -> tuple[float, list[int]]:
    input_ids = torch.tensor([prompt_token_ids], device="cuda", dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

    past_key_values = outputs.past_key_values
    next_token_id = sample_next_token(outputs.logits[:, -1, :], TEMPERATURE)
    generated_ids = [next_token_id]
    next_token = torch.tensor([[next_token_id]], device="cuda", dtype=input_ids.dtype)
    running_attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), device="cuda", dtype=attention_mask.dtype)],
        dim=1,
    )

    warmup_token = next_token.clone()
    warmup_mask = running_attention_mask.clone()
    warmup_cache = past_key_values
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            warmup_outputs = model(
                input_ids=warmup_token,
                attention_mask=warmup_mask,
                past_key_values=warmup_cache,
                use_cache=True,
                return_dict=True,
            )
        warmup_cache = warmup_outputs.past_key_values
        sampled = sample_next_token(warmup_outputs.logits[:, -1, :], TEMPERATURE)
        warmup_token = torch.tensor([[sampled]], device="cuda", dtype=input_ids.dtype)
        warmup_mask = torch.cat(
            [warmup_mask, torch.ones((1, 1), device="cuda", dtype=warmup_mask.dtype)],
            dim=1,
        )

    sync_cuda()
    start = time.perf_counter()
    for _ in range(1, DECODE_TOKENS):
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=running_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values
        next_token_id = sample_next_token(outputs.logits[:, -1, :], TEMPERATURE)
        generated_ids.append(next_token_id)
        next_token = torch.tensor([[next_token_id]], device="cuda", dtype=input_ids.dtype)
        running_attention_mask = torch.cat(
            [
                running_attention_mask,
                torch.ones((1, 1), device="cuda", dtype=running_attention_mask.dtype),
            ],
            dim=1,
        )
    sync_cuda()
    end = time.perf_counter()

    return end - start, generated_ids


def run_nanovllm_fa2_eager(prompt_token_ids: list[int]) -> tuple[float, list[int]]:
    llm = LLM(MODEL_PATH, enforce_eager=True)
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=DECODE_TOKENS)
    warmup_params = SamplingParams(temperature=TEMPERATURE, max_tokens=WARMUP_STEPS)
    llm.add_request(prompt_token_ids, warmup_params)
    while not llm.is_finished():
        llm.step()

    llm.add_request(prompt_token_ids, sampling_params)

    outputs, num_tokens = llm.step()
    if num_tokens <= 0:
        raise RuntimeError("nano-vllm first step should be prefill.")

    seq_id = 0
    generated_ids: list[int] = []
    if outputs:
        seq_id, token_ids = outputs[0]
        generated_ids = token_ids
    else:
        seq = llm.scheduler.running[0]
        seq_id = seq.seq_id
        generated_ids = seq.completion_token_ids

    sync_cuda()
    start = time.perf_counter()
    while not llm.is_finished():
        outputs, _ = llm.step()
        if outputs:
            out_seq_id, token_ids = outputs[0]
            if out_seq_id == seq_id:
                generated_ids = token_ids
    sync_cuda()
    end = time.perf_counter()

    return end - start, generated_ids


def print_result(name: str, elapsed: float, token_ids: list[int]) -> None:
    measured_decode_tokens = max(len(token_ids) - 1, 0)
    tok_per_s = measured_decode_tokens / elapsed
    print(f"{name}: {tok_per_s:.2f} tok/s")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    prompt_token_ids = build_prompt_token_ids(tokenizer)

    torch.manual_seed(SEED)
    model_eager = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).to("cuda")
    model_eager.eval()

    hf_no_fa2_elapsed, hf_no_fa2_ids = run_hf_eager(model_eager, prompt_token_ids)
    del model_eager
    torch.cuda.empty_cache()

    torch.manual_seed(SEED)
    model_fa2 = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model_fa2.eval()

    hf_fa2_elapsed, hf_fa2_ids = run_hf_eager(model_fa2, prompt_token_ids)
    del model_fa2
    torch.cuda.empty_cache()

    torch.manual_seed(SEED)
    nvllm_elapsed, nvllm_ids = run_nanovllm_fa2_eager(prompt_token_ids)

    print_result("HF eager no FA2", hf_no_fa2_elapsed, hf_no_fa2_ids)
    print_result("HF eager + FA2", hf_fa2_elapsed, hf_fa2_ids)
    print_result("nano-vllm eager + FA2", nvllm_elapsed, nvllm_ids)


if __name__ == "__main__":
    main()

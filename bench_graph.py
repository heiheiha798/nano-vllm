import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

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


def prefill_static_cache(
    model: AutoModelForCausalLM,
    prompt_token_ids: list[int],
    max_cache_len: int,
) -> tuple[StaticCache, int, torch.dtype, torch.dtype]:
    input_ids = torch.tensor([prompt_token_ids], device="cuda", dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]
    cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    cache_position = torch.arange(prompt_len, device="cuda", dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )

    first_decode_token_id = sample_next_token(outputs.logits[:, -1, :], TEMPERATURE)
    return cache, first_decode_token_id, input_ids.dtype, attention_mask.dtype


def fill_decode_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache_position: torch.Tensor,
    token_id: int,
    total_len: int,
) -> None:
    input_ids[0, 0] = token_id
    attention_mask.zero_()
    attention_mask[:, :total_len] = 1
    cache_position[0] = total_len - 1


def run_hf_sdpa_graph(
    model: AutoModelForCausalLM,
    prompt_token_ids: list[int],
) -> tuple[float, list[int]]:
    prompt_len = len(prompt_token_ids)
    max_cache_len = prompt_len + DECODE_TOKENS
    cache, first_decode_token_id, input_dtype, mask_dtype = prefill_static_cache(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_cache_len=max_cache_len,
    )

    static_input_ids = torch.zeros((1, 1), device="cuda", dtype=input_dtype)
    static_attention_mask = torch.zeros((1, max_cache_len), device="cuda", dtype=mask_dtype)
    static_cache_position = torch.zeros((1,), device="cuda", dtype=torch.long)

    fill_decode_inputs(
        input_ids=static_input_ids,
        attention_mask=static_attention_mask,
        cache_position=static_cache_position,
        token_id=first_decode_token_id,
        total_len=prompt_len + 1,
    )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        with torch.no_grad():
            for _ in range(WARMUP_STEPS):
                graph_outputs = model(
                    input_ids=static_input_ids,
                    attention_mask=static_attention_mask,
                    past_key_values=cache,
                    cache_position=static_cache_position,
                    use_cache=True,
                    return_dict=True,
                )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    sync_cuda()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            graph_outputs = model(
                input_ids=static_input_ids,
                attention_mask=static_attention_mask,
                past_key_values=cache,
                cache_position=static_cache_position,
                use_cache=True,
                return_dict=True,
            )

    generated_ids = [first_decode_token_id]
    token_id = first_decode_token_id
    sync_cuda()
    start = time.perf_counter()
    for step in range(1, DECODE_TOKENS):
        total_len = prompt_len + step + 1
        fill_decode_inputs(
            input_ids=static_input_ids,
            attention_mask=static_attention_mask,
            cache_position=static_cache_position,
            token_id=token_id,
            total_len=total_len,
        )
        graph.replay()
        token_id = sample_next_token(graph_outputs.logits[:, -1, :], TEMPERATURE)
        generated_ids.append(token_id)
    sync_cuda()
    end = time.perf_counter()

    return end - start, generated_ids


def run_nanovllm_fa2_graph(prompt_token_ids: list[int]) -> tuple[float, list[int]]:
    llm = LLM(MODEL_PATH, enforce_eager=False)
    warmup_params = SamplingParams(temperature=TEMPERATURE, max_tokens=WARMUP_STEPS)
    llm.add_request(prompt_token_ids, warmup_params)
    while not llm.is_finished():
        llm.step()

    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=DECODE_TOKENS)
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
    model_sdpa = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model_sdpa.eval()

    hf_elapsed, hf_ids = run_hf_sdpa_graph(model_sdpa, prompt_token_ids)
    del model_sdpa
    torch.cuda.empty_cache()

    torch.manual_seed(SEED)
    nvllm_elapsed, nvllm_ids = run_nanovllm_fa2_graph(prompt_token_ids)

    print_result("HF + SDPA + graph", hf_elapsed, hf_ids)
    print_result("nano-vllm + FA2 + graph", nvllm_elapsed, nvllm_ids)


if __name__ == "__main__":
    main()

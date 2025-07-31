from typing import Optional, Union, Any, AsyncGenerator
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncEngineArgs, PromptType, SamplingParams, RequestOutput
import time
import os
import asyncio

model = "meta-llama/Llama-3.2-1B"
output_token_count = 64

hf_token = os.environ.get("HF_TOKEN")

# Check environment with:
# python .venv/lib64/python3.9/site-packages/vllm/collect_env.py

engine_args = AsyncEngineArgs(model=model, hf_token=hf_token)
llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

async def handle_llm_output(
    generator: AsyncGenerator[RequestOutput, None],
    start_time: float,
    request_id: Optional[str],
) -> None: # AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
    response_prompt_count: Optional[int] = None
    response_output_count: Optional[int] = None
    iter_count = 0
    start_time = time.time()
    iter_time = start_time
    first_iter_time: Optional[float] = None
    last_iter_time: Optional[float] = None

    # yield StreamingTextResponse(
    #     type_="start",
    #     value="",
    #     start_time=start_time,
    #     first_iter_time=None,
    #     iter_count=iter_count,
    #     delta="",
    #     time=start_time,
    #     request_id=request_id,
    # )

    async for output in generator:
        iter_count += 1
        if output.finished:
            print("Got finished output:", output)
            return # yield
        else:
            print("Got unfinished output:", output)
            # yield StreamingTextResponse(
            #     type_="iter",
            #     value=response_value,
            #     iter_count=iter_count,
            #     start_time=start_time,
            #     first_iter_time=first_iter_time,
            #     delta=delta,
            #     time=iter_time,
            #     request_id=request_id,
            # )

params = SamplingParams(
    max_tokens = output_token_count,
    # Set min token count?
)

request_id = "request_" + str(int(time.time()))
start_time = time.time()
llm_generator = llm_engine.generate("write a summary of the purpose of HTML", params, request_id)
asyncio.run(handle_llm_output(llm_generator, start_time, request_id))

from typing import Optional, Union, Any, AsyncGenerator
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncEngineArgs, PromptType, SamplingParams, RequestOutput
import time
import os
import asyncio

# Note: The Llama models don't appear to support chat.
#model = "HuggingFaceH4/zephyr-7b-beta"
model = "ibm-granite/granite-3.3-2b-instruct"
output_token_count = 512

hf_token = os.environ.get("HF_TOKEN")

# Check environment with:
# python .venv/lib64/python3.9/site-packages/vllm/collect_env.py

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
            print("\nGot finished output:", output)
            return # yield
        else:
            print("\nGot unfinished output:", output)
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

async def main():

    engine_args = AsyncEngineArgs(model=model, hf_token=hf_token)
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    params = SamplingParams(
        max_tokens=output_token_count,
        temperature=0.9, top_p=0.95,
        # Set min token count?
    )

    request_id = "request_" + str(int(time.time()))
    start_time = time.time()
    tokenizer = await llm_engine.get_tokenizer()
    prompt = "How can I write a function in Java?"
    conversation =  [

        {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
        {"role": "user", "content": prompt}
    ]
    llm_input = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


    llm_generator = llm_engine.generate(llm_input, params, request_id)
    await handle_llm_output(llm_generator, start_time, request_id)


if __name__ == "__main__":
    asyncio.run(main())

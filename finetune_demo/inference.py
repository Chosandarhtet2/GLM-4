from pathlib import Path
from typing import Annotated, Union

import typer
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

app = typer.Typer(pretty_exceptions_show_locals=False)


def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    model_dir = Path(model_dir).expanduser().resolve()
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code, encode_special_tokens=True, use_fast=False
    )
    return model, tokenizer


@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='')],
):
    messages = [
        {"role":"system",
        "content":'仅从这段input文字，按这些标准，以一个JSON dictionary的结构，每个key是一个问题，把该key的<answer>替换为结构化提取的字符串答案，未提及或未知的答案以null来表示。直接展示JSON结构，不要其他说明：{"用药频次？只输出这些选项的其中之一：qd, bid或q12h。": "<answer/>","次剂量？输出数值。": "<answer/>","日剂量？输出数值。": "<answer/>","给药途径？只输出这些选项的其中之一：口服, 持续静脉泵入, 静推或动脉给药。": "<answer/>","泵入速度？输出数值。": "<answer/>"}'
        },
        {"role":"user",
        "content": '<input>替格瑞洛片 62mg 口服 3/日</input>'},
        # {"role":"assistant",
        #  "content": '{"用药频次？只输出这些选项的其中之一：qd, bid或q12h。": "bid","次剂量？输出数值。": 45,"日剂量？输出数值。": 90,"给药途径？只输出这些选项的其中之一：口服, 持续静脉泵入, 静推或动脉给药。": "口服","泵入速度？输出数值。": null}'
        #  }
        ]
    model, tokenizer = load_model_and_tokenizer(model_dir)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)
    generate_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": 1024,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8,
        "repetition_penalty": 1.2,
        "eos_token_id": model.config.eos_token_id,
    }
    outputs = model.generate(**generate_kwargs)
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()
    print("=========")
    print(response)


if __name__ == '__main__':
    app()

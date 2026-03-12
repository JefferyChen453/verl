# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def get_gsm8k_data():
    # prepare test dataset
    local_folder = os.path.expanduser("~/verl-data/gsm8k/")
    local_path = os.path.join(local_folder, "train.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def test_rl_dataset():
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

    tokenizer = hf_tokenizer("deepseek-ai/deepseek-coder-1.3b-instruct")
    local_path = get_gsm8k_data()
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 256,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 2,
        }
    )
    dataset = RLHFDataset(data_files=local_path, tokenizer=tokenizer, config=config)

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

    a = next(iter(dataloader))

    from verl import DataProto

    tensors = {}
    non_tensors = {}

    for key, val in a.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    assert "input_ids" in data_proto.batch

    data = dataset[0]["input_ids"]
    output = tokenizer.batch_decode([data])[0]
    print(f"type: type{output}")
    print(f"\n\noutput: {output}")


def test_rl_dataset_with_max_samples():
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset

    tokenizer = hf_tokenizer("deepseek-ai/deepseek-coder-1.3b-instruct")
    local_path = get_gsm8k_data()
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 256,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 2,
            "max_samples": 5,
        }
    )
    dataset = RLHFDataset(data_files=local_path, tokenizer=tokenizer, config=config, max_samples=5)
    assert len(dataset) == 5


def test_image_rl_data():
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

    tokenizer = hf_tokenizer("Qwen/Qwen2-VL-2B-Instruct")
    processor = hf_processor("Qwen/Qwen2-VL-2B-Instruct")
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 1024,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": None,  # num_workers=1 hang in ci
        }
    )
    dataset = RLHFDataset(
        data_files=os.path.expanduser("~/data/geo3k/train.parquet"),
        tokenizer=tokenizer,
        config=config,
        processor=processor,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

    a = next(iter(dataloader))

    from verl import DataProto

    tensors = {}
    non_tensors = {}

    for key, val in a.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    assert "multi_modal_data" in data_proto.non_tensor_batch, data_proto
    assert "multi_modal_inputs" in data_proto.non_tensor_batch, data_proto

    data = dataset[0]["input_ids"]
    output = tokenizer.batch_decode([data])[0]
    print(f"type: type{output}")
    print(f"\n\noutput: {output}")


def test_rl_dataset_with_preformatted_string_prompt(tmp_path):
    import pandas as pd

    from tests.special_e2e.envs.digit_completion.tokenizer import CharTokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset

    chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    tokenizer = CharTokenizer(
        characters=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ: ?!\n"),
        model_max_length=256,
        chat_template=chat_template,
    )

    messages = [{"role": "user", "content": "Hello?"}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    parquet_path = tmp_path / "preformatted_prompt.parquet"
    pd.DataFrame([{"prompt": formatted_prompt, "data_source": "unit_test"}]).to_parquet(parquet_path)

    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 256,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": None,
            "return_full_prompt": True,
        }
    )
    dataset = RLHFDataset(data_files=str(parquet_path), tokenizer=tokenizer, config=config)

    assert len(dataset) == 1
    item = dataset[0]

    non_pad_input_ids = item["input_ids"][item["attention_mask"].bool()]
    decoded_prompt = tokenizer.decode(non_pad_input_ids, skip_special_tokens=False)

    assert item["full_prompts"] == formatted_prompt
    assert decoded_prompt == formatted_prompt

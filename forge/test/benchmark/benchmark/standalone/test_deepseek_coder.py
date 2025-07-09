# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from forge._C.runtime import (
    Binary,
    Tensor as CTensor,
    ModelState,
    ProgramType,
)
import forge._C.runtime as rt
from forge.tensor import to_pt_tensors, cast_unsupported_torch_dtype

from test.models.models_utils import generate_no_cache, pad_inputs
from test.models.pytorch.multimodal.deepseek_coder.utils.model_utils import download_model_and_tokenizer


def generate_no_cache(max_new_tokens, runtime_model_state, inputs, seq_len, tokenizer):
    current_pos = seq_len

    for _ in range(max_new_tokens):
        inputs = [*to_pt_tensors(inputs)]
        inputs = [cast_unsupported_torch_dtype(input_tensor) for input_tensor in inputs]
        ctensors = [CTensor(input) for input in inputs]

        runtime_model_state.run_program(ProgramType.Forward, ctensors)
        all_outputs = runtime_model_state.get_outputs(ProgramType.Forward)
        logits = [i.to_torch() for i in all_outputs]

        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        elif not isinstance(logits, torch.Tensor):
            raise TypeError(f"Expected logits to be a list or tuple or torch.Tensor, but got {type(logits)}")

        next_token_logits = logits[:, current_pos - 1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        inputs[0][:, current_pos] = next_token_id
        current_pos += 1

        v_tokens = inputs[0][:, seq_len:current_pos].view(-1).tolist()
        token = tokenizer.decode(v_tokens, skip_special_tokens=True)
        print("Token-> ", token)

    valid_tokens = inputs[0][:, seq_len:current_pos].view(-1).tolist()
    answer = tokenizer.decode(valid_tokens, skip_special_tokens=True)
    return answer


# Add model_path as a fixture from command-line option
@pytest.fixture
def model_path(request):
    return request.config.getoption("--model-path")


def pytest_addoption(parser):
    parser.addoption(
        "--model-path", action="store", default=None, help="Path to compiled model and program state files"
    )


@pytest.mark.nightly
def test_runner(model_artifact_path):
    if model_artifact_path is None:
        raise ValueError("Please provide --model-artificat-path when running pytest")

    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    model, tokenizer, inputs = download_model_and_tokenizer(model_name)
    padded_inputs, seq_len = pad_inputs(inputs)

    compiled_binary = Binary.load_from_file(f"{model_artifact_path}/model_fb.ttnn")
    runtime_model_state = ModelState(compiled_binary)

    pstate = rt.load_program_state_from_bin(f"{model_artifact_path}/programstate.bin")
    runtime_model_state.init_program_state(pstate)

    generated_text = generate_no_cache(
        max_new_tokens=512,
        runtime_model_state=runtime_model_state,
        inputs=padded_inputs,
        seq_len=seq_len,
        tokenizer=tokenizer,
    )

    print(generated_text)

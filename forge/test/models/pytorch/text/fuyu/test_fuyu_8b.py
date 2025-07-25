# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from transformers import (
    AutoTokenizer,
    FuyuConfig,
    FuyuForCausalLM,
    FuyuImageProcessor,
    FuyuProcessor,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.text.fuyu.model_utils.model import (
    FuyuModelWrapper,
    generate_fuyu_embedding,
)


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("adept/fuyu-8b"),
    ],
)
def test_fuyu8b(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model=ModelArch.FUYU, variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )
    pytest.xfail(reason="Requires multi-chip support")

    config = FuyuConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config_dict["text_config"]["num_hidden_layers"] = 1
    config = FuyuConfig(**config_dict)

    # Load post-processing modules  (run on CPU)
    tokenizer = AutoTokenizer.from_pretrained(variant)
    image_processor = FuyuImageProcessor()
    processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # Create Forge module from PyTorch model
    fuyu_model = FuyuForCausalLM.from_pretrained(variant, config=config)
    framework_model = FuyuModelWrapper(fuyu_model)
    framework_model.eval()

    # Prepare inputs
    text_prompt = "Generate a coco-style caption.\n"

    input_image = get_file("https://huggingface.co/adept-hf-collab/fuyu-8b/resolve/main/bus.png")
    image_pil = Image.open(str(input_image))

    model_inputs = processor(text=text_prompt, images=[image_pil], device="cpu", return_tensor="pt")
    inputs_embeds = generate_fuyu_embedding(
        fuyu_model, model_inputs["input_ids"], model_inputs["image_patches"][0], model_inputs["image_patches_indices"]
    )
    inputs_embeds = inputs_embeds.clone().detach()

    inputs = [inputs_embeds]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)

    os.remove("bus.png")

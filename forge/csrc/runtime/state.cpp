// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "runtime/state.hpp"

#include <utils/logger.hpp>

#include "tt/runtime/runtime.h"

namespace tt
{

std::ostream& operator<<(std::ostream& os, ProgramType program_type)
{
    switch (program_type)
    {
        case ProgramType::Forward: os << "Forward"; break;
        case ProgramType::Backward: os << "Backward"; break;
        case ProgramType::Optimizer: os << "Optimizer"; break;
        default: os << "Unknown"; break;
    }
    return os;
}

ProgramState create_program_state(
    ProgramType program_type, const TensorPool& tensor_pool, std::vector<std::string> persistent_input_names)
{
    std::vector<tt::Tensor> persistent_inputs;
    persistent_inputs.reserve(persistent_input_names.size());

    for (auto& name : persistent_input_names)
    {
        auto tensor = tensor_pool.get_tensor(name);
        persistent_inputs.emplace_back(tensor);
    }
    return ProgramState{program_type, persistent_inputs, {}};
}

void save_program_state_as_bin(const ProgramState& state, const std::string& path)
{
    torch::serialize::OutputArchive archive;

    archive.write("program_type", torch::tensor(static_cast<int64_t>(state.program_type)));

    // Save persistent inputs
    torch::serialize::OutputArchive inputs_archive;
    inputs_archive.write("count", torch::tensor(static_cast<int64_t>(state.persistent_inputs.size())));

    for (size_t i = 0; i < state.persistent_inputs.size(); ++i)
    {
        state.persistent_inputs[i].to_torch().sizes() << "\n";
        inputs_archive.write(std::to_string(i), state.persistent_inputs[i].to_torch());
    }

    archive.write("persistent_inputs", inputs_archive);

    // Save outputs
    torch::serialize::OutputArchive outputs_archive;
    outputs_archive.write("count", torch::tensor(static_cast<int64_t>(state.outputs.size())));

    for (size_t i = 0; i < state.outputs.size(); ++i)
    {
        outputs_archive.write(std::to_string(i), state.outputs[i].to_torch());
    }

    archive.write("outputs", outputs_archive);

    // Save the archive to file
    archive.save_to(path);
}

ProgramState load_program_state_from_bin(const std::string& path)
{
    torch::serialize::InputArchive archive;
    archive.load_from(path);

    // Load program type
    torch::Tensor pt_tensor;
    archive.read("program_type", pt_tensor);
    ProgramType program_type = static_cast<ProgramType>(pt_tensor.item<int64_t>());

    // Load persistent inputs
    torch::serialize::InputArchive inputs_archive;
    archive.read("persistent_inputs", inputs_archive);

    torch::Tensor input_count_tensor;
    inputs_archive.read("count", input_count_tensor);
    int64_t input_count = input_count_tensor.item<int64_t>();

    std::vector<Tensor> persistent_inputs;
    for (int64_t i = 0; i < input_count; ++i)
    {
        torch::Tensor t;
        inputs_archive.read(std::to_string(i), t);
        persistent_inputs.emplace_back(Tensor(t));
    }

    // Load outputs
    torch::serialize::InputArchive outputs_archive;
    archive.read("outputs", outputs_archive);

    torch::Tensor output_count_tensor;
    outputs_archive.read("count", output_count_tensor);
    int64_t output_count = output_count_tensor.item<int64_t>();

    std::vector<tt::Tensor> outputs;
    for (int64_t i = 0; i < output_count; ++i)
    {
        torch::Tensor t;
        outputs_archive.read(std::to_string(i), t);
        outputs.emplace_back(tt::Tensor(t));
    }

    return ProgramState{program_type, persistent_inputs, outputs};
}

void ModelState::run_program(ProgramType program_type, std::vector<tt::Tensor> act_inputs)
{
    // ISSUE(#1346): So far, the device_id is hardcoded to 0 - make it user-configurable.
    constexpr size_t device_id = 0;
    size_t pg_id = program_idx(program_type);
    std::optional<ProgramState>& opt_program_state = program_states[pg_id];

    TT_ASSERT(opt_program_state.has_value(), "Program state for {} not initialized", program_type);

    if (!TTSystem::get_system().devices[device_id]->is_open())
    {
        TTSystem::get_system().devices[device_id]->open_device();
    }

    auto& program_state = opt_program_state.value();

    // Clear the outputs from the previous run.
    program_state.outputs.clear();

    std::vector<tt::Tensor> inputs;
    inputs.reserve(act_inputs.size() + program_state.persistent_inputs.size());

    // Push activation inputs to the device if they are not already there.
    // NOTE: there is an ordering requirement for the activation inputs and the persistent inputs, i.e.
    // in the input list (`inputs` vector here), the activation inputs come first. Unfortunately, there isn't any
    // mechanism which enforces this, yet. It's an informal contract between the compiler and the runtime.
    size_t input_idx = 0;
    for (auto tensor : act_inputs)
    {
        if (!tensor.on_device())
        {
            auto layout = tt::runtime::getLayout(binary, pg_id, input_idx++);
            tensor.to_device(device_id, layout);
        }

        inputs.emplace_back(tensor);
    }

    // Push persistent inputs to the device if they are not already there.
    for (auto& persistent_input : program_state.persistent_inputs)
    {
        size_t curr_input_id = input_idx++;
        if (!persistent_input.on_device())
        {
            auto layout = tt::runtime::getLayout(binary, pg_id, curr_input_id);
            persistent_input.to_device(device_id, layout);
        }

        inputs.emplace_back(persistent_input);
    }

    program_state.outputs = ::tt::run_program(binary, pg_id, inputs);
}

};  // namespace tt

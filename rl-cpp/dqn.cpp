//
// Created by saminda on 1/26/20.
//

#include "dqn.h"
#include "common.h"
#include "gym.h"

#include <torch/torch.h>

namespace deepx {

DqnTrainer::DqnTrainer(std::shared_ptr<Client> client,
                       const EnvConfig &config,
                       int64_t buffer_size,
                       int64_t batch_size,
                       double epsilon_decay) : batch_size_(batch_size), epsilon_decay_(epsilon_decay),
                                               client_(client),
                                               config_(config),
                                               buffer_{buffer_size},
                                               rand_generator_(std::random_device{}()),
                                               device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {

  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
  } else {
    std::cout << "Training on CPU." << std::endl;
  }
}

void DqnTrainer::define_models_and_optim() {
  current_model_ = std::make_shared<Dqn>(config_.observation_space().box().shape()[0],
                                         config_.action_space().discrete().n());
  target_model_ = std::make_shared<Dqn>(config_.observation_space().box().shape()[0],
                                        config_.action_space().discrete().n());
  current_model_->to(device_);
  target_model_->to(device_);

  opt_ = std::make_shared<torch::optim::Adam>(current_model_->parameters(), torch::optim::AdamOptions{1e-3});

  load_from_state_dict(target_model_, current_model_);
}

double DqnTrainer::epsilon_by_frame(int64_t frame_id) {
  return epsilon_final_ + (epsilon_start_ - epsilon_final_) * exp(-1. * frame_id / epsilon_decay_);
}

torch::Tensor DqnTrainer::compute_td_loss(int64_t batch_size, float gamma) {
  std::vector<ReplayBuffer::Experience> batch = buffer_.sample(batch_size);

  std::vector<torch::Tensor> state;
  std::vector<torch::Tensor> action;
  std::vector<torch::Tensor> reward;
  std::vector<torch::Tensor> next_state;
  std::vector<torch::Tensor> done;

  for (const auto &i : batch) {
    const auto&[b_state, b_action, b_reward, b_next_state, b_done] = i;
    state.push_back(b_state);
    action.push_back(b_action);
    reward.push_back(b_reward);
    next_state.push_back(b_next_state);
    done.push_back(b_done);
  }

  auto state_tensor = torch::cat(state, 0).to(device_);
  auto action_tensor = torch::cat(action, 0).to(device_);
  auto reward_tensor = torch::cat(reward, 0).to(device_);
  auto next_state_tensor = torch::cat(next_state, 0).to(device_);
  auto done_tensor = torch::cat(done, 0).to(device_);

  torch::Tensor q_values = current_model_->forward(state_tensor);
  torch::Tensor next_q_values = current_model_->forward(next_state_tensor);
  torch::Tensor next_target_q_values = target_model_->forward(next_state_tensor);

//  action_tensor = action_tensor.to(torch::kInt64);

  torch::Tensor q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1);
  torch::Tensor maximum = std::get<1>(next_q_values.max(1));
  torch::Tensor next_q_value = next_target_q_values.gather(1, maximum.unsqueeze(1)).squeeze(1);
  torch::Tensor expected_q_value = reward_tensor + gamma * next_q_value * (1 - done_tensor);
  torch::Tensor loss = torch::mse_loss(q_value, expected_q_value.detach());

  opt_->zero_grad();
  loss.backward();
  opt_->step();

  return loss;
}

torch::Tensor DqnTrainer::get_state_tensor(const State &state) const {
  return torch::tensor(std::vector<float>{state.obs().begin(), state.obs().end()}).unsqueeze(0);
}

void DqnTrainer::train(int64_t num_frames) {
  current_model_->train();
  std::vector<float> losses, all_rewards;
  float episode_reward = 0;
  std::uniform_int_distribution<int> randint(0, config_.action_space().discrete().n() - 1);
  std::uniform_real_distribution<float> rand(0.0f, 1.0f);

  State state = client_->Reset(config_);
  torch::Tensor state_tensor = get_state_tensor(state);

  for (int64_t frame_id = 0; frame_id < num_frames + 1; ++frame_id) {
    double epsilon = epsilon_by_frame(frame_id);
    torch::Tensor action_tensor = current_model_->act(state_tensor.to(device_));
    if (rand(rand_generator_) < epsilon) {
      action_tensor.fill_(randint(rand_generator_));
    }
    Step step = client_->Step(config_, action_tensor);

    torch::Tensor next_state_tensor = get_state_tensor(step.state());
    torch::Tensor reward_tensor = torch::tensor(step.reward());
    torch::Tensor done_tensor = torch::tensor(static_cast<float>(step.done()));
    buffer_.push(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor);

    state_tensor = next_state_tensor;
    episode_reward += step.reward();

    if (step.done()) {
      state = client_->Reset(config_);
      state_tensor = get_state_tensor(state);
      all_rewards.emplace_back(episode_reward);
      episode_reward = 0.0f;
    }

    if (buffer_.size() > batch_size_) {
      torch::Tensor loss = compute_td_loss(batch_size_, gamma_);
      losses.emplace_back(loss.template item<float>());
    }

    if (frame_id % 100 == 0) {
      load_from_state_dict(target_model_, current_model_);
    }

    if (frame_id % 100 == 0 && all_rewards.size() >= 10) {
      // means of last rewards: testing only.
      float mean_value =
          torch::tensor(std::vector<float>{all_rewards.rbegin(), all_rewards.rbegin() + 10}).mean().item<float>();
      std::cout << "frame_id: " << frame_id << " mean_value: " << mean_value << std::endl;
    }
  }
}

void DqnTrainer::test(bool render) {
  torch::NoGradGuard no_grad;
  current_model_->eval();

  float total_reward = 0.0f;
  int64_t steps = 0;

  State state = client_->Reset(config_);
  torch::Tensor state_tensor = get_state_tensor(state);

  while (true) {
    torch::Tensor action_tensor = current_model_->act(state_tensor.to(device_));
    Step step = client_->Step(config_, action_tensor, render);
    torch::Tensor next_state_tensor = get_state_tensor(step.state());
    torch::Tensor reward_tensor = torch::tensor(step.reward());
    torch::Tensor done_tensor = torch::tensor(static_cast<float>(step.done()));

    state_tensor = next_state_tensor;
    total_reward += step.reward();
    ++steps;
    if (step.done()) {
      break;
    }
  }
  std::cout << "env_id: " << config_.env_id() << " test reward: " << total_reward << " test steps: " << steps
            << std::endl;
}

} // namespace deepx


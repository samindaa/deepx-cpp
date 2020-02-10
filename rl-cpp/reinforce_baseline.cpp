//
// Created by saminda on 2/3/20.
//

#include "reinforce_baseline.h"

#include <random>

namespace deepx {

ReinforceBaselineTrainer::ReinforceBaselineTrainer(std::shared_ptr<Client> client,
                                                   const EnvConfig &config) :
    client_(client),
    config_(config),
    rand_generator_(std::random_device{}()),
    device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {}

void ReinforceBaselineTrainer::train(int64_t num_frames) {
  policy_net_->train();
  std::vector<float> losses, all_rewards;
  running_reward_ = 0;
  float episode_reward = 0;

  State state = client_->Reset(config_);
  torch::Tensor state_tensor = get_state_tensor(state);

  for (int64_t frame_id = 0; frame_id < num_frames + 1; ++frame_id) {
    auto[action_tensor, log_prob_tensor] = policy_net_->get_action(state_tensor.to(device_));

    Step step = client_->Step(config_, action_tensor);
    saved_log_probs_.emplace_back(log_prob_tensor.unsqueeze(0));
    rewards_.emplace_back(step.reward());
    episode_reward += step.reward();
    torch::Tensor next_state_tensor = get_state_tensor(step.state());
    state_tensor = next_state_tensor;
    if (step.done()) {
      all_rewards.emplace_back(episode_reward);
      // start learning
      running_reward_ = 0.05 * episode_reward + (1 - 0.05) * running_reward_;
      episode_reward = 0.0f;
      compute_td_loss(rewards_.size(), gamma_);
      saved_log_probs_.clear();
      rewards_.clear();
    }

    if (frame_id % 100 == 0 && all_rewards.size() >= 10) {
      // means of last rewards: testing only.
      float mean_value =
          torch::tensor(std::vector<float>{all_rewards.rbegin(), all_rewards.rbegin() + 10}).mean().item<float>();
      std::cout << "frame_id: " << frame_id << " mean_value: " << mean_value << " running_reward: " << running_reward_
                << std::endl;
    }
  }
}

void ReinforceBaselineTrainer::test(bool render) {}

void ReinforceBaselineTrainer::define_models_and_optim() {
  policy_net_ = std::make_shared<PolicyNetwork>(config_.observation_space().box().shape()[0],
                                                config_.action_space().discrete().n(), /*n_hidden=*/128);
  policy_net_->to(device_);
  policy_opt_ = std::make_shared<torch::optim::Adam>(policy_net_->parameters(), torch::optim::AdamOptions{1e-2});
}

torch::Tensor ReinforceBaselineTrainer::compute_td_loss(int64_t batch_size, float gamma) {

  std::deque<float> returns;
  float reward = 0;
  for (auto r_iter = rewards_.rbegin(); r_iter != rewards_.rend(); ++r_iter) {
    reward = (*r_iter) + gamma * reward;
    returns.push_front(reward);
  }

  torch::Tensor rewards_to_go = torch::tensor(std::vector<float>(returns.begin(), returns.end())).to(device_);
  rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-3);
  torch::Tensor log_prob = torch::cat(saved_log_probs_).to(device_);
  torch::Tensor log_prob_rewards_to_go = log_prob * rewards_to_go;
  torch::Tensor loss = -log_prob_rewards_to_go.sum();

//  torch::Tensor prob_v = torch::softmax(logits_v, /*dim=*/1);
//  torch::Tensor entropy_v = -(prob_v * log_prob_v).sum(/*dim=*/1).mean();
//  torch::Tensor entropy_loss_v = -0.01 * entropy_v;
//  torch::Tensor loss_v = loss_policy_v + entropy_loss_v;

  policy_opt_->zero_grad();
  loss.backward();
  policy_opt_->step();

  return loss;
}

torch::Tensor ReinforceBaselineTrainer::get_state_tensor(const State &state) const {
  return torch::tensor(std::vector<float>{state.obs().begin(), state.obs().end()}).unsqueeze(0);
}

} // namespace deepx
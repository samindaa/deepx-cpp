//
// Created by saminda on 1/26/20.
//

#include "dueling_dqn.h"
#include "common.h"

namespace deepx {

DuelingDqnTrainer::DuelingDqnTrainer(std::shared_ptr<Client> client, const EnvConfig &config, int64_t buffer_size)
    : DqnTrainer(client, config, buffer_size) {}

void DuelingDqnTrainer::define_models_and_optim() {
  current_model_ = std::make_shared<DuelingDqn>(config_.observation_space().box().shape()[0],
                                                config_.action_space().discrete().n());
  target_model_ = std::make_shared<DuelingDqn>(config_.observation_space().box().shape()[0],
                                               config_.action_space().discrete().n());
  current_model_->to(device_);
  target_model_->to(device_);
  opt_ = std::make_shared<torch::optim::Adam>(current_model_->parameters(), torch::optim::AdamOptions{1e-3});

  load_from_state_dict(target_model_, current_model_);
}

torch::Tensor DuelingDqnTrainer::compute_td_loss(int64_t batch_size, float gamma) {
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

  auto state_tensor = torch::cat(state, 0);
  auto action_tensor = torch::cat(action, 0);
  auto reward_tensor = torch::cat(reward, 0);
  auto next_state_tensor = torch::cat(next_state, 0);
  auto done_tensor = torch::cat(done, 0);

  torch::Tensor q_values = current_model_->forward(state_tensor);
  torch::Tensor next_target_q_values = target_model_->forward(next_state_tensor);

  torch::Tensor q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1);
  torch::Tensor next_target_q_value = std::get<0>(next_target_q_values.max(1));
  torch::Tensor expected_q_value = reward_tensor + gamma * next_target_q_value * (1 - done_tensor);
  torch::Tensor loss = torch::mse_loss(q_value, expected_q_value.detach());

  opt_->zero_grad();
  loss.backward();
  opt_->step();

  return loss;

}

} // namespace deepx
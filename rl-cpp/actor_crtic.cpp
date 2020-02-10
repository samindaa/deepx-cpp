//
// Created by saminda on 2/9/20.
//

#include <iostream>
#include <torch/torch.h>

#include "gym.h"

namespace deepx {

class ActorCriticModel : public torch::nn::Module {
 public:
  ActorCriticModel(int64_t n_input, int64_t n_output, int64_t n_hidden) {
    fc_ = register_module("fc", torch::nn::Linear(n_input, n_hidden));
    action_ = register_module("action", torch::nn::Linear(n_hidden, n_output));
    value_ = register_module("value", torch::nn::Linear(n_hidden, 1));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
    x = torch::relu(fc_->forward(x));
    auto action_probs = torch::softmax(action_->forward(x), /*dim=*/1);
    auto state_values = value_->forward(x);
    return {action_probs, state_values};
  }

 private:
  torch::nn::Linear fc_{nullptr};
  torch::nn::Linear action_{nullptr};
  torch::nn::Linear value_{nullptr};

};

class PolicyNetwork {
 public:
  PolicyNetwork(int64_t n_state, int64_t n_action, int64_t n_hidden = 50, double learning_rate = 1e-3) {
    model_ = std::make_shared<ActorCriticModel>(n_state, n_action, n_hidden);
    optimizer_ = std::make_shared<torch::optim::Adam>(model_->parameters(), torch::optim::AdamOptions(learning_rate));
  }

  std::tuple<torch::Tensor, torch::Tensor> predict(torch::Tensor s) {
    return model_->forward(s);
  }

  void update(torch::Tensor returns_tensor,
              const std::vector<torch::Tensor> &log_probs,
              const std::vector<torch::Tensor> &state_values) {
    // TODO(saminda): update the logic
    auto log_probs_tensor = torch::stack(log_probs),
        state_values_tensor = torch::cat(state_values, /*dim=*/0).squeeze(/*dim=*/-1);
    auto advantage = returns_tensor - state_values_tensor;
    auto policy_loss = -(log_probs_tensor * advantage.detach()).mean();
    auto value_loss = torch::smooth_l1_loss(state_values_tensor, returns_tensor);
    auto loss = policy_loss + value_loss;
    optimizer_->zero_grad();
    loss.backward();
    optimizer_->step();
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_action(torch::Tensor s) {
    // TODO(saminda): check the indices
    auto[action_probs, state_value] = model_->forward(s);
    auto action = torch::multinomial(action_probs, /*num_samples*/1);
    auto log_prob = torch::log(action_probs.squeeze(0)[action.item<int64_t>()]);
    return {action, log_prob, state_value};
  }

 private:
  std::shared_ptr<ActorCriticModel> model_;
  std::shared_ptr<torch::optim::Optimizer> optimizer_;
};

torch::Tensor computer_returns(const std::vector<float> &rewards, float gamma) {
  float Gt = 0;
  std::deque<float> returns;
  for (auto reward = rewards.rbegin(); reward != rewards.rend(); ++reward) {
    Gt = (*reward) + gamma * Gt;
    returns.push_front(Gt);
  }
  torch::Tensor returns_tensor = torch::tensor(std::vector<float>(returns.begin(), returns.end()));
  return (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-3);
}

void actor_critic(std::shared_ptr<Client> client,
                  std::shared_ptr<PolicyNetwork> estimator,
                  const EnvConfig &config,
                  int64_t n_episode,
                  float gamma) {
  std::vector<float> total_reward_episode(n_episode, 0);
  for (auto episode = 0; episode < n_episode; ++episode) {
    std::vector<torch::Tensor> log_probs;
    std::vector<float> rewards;
    std::vector<torch::Tensor> state_values;

    auto state_tensor = client->Reset(config);

    while (true) {
      auto[action_tensor, log_prob_tensor, state_value_tensor] = estimator->get_action(state_tensor);
      log_probs.emplace_back(log_prob_tensor);
      state_values.emplace_back(state_value_tensor);

      auto [next_state_tensor, reward_tensor, done_tensor] = client->Step(config, action_tensor);
      total_reward_episode[episode] += reward_tensor.item<float>();
      rewards.emplace_back(reward_tensor.item<float>());

      if (done_tensor.item<bool>()) {
        auto returns_tensor = computer_returns(rewards, gamma);
        estimator->update(returns_tensor, log_probs, state_values);
        std::cout << "Episode: " << episode << ", total reward: " << total_reward_episode[episode] << std::endl;
        break;
      }
      state_tensor = next_state_tensor;
    }
  }
}

} // deepx

int main(int argc, char **argv) {
  auto client = deepx::Client::create_client("localhost", "50051");
  auto config = client->Create(/*id=*/"CartPole-v0", /*num_envs=*/1);
  auto n_state = config.observation_space().box().shape()[0];
  auto n_action = config.action_space().discrete().n();
  auto n_hidden = 128;
  auto learning_rate = 0.003;
  auto policy_net = std::make_shared<deepx::PolicyNetwork>(n_state, n_action, n_hidden, learning_rate);

  auto n_episode = 1000;
  auto gamma = 0.9;
  deepx::actor_critic(client, policy_net, config, n_episode, gamma);
  return 0;
}

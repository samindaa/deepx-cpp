//
// Created by Saminda Abeyruwan on 2019-06-08.
//

#ifndef SAMINDA_AIMA_CPP__SEARCH4E_H_
#define SAMINDA_AIMA_CPP__SEARCH4E_H_

#include <deque>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <deque>

namespace aima_cpp {

// The abstract class for a formal problem. A new domain subclasses this,
// overriding 'actions' and 'results', and perhaps other methods.
// The default heuristic is 0 and the default action cost is 1 for all states
// (or given an 'is_goal' method) and perhaps other keywords args for the
// subclass.
class Problem {
 public:
  virtual std::vector<std::string> Actions(const std::string &state) = 0;
  virtual std::string Result(const std::string &state,
                             const std::string &action) = 0;
  virtual bool IsGoal(const std::string &state) = 0;
  virtual float ActionCost(const std::string &state,
                           const std::string &action,
                           const std::string &next_state) = 0;
  virtual float h(const std::string &state) = 0;
  virtual std::string initial() = 0;
};

// A node in a search tree
class Node {
 public:
  std::string state;
  std::string action;
  float path_cost;
  std::shared_ptr<Node> parent;

  explicit Node(const std::string &state,
                std::shared_ptr<Node> parent = nullptr,
                const std::string &action = "",
                float path_cost = 0)
      : state(state), parent(parent), action(action), path_cost(path_cost) {}

};

std::vector<std::shared_ptr<Node>> expand(Problem &problem,
                                          std::shared_ptr<Node> node) {
//  std::cout << "--" << std::endl;
//  std::cout << node->state << ": " << node->path_cost << " => ";
  std::vector<std::shared_ptr<Node>> next_nodes;
  for (auto next_action : problem.Actions(node->state)) {
    auto next_state = problem.Result(node->state, next_action);
    auto
        cost = node->path_cost
        + problem.ActionCost(node->state, next_action, next_state);
//    std::cout << "(" << next_state << ", " << cost << ") ";
    next_nodes.push_back(std::make_shared<Node>(next_state,
                                                node,
                                                next_action,
                                                cost));
  }
//  std::cout << std::endl;
  return next_nodes;
}

class NodeComp {
 public:
  bool operator()(const std::pair<float,
                                  std::shared_ptr<Node>> &a,
                  const std::pair<float,
                                  std::shared_ptr<Node>> &b) {
    return a.first > b.first;
  }
};

float g(std::shared_ptr<Node> n) {
  return n->path_cost;
}

class PriorityQueue : public std::priority_queue<std::pair<float,
                                                           std::shared_ptr<Node>>,
                                                 std::vector<std::pair<float,
                                                                       std::shared_ptr<
                                                                           Node>>>,
                                                 NodeComp> {
 public:
  explicit PriorityQueue(std::function<float(std::shared_ptr<Node>)> f)
      : f(f) {}

  void Add(std::shared_ptr<Node> n) {
    this->emplace(f(n), n);
  }

  std::shared_ptr<Node> Top() {
    return this->top().second;
  }

  std::shared_ptr<Node> Pop() {
    auto top_value = Top();
    this->pop();
    return top_value;
  }

 private:
  std::function<float(std::shared_ptr<Node>)> f;
};

void PathActions(std::shared_ptr<Node> node,
                 std::vector<std::string> &path_actions) {
  if (!node) {
    return;
  }
  path_actions.emplace_back(node->action);
  PathActions(node->parent, path_actions);
}

const std::string kFailure = "failure";

void PathStates(std::shared_ptr<Node> node,
                std::vector<std::string> &path_states) {
  if (!node || node->state == kFailure) {
    return;
  }
  path_states.emplace_back(node->state);
  PathStates(node->parent, path_states);
}

std::shared_ptr<Node> BestFirstSearch(Problem &problem,
                                      std::function<float(std::shared_ptr<Node>)> f) {
  auto node = std::make_shared<Node>(problem.initial());
  PriorityQueue frontier(f);
  frontier.Add(node);
  std::unordered_map<std::string, std::shared_ptr<Node>> reached;
  reached.emplace(node->state, node);
  while (!frontier.empty()) {
    node = frontier.Pop();
    if (problem.IsGoal(node->state)) {
      return node;
    }
    for (const auto &child : expand(problem, node)) {
      auto s = child->state;
      if (reached.find(s) == reached.end()
          || (child->path_cost < reached[s]->path_cost)) {
        reached[s] = child;
        frontier.Add(child);
      }
    }
  }
  return std::make_shared<Node>(kFailure,
                                nullptr,
                                "",
                                std::numeric_limits<float>::max());
}

std::shared_ptr<Node> UniformCostSearch(Problem &problem) {
  return BestFirstSearch(problem, g);
}

//std::shared_ptr<Node> BreadthFirstSearch(Problem &problem) {
//  auto node = std::make_shared<Node>(problem.initial());
//  if (problem.IsGoal(node->state)) {
//    return node;
//  }
//  std::deque<std::shared_ptr<Node>> frontier;
//  frontier.push_back(node);
//  std::unordered_set<std::string> reached;
//  reached.emplace(node->state);
//  while (!frontier.empty()) {
//    node = frontier.front();
//    frontier.pop_front();
//    for (auto next_node : node->expand(problem)) {
//      auto new_state = next_node->state;
//      if (problem.IsGoal(new_state)) {
//        return node;
//      }
//      if (reached.find(new_state) == reached.end()) {
//        reached.emplace(new_state);
//        frontier.emplace(node);
//      }
//    }
//  }
//  return std::shared_ptr<Node>(true);
//}

//

struct PairHash {
  template<class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

class Map {
 public:
  Map(const std::unordered_map<std::pair<std::string, std::string>,
                               int, PairHash> &links,
      const std::unordered_map<std::string,
                               std::pair<int, int>> &locations) : distances(
      links), locations(locations) {

    for (const auto &link : links) {
      distances[{link.first.second, link.first.first}] = link.second;
    }

    for (const auto &kv: links) {
      neighbors[kv.first.first].emplace_back(kv.first.second);
      // undirected
      neighbors[kv.first.second].emplace_back(kv.first.first);
    }
  }

  std::unordered_map<std::pair<std::string, std::string>,
                     int, PairHash> distances;
  std::unordered_map<std::string,
                     std::pair<int, int>> locations;
  std::unordered_map<std::string, std::vector<std::string>> neighbors;

};

class RouteProblem : public Problem {
 public:
  RouteProblem(const std::string &initial_state,
               const std::string &goal_state,
               const Map &map)
      : initial_state(initial_state), goal_state(goal_state), map(map) {}

  std::vector<std::string> Actions(const std::string &state) override {
    return map.neighbors[state];
  }

  std::string Result(const std::string &state,
                     const std::string &action) override {
    return map.neighbors.find(action) != map.neighbors.end() ? action : state;
  }

  bool IsGoal(const std::string &state) override {
    return state == goal_state;
  }

  float ActionCost(const std::string &state,
                   const std::string &action,
                   const std::string &next_state) override {
    return map.distances[{state, next_state}];
  }

  float h(const std::string &state) override {
    return StraightLineDistance(map.locations[state],
                                map.locations[goal_state]);
  }

  std::string initial() override {
    return initial_state;
  }

 private:
  float StraightLineDistance(const std::pair<float, float> &a,
                             const std::pair<float, float> &b) {
    return std::sqrt(float(
        std::pow(a.first - b.first, 2) + std::pow(a.second - b.second, 2)));
  }

  std::string initial_state;
  std::string goal_state;
  Map map;
};

} // namespace aima_cpp

#endif //SAMINDA_AIMA_CPP__SEARCH4E_H_

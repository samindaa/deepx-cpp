import uuid
import gym
import grpc
import typing
import six
import numpy as np

from concurrent import futures

import gym_env_pb2
import gym_env_pb2_grpc

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string("host", "[::]", "Gym host name.")
flags.DEFINE_string("port", "50051", "Gym port.")


# Gym Environment wrapper
class Envs:

    def __init__(self):
        self._envs = {}

    def create(self, id, seed=None):
        try:
            env = gym.make(id)
            if seed:
                env.seed(seed)
        except gym.error.Error as e:
            raise ValueError("Failure to create the environment '{}'. Error: {}".format(id, e))

        env_id = uuid.uuid4().hex[:4]
        self._envs[env_id] = env
        config = gym_env_pb2.EnvConfig()
        config.env_id = env_id
        config.action_space.CopyFrom(self._get_space_properties(env.action_space))
        config.observation_space.CopyFrom(self._get_space_properties(env.observation_space))
        return config

    def reset(self, env_id):
        env = self._envs[env_id]
        state = gym_env_pb2.State()
        state.obs.extend(env.reset().tolist())
        return state

    def step(self, env_id, action, render=False):
        env = self._envs[env_id]

        # TODO(saminda): extensions
        if render:
            env.render()

        if isinstance(action, six.integer_types) or len(action) == 1:
            nice_action = action[0]
        else:
            nice_action = np.array(action)
        obs, reward, done, info = env.step(nice_action)
        next_state = gym_env_pb2.State()
        next_state.obs.extend(obs.tolist())
        step_response = gym_env_pb2.Step(reward=reward, done=done)
        step_response.state.CopyFrom(next_state)
        return step_response

    @staticmethod
    def _get_space_properties(space: typing.Union[gym.spaces.Space, gym.spaces.Box]):
        space_config = gym_env_pb2.Space()
        if isinstance(space, gym.spaces.Discrete):
            space_config.space_discrete.n = space.n
        else:
            space_config.space_box.shape.extend(space.shape)
            space_config.space_box.low.extend(space.low.tolist())
            space_config.space_box.high.extend(space.high.tolist())
        return space_config


class GymService(gym_env_pb2_grpc.GymServicer):

    def __init__(self):
        super(GymService, self).__init__()
        self._envs = Envs()

    def Create(self, request, context):
        config = self._envs.create(request.id)
        response = gym_env_pb2.CreateResponse()
        response.config.CopyFrom(config)
        return response

    def Reset(self, request, context):
        state = self._envs.reset(request.env_id)
        response = gym_env_pb2.ResetResponse()
        response.state.CopyFrom(state)
        return response

    def Step(self, request, context):
        step_response = self._envs.step(request.env_id, request.action)
        response = gym_env_pb2.StepResponse()
        response.step.CopyFrom(step_response)
        return response


def main(_):
    logging.info("GymService starting ...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    gym_env_pb2_grpc.add_GymServicer_to_server(GymService(), server)
    server.add_insecure_port(":".join([FLAGS.host, FLAGS.port]))
    server.start()
    logging.info("Server started ...")
    server.wait_for_termination()

if __name__ == "__main__":
    app.run(main)
import uuid
import gym
import grpc
import typing
import six
import numpy as np
import multiprocessing_env

from concurrent import futures

import gym_env_pb2
import gym_env_pb2_grpc

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string("host", "[::]", "Gym host name.")
flags.DEFINE_string("port", "50051", "Gym port.")
flags.DEFINE_integer("max_workers", 10, "Max workers.")


# Gym Environment wrapper
class Envs:

    def __init__(self):
        self._envs = {}

    def create(self, id, num_envs, seed=None):
        def _make_env():
            def _env():
                env = gym.make(id)
                if seed:
                    env.seed(seed)
                return env
            return _env

        envs = [_make_env() for _ in range(num_envs)]
        env = multiprocessing_env.SubprocVecEnv(envs)

        env_id = uuid.uuid4().hex[:4]
        self._envs[env_id] = env
        config = gym_env_pb2.EnvConfig()
        config.env_id = env_id
        config.num_envs = env.nenvs
        config.action_space.CopyFrom(self._get_space_properties(env.action_space))
        config.observation_space.CopyFrom(self._get_space_properties(env.observation_space))
        return config

    def reset(self, env_id, response: gym_env_pb2.ResetResponse):
        env = self._envs[env_id]
        obs = env.reset()
        for i in range(env.nenvs):
            state = response.states.add()
            state.obs.extend(obs[i].tolist())
        return response

    def step(self, env_id, action, response: gym_env_pb2.StepResponse):
        env = self._envs[env_id]
        # TODO (saminda): handle multi actions
        obs, rewards, dones, _ = env.step(action)

        for i in range(env.nenvs):
            next_step = response.steps.add()
            next_step.reward = rewards[i]
            next_step.done = dones[i]
            next_step.state.obs.extend(obs[i].tolist())
        return response

    @staticmethod
    def _get_space_properties(space: typing.Union[gym.spaces.Space, gym.spaces.Box]):
        space_config = gym_env_pb2.Space()
        if isinstance(space, gym.spaces.Discrete):
            space_config.discrete.n = space.n
        else:
            space_config.box.shape.extend(space.shape)
            space_config.box.low.extend(space.low.tolist())
            space_config.box.high.extend(space.high.tolist())
        return space_config

    def close(self, env_id, response: gym_env_pb2.StepResponse):
        env = self._envs[env_id]
        env.close()
        del self._envs[env_id]
        return response

    def list(self, response: gym_env_pb2.ListResponse):
        for key in self._envs.keys():
            response.env_ids.append(key)
        return response


class GymService(gym_env_pb2_grpc.GymServicer):

    def __init__(self):
        super(GymService, self).__init__()
        self._envs = Envs()

    def Create(self, request, context):
        config = self._envs.create(request.id, request.num_envs)
        response = gym_env_pb2.CreateResponse()
        response.config.CopyFrom(config)
        return response

    def Reset(self, request, context):
        response = gym_env_pb2.ResetResponse()
        return self._envs.reset(request.env_id, response)

    def Step(self, request, context):
        response = gym_env_pb2.StepResponse()
        return self._envs.step(request.env_id, request.action, response)

    def Close(self, request, context):
        response = gym_env_pb2.CloseResponse()
        return self._envs.close(request.env_id, response)

    def List(self, request, context):
        response = gym_env_pb2.ListResponse()
        return self._envs.list(response)


def main(_):
    logging.info("GymService starting ...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=FLAGS.max_workers))
    gym_env_pb2_grpc.add_GymServicer_to_server(GymService(), server)
    server.add_insecure_port(":".join([FLAGS.host, FLAGS.port]))
    server.start()
    logging.info("Server started ...")
    server.wait_for_termination()


if __name__ == "__main__":
    app.run(main)

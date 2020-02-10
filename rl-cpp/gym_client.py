import grpc

import gym_env_pb2
import gym_env_pb2_grpc

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string("host", "localhost", "Gym host name.")
flags.DEFINE_string("port", "50051", "Gym port.")


def main(_):
    with grpc.insecure_channel(":".join([FLAGS.host, FLAGS.port])) as channel:
        stub = gym_env_pb2_grpc.GymStub(channel)
        # Testing loop
        create_response = stub.Create(gym_env_pb2.CreateRequest(id="CartPole-v1", num_envs=4))
        env_id = create_response.config.env_id
        logging.info("config '%s'", create_response)

        state_response = stub.Reset(gym_env_pb2.ResetRequest(env_id=env_id))
        logging.info("state: %s", state_response)
        for i in range(5):
            step_response = stub.Step(gym_env_pb2.StepRequest(env_id=env_id, action=[0]*4))
            logging.info("step: %s", step_response)

        list_response = stub.List(gym_env_pb2.ListResponse())
        logging.info("list: %s", list_response)

        stub.Close(gym_env_pb2.CloseRequest(env_id=env_id))

        list_response = stub.List(gym_env_pb2.ListResponse())
        logging.info("list: %s", list_response)


if __name__ == "__main__":
    app.run(main)

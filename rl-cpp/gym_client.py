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
        create_response = stub.Create(gym_env_pb2.CreateRequest(id="CartPole-v1"))
        env_id = create_response.config.env_id
        logging.info("config '%s'", create_response)

        state_response = stub.Reset(gym_env_pb2.ResetRequest(env_id=env_id))
        logging.info("state: %s", state_response)
        while True:
            step_response = stub.Step(gym_env_pb2.StepRequest(env_id=env_id, action=[0], render=False))
            logging.info("step: %s", step_response)

            if step_response.step.done:
                break


if __name__ == "__main__":
    app.run(main)

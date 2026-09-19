import pytest

from longctx.config import load_env_config, load_generator_config
from longctx.generate import InstanceGenerator


@pytest.fixture(scope="session")
def env_cfg():
    return load_env_config()


@pytest.fixture(scope="session")
def gen_cfg():
    return load_generator_config()


@pytest.fixture(scope="session")
def generator(env_cfg, gen_cfg):
    return InstanceGenerator(gen_cfg, env_cfg)


@pytest.fixture(scope="session")
def instances(generator):
    """A small mixed set: hops 2..5, 32k tokens."""
    return [generator.generate_valid(1000 + s, 2 + s % 4, 32000, "test")[0] for s in range(48)]


@pytest.fixture(scope="session")
def inst3(generator):
    return generator.generate_valid(4242, 3, 32000, "test")[0]

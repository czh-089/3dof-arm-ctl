"""pytest fixtures for arm-ctl tests"""
import os
import sys
import pytest
import numpy as np


@pytest.fixture(scope='module')
def dyn():
    from arm import ArmDynamics
    return ArmDynamics()


@pytest.fixture(scope='module')
def fk():
    from arm import ForwardKinematics
    return ForwardKinematics()


@pytest.fixture
def sample_state():
    q = np.array([0.5, 0.3, -0.8])
    dq = np.array([0.1, -0.2, 0.15])
    ddq = np.array([1.0, -2.0, 3.0])
    return q, dq, ddq

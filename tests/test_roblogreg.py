"""Test roblogreg"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from roblogreg import roblogreg


def test_roblogreg():
    sample_size = 2000
    n_class = 3
    x_ncol = 2

    # Set seed
    g_cpu = torch.Generator()
    g_cpu.manual_seed(372)
    # Generate random X
    X = torch.rand(size=(sample_size, x_ncol), generator=g_cpu)
    X = torch.cat([torch.ones([sample_size, 1]), X], dim=-1)
    # Define Theta
    theta = torch.tensor([[-0.1, 0.25, -0.25], [0.15, -0.15, -0.1]]).t()
    # Generate logits
    logits = F.pad(torch.matmul(X, theta), [0, 1, 0, 0])
    # Generate e
    g_cpu = torch.Generator()
    g_cpu.manual_seed(172)
    e = torch.rand(size=(sample_size, n_class), generator=g_cpu)
    # Generate Y
    Y = torch.argmax(
        logits + e.log().neg().log().neg(),
        dim=-1,
        keepdim=True,
    )

    model_robust = roblogreg.MMR(
        X, Y, x_ncol + 1, n_class, learning_rate=0.001, model_type="BY"
    )
    model_ml = roblogreg.MMR(
        X, Y, x_ncol + 1, n_class, learning_rate=0.001, model_type="ML"
    )

    model_robust.train(epochs=10000)
    model_ml.train(epochs=10000)

    max_err = (model_ml.get_theta() - model_robust.get_theta()).max()

    assert max_err < 0.01

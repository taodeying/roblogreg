"""Robust Multinomial Regression"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import MinCovDet

torch.set_default_dtype(torch.float64)


class LogitProb(nn.Module):
    """
    Return de log probability

    LogitProb is probability of each class in logit scale. \
    It takes the input x and returns the output LogitProb, \
    pading the last column with 0. the last column serves as \
    the base-category: \
    LogitProb = pad(input * theta, [0, 1, 0, 0])

    Args:
        x_ncol (int): the number of columns of the input x
        n_class (int): the number of classes

    Attributes:
        theta: The matrix theta. Is multinomial glm parameter, \
            whose dimension is [x_ncol, n_class-1].

    Examples::
        >>> x_input = torch.randn(10, 5)
        >>> LogitProb = LogitProb(5, 2)
        >>> output = LogitProb(x_input)
        >>> print(output.size())
        torch.Size([10, 3])
    """

    def __init__(self, x_ncol: int, n_class: int) -> torch.Tensor:
        super(LogitProb, self).__init__()

        init_param = torch.randn(x_ncol, n_class - 1) / math.sqrt(784)

        self.theta = nn.Parameter(init_param)

    def forward(self, x_input: torch.Tensor):
        """
        forward

        returns the Logit Probability for each of the clases.

        Args:
            x_input(torch.Tensor)[x_nrow, x_ncol-1]: The X. \
                First column should be a 1 column

        Returns:
            torch.Tensor: LogitProb[x_nrow, n_class], where \
                the last column serves as the base-category.
        """
        return F.pad(torch.matmul(x_input, self.theta), [0, 1, 0, 0])


def logpdf_multinom(
    y_input: torch.Tensor, logits: torch.Tensor, w_x: torch.Tensor
) -> torch.Tensor:
    """
    logpdf_multinom Log pdf of the multinomial

    Log pdf of the multinomial

    Args:
        y_input (torch.Tensor)[x_nrow, 1]: a tensor that contains the clases. It takes values from [0:n_clases-1]
        logits (torch.Tensor)[x_nrow, n_clases]: logit probability
        w_x (torch.Tensor)[x_nrow, 1]: MCD weight . Not used

    Returns:
        torch.Tensor: Log pdf of the multinomial
    """
    log_p = F.log_softmax(logits, dim=-1)
    log_pdf = torch.gather(log_p, -1, y_input).sum()

    return log_pdf.sum().neg()


def rho(x: torch.Tensor, d=0.5) -> torch.Tensor:
    """
    rho BY rho function

    Bianco-Yohai rho robust function for multinomial logit. Croux y Haesbroeck type

    Args:
        x (torch.Tensor): logit probability
        d (float, optional): robustness hyperparameter. Defaults to 0.5.

    Returns:
        torch.Tensor: rho(-log(pi_ij))
    """
    x = F.log_softmax(x, dim=-1).neg()
    # x = F.softmax(x, dim=-1)

    k_0 = math.exp(-math.sqrt(d))
    k_1 = k_0 * (2.0 * (1.0 + math.sqrt(d)) + d)
    out = torch.where(
        x <= d,
        x * k_0,
        -2 * x.sqrt().neg().exp() + -2 * x.sqrt().neg().exp() * x.sqrt() + k_1
        # (x.sqrt().neg() + x.sqrt().log1p() + math.log(2.0)).exp().neg() + k_1,
    )
    return out


def phi(y: torch.Tensor) -> torch.Tensor:
    """
    phi phi function

    calculates the cdf values of a normal variable

    Args:
        y_input (torch.Tensor): imput value

    Returns:
        torch.Tensor: normal_cdf(X)
    """
    return 0.5 * (1 + torch.erf(y / math.sqrt(2.0)))


def G(x: torch.Tensor, d=0.5) -> torch.Tensor:
    """
    G correction term

    Correction term to get a fisher consisten estimator. Croux y Haesbroeck type

    Args:
        x (torch.Tensor): _description_
        d (float, optional): _description_. Defaults to 0.5.

    Returns:
        torch.Tensor: _description_
    """
    log_x = F.log_softmax(x, dim=-1)
    x = F.softmax(x, dim=-1)
    c_1 = math.exp(1.0 / 4.0) * math.sqrt(math.pi)
    c_2 = math.exp(-1.0 / 4.0) * math.sqrt(math.pi)
    c_3 = math.sqrt(d)
    aux = torch.tensor([math.sqrt(2) * (0.5 + math.sqrt(d))])
    c_4 = -c_2 + c_1 * torch.special.ndtr(aux)

    c_5 = math.exp(-math.sqrt(d))

    out = torch.where(
        x <= math.exp(-d),
        (
            (log_x - log_x.neg().sqrt()).exp()
            + c_1
            * torch.special.ndtr(math.sqrt(2) * (0.5 + log_x.neg().sqrt()))
            - c_2
        ),
        x * c_5 + c_4,
    )

    return out


def loss_by(
    y_input: torch.Tensor, logits: torch.Tensor, w_x: torch.Tensor
) -> torch.Tensor:
    """
    loss_by BY loss

    BY loss in the form of Croux y Haesbroeck

    Args:
        y_input (torch.Tensor)[x_nrow, 1]: a tensor that contains the clases. \
             It takes values from [0:n_clases-1]
        logits (torch.Tensor)[x_nrow, n_clases]: logit probability
        w_x (torch.Tensor)[x_nrow, 1]: MCD weight 

    Returns:
        torch.Tensor: BY loss for the sample.
    """
    y_rho = torch.gather(rho(logits), -1, y_input)
    # y_G = torch.gather(G(logits), -1, y_input)
    y_G = G(logits).sum(dim=-1).unsqueeze(-1)
    return ((y_rho + y_G) * w_x).sum()


class MMR:
    """
    Multinomial Multivariate Regresion

    Initializates a Multinomial Multivariate Regresion class. \
        It can be a robust version (model_type="BY"), or a clasical\
        one, based on maximum likelihood

    Args:
        x_input(torch.Tensor)[x_nrow, x_ncol-1]: The X. \
                First column should be a 1 column
        y_input (torch.Tensor)[x_nrow, 1]: a tensor that contains the clases. \
                It takes values from [0:n_clases-1]
        x_ncol (int): the number of columns of the input x
        n_class (int): the number of classes
        learning_rate (float, optional): Learning Rate por AdaMax. Defaults to 1e-2.
        model_type (str, optional): BY for robust, ML for maximum likelihood. \
            Defaults to "BY".
    
    Attributes:
        optimizer: torch.optim.Adamax
        w_x: Hubber Weights on X, Minimum Covariance Determinant 
    """

    def __init__(
        self,
        x_input: torch.Tensor,
        y_input: torch.Tensor,
        x_ncol: int,
        n_class: int,
        learning_rate: float = 1e-2,
        model_type: str = "BY",
    ):
        super(MMR, self).__init__()
        self.x_input = x_input
        self.y_input = y_input
        self.x_ncol = x_ncol
        self.n_class = n_class
        self.logit_prob = LogitProb(x_ncol, n_class)

        match model_type:
            case "ML":
                self.loss = logpdf_multinom
                print("ML model")
            case "BY":
                self.loss = loss_by
                print("robust model")
            case _:
                print("Err: model type not found")

        self.optimizer = torch.optim.Adamax(
            self.logit_prob.parameters(), lr=learning_rate
        )

        # Minimum Covariance Determinant
        mcd = MinCovDet(random_state=0).fit(x_input[:, 1:3].numpy())
        out = mcd.reweight_covariance(x_input[:, 1:3].numpy())
        self.w_x = torch.tensor(out[2] + 0.0).unsqueeze(-1)

    def train(self, epochs: int = 100):
        """
        train trains the model

        Trains the model using Adamax

        Args:
            epochs (int, optional): number of epochs in the training loop. \
                 Defaults to 100.
        """
        loss_t0 = 1000000.0
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            logits = self.logit_prob(self.x_input)
            loss = self.loss(self.y_input, logits, self.w_x)
            loss.backward()
            self.optimizer.step()

            if epoch % 999 == 0:
                if (loss_t0 - loss.item()) < 0.0001:
                    break
                else:
                    loss_t0 = loss.item()

        print(loss)

    def predict(
        self, x_input: torch.Tensor, pred_type: str = "prob"
    ) -> torch.Tensor:
        """
        predict Prediction for mmr

        Prediction for multivariate multinomial regression. The option \
            "prob" returns the probability of each class. The option \
            "logit" returns the logit of the probability of each class. 
            The option "class" returns the most probable class. 

        Args:
            x_input (torch.Tensor): The X. \
                First column should be a 1 column
            pred_type (str, optional): prediction type \
                ("prob", "logit", "class"). Defaults to "prob".

        Returns:
            torch.Tensor: a tensor with the prediction
        """
        logits = self.logit_prob(x_input)

        match pred_type:
            case "prob":
                out = F.softmax(logits, dim=-1)
            case "logit":
                out = logits
            case "class":
                out = torch.argmax(logits, dim=-1)
            case _:
                print("Err: prediction type not found")

        return out

    def get_theta(self) -> torch.Tensor:
        """
        get_theta Returns Theta

        Returns Theta, the parameters of the glm multinomial logit

        Returns:
            torch.Tensor: a matrix with the parameters
        """
        return self.logit_prob.theta

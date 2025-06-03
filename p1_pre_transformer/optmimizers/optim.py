from torch import nn
from torch.optim import Optimizer



class VanillaGD(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(VanillaGD, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                # Optimizer state
                state = self.state[p]
                grad = p.grad.data

                # Update parameters
                p.data -= lr * grad

class GDMomentum(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01, rho: float = 0.99, nesterov: bool = False):
        super(GDMomentum, self).__init__(params, dict(lr=lr, rho = rho, nesterov = nesterov))

    def step(self):
        # TODO: add "Nesterov" case
        for group in self.param_groups:
            lr = group["lr"]
            rho = group["rho"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                # Optimizer state
                state = self.state[p]
                grad = p.grad.data

                # Get Velocity
                old_v = state.get("v", torch.zeros_like(grad))
                
                # Update Velocity state
                if nesterov:
                    v = rho * old_v - lr * grad     
                else:
                    v = rho * old_v + grad
                state["v"] = v

                # Update parameters
                if nesterov:
                    p.data += -rho * old_v + (1 + rho) * v
                else:
                    p.data -= lr * v

class AdaGrad(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                # Optimizer state
                state = self.state[p]
                grad = p.grad.data

                # Get squared gradients g2
                g2 = state.get("g2", torch.zeros_like(grad))

                # Update optimizer state
                g2 += torch.square(grad)
                state["g2"] = g2

                # Update parameters
                p.data -= lr * grad / torch.sqrt(g2 +1e-5)

class RMSProp(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01, decay_rate: float = 0.99):
        super(RMSProp, self).__init__(params, dict(lr=lr, decay_rate=decay_rate))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            decay_rate = group["decay_rate"]
            for p in group["params"]:
                # Optimizer state
                state = self.state[p]
                grad = p.grad.data

                # Get squared gradients g2
                g2 = state.get("g2", torch.zeros_like(grad))

                # Update optimizer state
                g2 = decay_rate * g2  + (1 - decay_rate) * torch.square(grad)
                state["g2"] = g2

                # Update parameters
                p.data -= lr * grad / torch.sqrt(g2 +1e-5)

class Adam(Optimizer):
    """
    Adam optimizer implementation. 

    Attributes:
        lr (float): Learning rate of the optimization.
        beta1 (float): first moment decay ratio.
        beta2 (float): second moment decay ratio.
        weight_decay (float): weight decay ration or L2 regularization ratio.
        decoupled (bool): AdamW vs Adam
    """
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 5e-4, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.0, decoupled: bool = False):
        super(Adam, self).__init__(params, dict(lr=lr, beta1=beta1, beta2=beta2, weight_decay = weight_decay, decoupled = decoupled))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            wd = group["weight_decay"]
            adamW = group["decoupled"]
            t = group.get("t", 1) # NOTE: should I move it not to be atatched to each group?
            for p in group["params"]:
                # Optimizer state
                state = self.state[p]
                grad = p.grad.data
                if not adamW:
                    grad += wd * p.data

                # Get params from the state
                first_moment = state.get("first_moment", torch.zeros_like(grad))
                second_moment = state.get("second_moment", torch.zeros_like(grad))

                # Update optimizer state
                first_moment = beta1 * first_moment + (1 - beta1) * grad
                second_moment = beta2 * second_moment + (1 - beta2) * grad ** 2
                state["first_moment"] = first_moment
                state["second_moment"] = second_moment
                
                # Update parameters
                first_unbias = first_moment / (1 - beta1 ** t)
                second_unbias = second_moment / (1 - beta2 ** t)
                decoupled_wd = 0 if not adamW else wd * p.data 
                p.data -= lr * (first_unbias / torch.sqrt(second_unbias +1e-7) + decoupled_wd)
            
            # update state of "t" (current gradient step)
            group["t"] += 1

class Lion(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 5e-4):
        super(Lion, self).__init__(params, dict(lr=lr))

    def step(self):
        raise NotImplementedError("This method has not been implemented yet.")

class Adan(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 5e-4):
        super(Adan, self).__init__(params, dict(lr=lr))

    def step(self):
        raise NotImplementedError("This method has not been implemented yet.")

class AdaFactor(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 5e-4):
        super(AdaFactor, self).__init__(params, dict(lr=lr))

    def step(self):
        raise NotImplementedError("This method has not been implemented yet.")

class Lamb(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 5e-4):
        super(Lamb, self).__init__(params, dict(lr=lr))

    def step(self):
        raise NotImplementedError("This method has not been implemented yet.")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--optimizer", 
        type=str, 
        required=True, 
        choices=["sgd", "sgd-momentum", "sgd-nesterov", "adagrad", "rmsprop", "adam"],
        default = "adagrad",
        help="Optimizer to train"
    )
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Number of epochs to train')
    parser.add_argument('--device', type = str, default = 'cuda:0', help = 'cuda, cuda:<n>, or cpu')
    inputs = parser.parse_args()
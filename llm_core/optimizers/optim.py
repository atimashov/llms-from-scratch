import torch
from torch import nn
from torch.optim import Optimizer
from collections.abc import Callable, Iterable
import math
from typing import Dict


class VanillaGD(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr))

    def step(self, closure: Callable | None = None):
        # NOTE: "closure" not necessary for classic SGD, but can be used for optimizers requiring multiple loss/grad calculations per step (e.g. LBFGS)
        loss = None if closure is None else closure() 
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Optimizer state
                state = self.state[p]
                grad = p.grad.data

                # Update parameters
                p.data -= lr * grad
        return loss


class GDMomentum(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01, rho: float = 0.99, weight_decay: float = 0.01, nesterov: bool = False):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr, rho = rho, weight_decay = weight_decay, nesterov = nesterov))

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure() 
        for group in self.param_groups:
            lr = group["lr"]
            rho = group["rho"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # Optimizer State and Gradients
                state = self.state[p]
                
                # Apply Weight Decay
                grad = p.grad.data + weight_decay * p.data

                # Get Velocity
                if "v" not in state:
                    state["v"] = torch.zeros_like(grad)
                state["v"].mul_(rho).add_(grad)

                # Update parameters
                update = rho * state["v"] + grad if nesterov else state["v"] 
                p.data -= lr * update
        return loss

class AdaGrad(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr))

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure() 
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
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
        return loss

class RMSProp(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01, decay_rate: float = 0.99):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr, decay_rate=decay_rate))

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure() 
        for group in self.param_groups:
            lr = group["lr"]
            decay_rate = group["decay_rate"]
            for p in group["params"]:
                if p.grad is None:
                    continue
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
        return loss


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
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 5e-4, betas: tuple = (0.9, 0.999), weight_decay: float = 0.0, eps: float = 1e-8, decoupled: bool = True):
        assert lr > 0, f"Invalid learning rate: {lr}"
        assert len(betas) == 2, f"Invalid betas: {betas}"
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay = weight_decay, eps = eps, decoupled = decoupled))

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure() 
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            adamW = group["decoupled"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # Optimizer state
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                if not adamW:
                    grad.add_(p.data, alpha = wd)

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

                # Docoupled Weight Decay
                if adamW and wd > 0:
                    p.data.mul_(1 - lr * wd) 
                p.data -= lr * first_unbias / torch.sqrt(second_unbias + eps)
            
                # update state of "t" (current gradient step)
                state["t"] = t + 1
        return loss


class Lion(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 5e-4, betas: tuple = (0.9, 0.99), nesterov: bool = False, weight_decay: float = 0.0, is_trust_ratio: bool = False, eps: float = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr, betas = betas, nesterov = nesterov, weight_decay = weight_decay, is_trust_ratio = is_trust_ratio, eps = eps))

    def step(self, closure: Callable | None = None, param_to_name: Dict | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            is_trust_ratio = group["is_trust_ratio"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # Optimizer State and Gradients
                state = self.state[p]
                grad = p.grad.data

                if "m" not in state:
                    state["m"] = torch.zeros_like(grad)

                # Get Trust Ratio (per layer)
                if is_trust_ratio: # TODO: remove 'bias', 'bn', 'layernorm' etc.
                    param_norm = torch.norm(p.data)
                    # grad_norm = torch.norm(p.grad) # we apply raw gradient without L2 weight deacy
                    update_vec = torch.sign(state["m"])
                    update_norm = torch.norm(update_vec)
                    trust_ratio = 1.0 if param_norm == 0 or update_norm == 0 else param_norm / (update_norm + eps) #(grad_norm + eps)
                    step_size = lr * trust_ratio
                    # TODO: DELETE
                    param_name = param_to_name[p]
                    # print(f"{' ' * (20 - len(param_name))}{param_name} | param_norm={param_norm.item():.4e}, update_norm={update_norm.item():.4e}, step_size={step_size:.4e}")
                    # TODO: DELETE
                else:
                    step_size = lr
                
                if nesterov:
                    raise NotImplementedError("This method has not been implemented yet.")
                else:
                    # Decoupled weight decay
                    if weight_decay > 0:
                        p.data.mul_(1 - step_size * weight_decay)
                    # Optimizer Step
                    update = beta1 * state["m"] + (1-beta1) * grad
                    p.data.add_(torch.sign(update), alpha = -step_size)
                    # State Update
                    state["m"].mul_(beta2).add_(grad, alpha = 1 - beta2)
        return loss


class Adan(Optimizer):
    """
    Adan optimizer implementation. It always uses decoupled weight decay. 

    Attributes:
        lr (float): Learning rate of the optimization.
        beta1 (float): first moment decay ratio.
        beta2 (float): delta first moment decay ratio.
        beta3 (float): delta second moment decay ratio.
        weight_decay (float): weight decay ratio.
    """
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 5e-4, betas: tuple = (0.98, 0.92, 0.99), weight_decay: float = 0.0, eps: float = 1e-8):
        assert lr > 0, f"Invalid learning rate: {lr}"
        assert len(betas) == 3, f"Invalid betas: {betas}"
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay = weight_decay, eps = eps))

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure() 
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                # Optimizer state
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data

                # Get params from the state
                if "prev_grad" not in state:
                    state["prev_grad"] = torch.zeros_like(grad)
                prev_grad = state["prev_grad"]
                if "ema_gradient" not in state:
                    state["ema_gradient"] = torch.zeros_like(grad)
                if "first_moment" not in state:
                    state["first_moment"] = torch.zeros_like(grad)
                if "second_moment" not in state:
                    state["second_moment"] = torch.zeros_like(grad)

                # Update optimizer state
                state["ema_gradient"].mul_(beta1).add_(grad, alpha = 1 - beta1)
                state["first_moment"].mul_(beta2).add_(grad - prev_grad, alpha = 1 - beta2)
                state["second_moment"].mul_(beta3).add_((grad + beta2 * (grad - prev_grad))**2, alpha = 1 - beta3)
                
                # Update parameters
                m_t = state["ema_gradient"] / (1 - beta1 ** t)
                v_t = state["first_moment"] / (1 - beta2 ** t)
                n_t = state["second_moment"]/ (1 - beta3 ** t)

                # Optimizer Step
                step_size = lr / torch.sqrt(n_t + eps)
                p.data.add_(-step_size * (m_t + beta2 * v_t))

                # Docoupled Weight Decay (in Adan it goes after step)
                if wd > 0:
                    p.data.mul_(1 / (1 + lr * wd)) 


                # update state of "t" (current gradient step)
                state["t"] = t + 1
                state["prev_grad"] = grad.clone()
        return loss

class AdaFactor(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 5e-4):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr))

    def step(self, closure: Callable | None = None):
        raise NotImplementedError("This method has not been implemented yet.")

class Lars(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01, rho: float = 0.99, weight_decay: float = 0.01, nesterov: bool = False, eps = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr, rho = rho, weight_decay = weight_decay, nesterov = nesterov, eps = eps))

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure() 
        for group in self.param_groups:
            lr = group["lr"]
            rho = group["rho"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                # Optimizer State and Gradients
                state = self.state[p]
                
                # Apply Weight Decay
                grad = p.grad.data + weight_decay * p.data

                # Get Velocity
                if "v" not in state:
                    state["v"] = torch.zeros_like(grad)
                state["v"].mul_(rho).add_(grad)

                # Get Trust Ratio (per layer)
                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(p.grad) # we apply raw gradient without L2 weight deacy
                trust_ratio = 1.0 if param_norm == 0 or grad_norm == 0 else param_norm / (grad_norm + eps)
                step_size = lr * trust_ratio

                # Update parameters
                update = rho * state["v"] + grad if nesterov else state["v"] 
                p.data -= step_size * update
        return loss


class Lamb(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 5e-4):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr))

    def step(self, closure: Callable | None = None):
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
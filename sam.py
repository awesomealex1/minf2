import torch

class SAM(torch.optim.Optimizer):
    '''
    Optimizer used for Sharpness-Aware Minimization
    '''

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False, tmp=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# class SAMSGD(SGD):
#     """ SGD wrapped with Sharp-Aware Minimization

#     Args:
#         params: tensors to be optimized
#         lr: learning rate
#         momentum: momentum factor
#         dampening: damping factor
#         weight_decay: weight decay factor
#         nesterov: enables Nesterov momentum
#         rho: neighborhood size

#     """

#     def __init__(self,
#                  params: Iterable[torch.Tensor],
#                  lr: float,
#                  momentum: float = 0,
#                  dampening: float = 0,
#                  weight_decay: float = 0,
#                  nesterov: bool = False,
#                  rho: float = 0.05,
#                  ):
#         if rho <= 0:
#             raise ValueError(f"Invalid neighborhood size: {rho}")
#         super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
#         # todo: generalize this
#         if len(self.param_groups) > 1:
#             raise ValueError("Not supported")
#         self.param_groups[0]["rho"] = rho

#     @torch.no_grad()
#     def step1(self,
#              closure
#              ) -> torch.Tensor:
#         """

#         Args:
#             closure: A closure that reevaluates the model and returns the loss.

#         Returns: the loss value evaluated on the original point

#         """
#         closure = torch.enable_grad()(closure)
#         loss = closure().detach()

#         for group in self.param_groups:
#             grads = []
#             params_with_grads = []

#             rho = group['rho']
#             # update internal_optim's learning rate

#             for p in group['params']:
#                 if p.grad is not None:
#                     # without clone().detach(), p.grad will be zeroed by closure()
#                     grads.append(p.grad.clone().detach())
#                     params_with_grads.append(p)
#             device = grads[0].device

#             # compute \hat{\epsilon}=\rho/\norm{g}\|g\|
#             grad_norm = torch.stack([g.detach().norm(2).to(device) for g in grads]).norm(2)
#             epsilon = grads  # alias for readability
#             torch._foreach_mul_(epsilon, rho / grad_norm)

#             # virtual step toward \epsilon
#             torch._foreach_add_(params_with_grads, epsilon)
#             # compute g=\nabla_w L_B(w)|_{w+\hat{\epsilon}}
#             closure()
#             # virtual step back to the original point
#             torch._foreach_sub_(params_with_grads, epsilon)

#         #super().step()
#         return loss
    
#     def step2(self):
#         super().step()
#         super().zero_grad()
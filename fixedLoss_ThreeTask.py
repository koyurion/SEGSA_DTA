# Custom loss-1: pairwise
class Masked_BCELoss(nn.Module):
    def __init__(self):
        super(Masked_BCELoss, self).__init__()
        self.criterion = nn.BCELoss(reduce=False)

    def forward(self,pred, label, pairwise_mask, vertex_mask, seq_mask):
        batch_size = pred.size(0)
        loss_all = self.criterion(pred, label)
        loss_mask = torch.matmul(vertex_mask.view(batch_size, -1, 1),
                                 seq_mask.view(batch_size, 1, -1)) * pairwise_mask.view(-1, 1, 1)
        loss = torch.sum(loss_all * loss_mask) / torch.sum(pairwise_mask).clamp(min=1e-10)
        return loss


# Custom loss-1: inter
class Masked_MSELoss(nn.Module):
    def __init__(self):
        super(Masked_MSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduce=False)

    def forward(self,pred, label, inter_mask):
        batch_size = pred.size(0)
        loss_all = self.criterion(pred, label)
        loss = 1.0 * torch.sum(loss_all * inter_mask) / batch_size
        return loss


class MnistLossFunc(ABC):
    @abstractmethod
    def get_raw_loss(self, output: Tensor, labels: Tensor, original: Tensor) -> Tensor:
        """Return the unweighted loss, where the implementing class defines the exact loss function.

        :param output of the model
        :param labels for classification losses
        :param original The original image, for reconstruction losses
        """
        pass

    @abstractmethod
    def weight_loss(self, loss: Tensor) -> Tensor:
        """Weights the given loss appropriately. e.g. by a fixed weight or a learned uncertainty weight"""
        pass

# matrix
class FixedCELoss(MnistLossFunc):
    def __init__(self, weight: float):
        assert isinstance(weight, float)
        self._weight = weight

    def get_raw_loss(self, output: Tensor, labels: Tensor, pairwise_mask, x_mask, ff_mask,inter_mask) -> Tensor:
        loss_function1 = Masked_BCELoss()
        return loss_function1(output, labels, pairwise_mask, x_mask, ff_mask)

    def weight_loss(self, loss: Tensor) -> Tensor:
        return self._weight * loss


class LearnedCELoss(MnistLossFunc):
    def __init__(self, s: nn.Parameter):
        assert isinstance(s, nn.Parameter)
        self._s = s

    def get_raw_loss(self, output: Tensor, labels: Tensor, pairwise_mask,x_mask,ff_mask,inter_mask) -> Tensor:
        loss_function1 = Masked_BCELoss()
        return loss_function1(output, labels,pairwise_mask,x_mask,ff_mask)

    def weight_loss(self, loss: Tensor) -> Tensor:
        # return torch.exp(-self._s) * loss + 0.5 * self._s
        return 1.0 / (self._s ** 2) * loss + torch.log(1 + self._s ** 2)

# aff
class FixedMSELoss(MnistLossFunc):
    def __init__(self, weight: float):
        assert isinstance(weight, float)
        self._weight = weight

    def get_raw_loss(self, output: Tensor, original: Tensor,pairwise_mask,x_mask,ff_mask,inter_mask) -> Tensor:
        loss_function2 = nn.MSELoss()
        return loss_function2(output, original)

    def weight_loss(self, loss: Tensor) -> Tensor:
        return self._weight * loss


class LearnedMSELoss(MnistLossFunc):  
    def __init__(self, s: nn.Parameter):
        assert isinstance(s, nn.Parameter)
        self._s = s

    def get_raw_loss(self, output: Tensor, original: Tensor,pairwise_mask,x_mask,ff_mask,inter_mask) -> Tensor:
        loss_function2 = nn.MSELoss()
        # return torch.sqrt(loss_function2(output, original))  # RMSE
        return loss_function2(output, original)

    def weight_loss(self, loss: Tensor) -> Tensor:
        # return 0.5 * torch.exp(-self._s) * loss + 0.5 * self._s
        return 0.5 / (self._s ** 2) * loss + torch.log(1 + self._s ** 2)

# Pearson
class FixedPearson(MnistLossFunc):
    def __init__(self, weight: float):
        assert isinstance(weight, float)
        self._weight = weight

    def get_raw_loss(self, aff_pred: Tensor, aff_val: Tensor,pairwise_mask,x_mask,ff_mask,inter_mask) -> Tensor:
        vx = aff_pred - torch.mean(aff_pred)
        vy = aff_val.view(-1, 1) - torch.mean(aff_val.view(-1, 1))
        pearson_s = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return 1 - pearson_s

    def weight_loss(self, loss: Tensor) -> Tensor:
        return self._weight * loss


class LearnedPearson(MnistLossFunc):
    def __init__(self, s: nn.Parameter):
        assert isinstance(s, nn.Parameter)
        self._s = s

    def get_raw_loss(self, aff_pred: Tensor, aff_val: Tensor,pairwise_mask,x_mask,ff_mask,inter_mask) -> Tensor:
        vx = aff_pred - torch.mean(aff_pred)
        vy = aff_val.view(-1, 1) - torch.mean(aff_val.view(-1, 1))
        pearson_s = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return 5 * (1 - pearson_s)

    def weight_loss(self, loss: Tensor) -> Tensor:
        return 0.5 * torch.exp(-self._s) * loss + 0.5 * self._s
# inter
class FixedMasked_MseLoss(MnistLossFunc):
    def __init__(self, weight: float):
        assert isinstance(weight, float)
        self._weight = weight

    def get_raw_loss(self, output: Tensor, original: Tensor,pairwise_mask,x_mask,ff_mask,inter_mask) -> Tensor:
        loss_function3 = Masked_MSELoss()
        return loss_function3(output, original,inter_mask)

    def weight_loss(self, loss: Tensor) -> Tensor:
        return self._weight * loss


class LearnedMasked_MseLoss(MnistLossFunc):
    def __init__(self, s: nn.Parameter):
        assert isinstance(s, nn.Parameter)
        self._s = s

    def get_raw_loss(self, output: Tensor, original: Tensor,pairwise_mask,x_mask,ff_mask,inter_mask) -> Tensor:
        loss_function3 = Masked_MSELoss()
        return loss_function3(output, original,inter_mask)

    def weight_loss(self, loss: Tensor) -> Tensor:
        # return 0.5 * torch.exp(-self._s) * loss + 0.5 * self._s
        return 0.5 / (self._s ** 2) * loss + torch.log(1 + self._s ** 2)




class MultitaskMnistLoss(ABC):
    def __init__(self, enabled_tasks: [bool], loss_funcs: [MnistLossFunc]):
        super().__init__()
        assert len(enabled_tasks) == len(loss_funcs), f'enabled_tasks={enabled_tasks}, loss_funcs={loss_funcs}'
        self._enabled_tasks = enabled_tasks
        self._loss_funcs = loss_funcs

    def __call__(self, outputs: [Tensor], labels: [Tensor],pairwise_mask,x_mask,ff_mask,inter_mask):
        """Returns (overall loss, [task losses])"""
        assert len(outputs) == len(self._enabled_tasks) == len(self._loss_funcs) == len(labels)

        raw_losses = [
            loss_func.get_raw_loss(output, label,pairwise_mask,x_mask,ff_mask,inter_mask) if enabled else torch.tensor([0.0], device=output.device)
            for enabled, loss_func, output, label in zip(self._enabled_tasks, self._loss_funcs, outputs,labels)]

        weighted_losses = [loss_func.weight_loss(raw_loss) for loss_func, raw_loss in zip(self._loss_funcs, raw_losses)]
        total_loss = weighted_losses[0] + weighted_losses[1] + weighted_losses[2] ###3task
        #print("weighted loss:",weighted_losses[0], weighted_losses[1], weighted_losses[2], weighted_losses[3])

        #if weighted_losses[0]< 0 or weighted_losses[1]<0 or  weighted_losses[2]<0 :
        #   print("weighted loss < 0 :",weighted_losses[0].item(), weighted_losses[1].item(), weighted_losses[2].item())
        return total_loss, (raw_losses[0], raw_losses[1], raw_losses[2]) ###3task


def get_fixed_loss(enabled_tasks: [bool], weights: [float], mnist_type: str):
    """Returns the fixed weight loss function."""

    task_1_loss_func = FixedMasked_MseLoss(weights[2])
    task_2_loss_func = FixedMSELoss(weights[0])
    task_3_loss_func = FixedCELoss(weights[1])
    return MultitaskMnistLoss(enabled_tasks, [task_1_loss_func, task_2_loss_func, task_3_loss_func])


def get_learned_loss(enabled_tasks: [bool], ses: [nn.Parameter],mnist_type: str):
    """Returns the learned uncertainties loss function.

    :param ses s=log(sigma^2) for each task, as in the paper
    """
    # task_1_loss_func = LearnedCELoss(ses[0])
    task_1_loss_func = LearnedMasked_MseLoss(ses[0])
    task_2_loss_func = LearnedMSELoss(ses[1])
    task_3_loss_func=   LearnedCELoss(ses[2])
    return MultitaskMnistLoss(enabled_tasks, [task_1_loss_func, task_2_loss_func,task_3_loss_func])   ###3task
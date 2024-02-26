"""This module consists of several algortihms for combining potentially conflicting gradients."""
import random
from typing import Dict, Optional

import torch

from enums.aggregation_method import AggregationMethod
from training.model.helper_classes import RunningSumAccumulator, StackedAccumulator


class GradientAggregator:
    """Aggregator class for combining possibly conflicting gradients into one "optimal" grad."""

    def __init__(self, aggregation_method: AggregationMethod = AggregationMethod.MEAN):
        """Initialize GradientAggregator."""
        self.aggregation_method = aggregation_method
        self.accumulator = (
            RunningSumAccumulator() if aggregation_method == AggregationMethod.MEAN else StackedAccumulator()
        )
        self._conflicting_gradient_count = 0
        self._nonconflicting_gradient_count = 0

    def reset_accumulator(self) -> None:
        """Reset the accumulator."""
        self.accumulator = (
            RunningSumAccumulator() if self.aggregation_method == AggregationMethod.MEAN else StackedAccumulator()
        )

    def find_nonconflicting_grad(self, grad_tensor: torch.tensor) -> torch.tensor:
        """Use on of the algorithms to find a nonconflicting gradient."""
        if self.aggregation_method == AggregationMethod.PCGRAD:
            return self.pcgrad(grad_tensor).mean(dim=0)

        elif self.aggregation_method == AggregationMethod.PCGRAD_ONLINE:
            assert len(grad_tensor) == 2
            return self.pcgrad_online(grad_tensor)

        else:
            raise Exception

    def aggregate_gradients(self) -> torch.tensor:
        """Aggregate possibly conflicting set of gradients (given as a list of dictionaries).

        This method is called only after all tasks were processed.
        """
        conflicting_grads = self.accumulator.get_gradients()
        length = len(conflicting_grads[list(conflicting_grads.keys())[0]])

        if (
            self.aggregation_method == AggregationMethod.PCGRAD_ONLINE
            or self.aggregation_method == AggregationMethod.MEAN
        ):
            assert length == 1
            return self.accumulator.get_avg_gradients()

        elif self.aggregation_method == AggregationMethod.PCGRAD:
            conflicting_grads = [{k: v[i, ...] for k, v in conflicting_grads.items()} for i in range(length)]
            final_grad: Dict[str, torch.Tensor] = {}

            # single grad cannot conflict
            if len(conflicting_grads) == 1:
                return conflicting_grads[0]

            keys = list(conflicting_grads[0].keys())
            # update layer-wise the final_grad dictionary
            for layer_key in keys:
                list_of_st_grads = [st_grad[layer_key] for st_grad in conflicting_grads]
                final_grad.update({layer_key: self.find_nonconflicting_grad(torch.stack(list_of_st_grads, dim=0))})

            return final_grad

        else:
            raise Exception

    def pcgrad(self, grad_tensor: torch.tensor) -> torch.tensor:
        """Project conflicting gradients onto orthogonal plane.

        Algorithm from paper Gradient surgery for Multi-Task learning.
        """
        pc_grads, num_of_tasks = grad_tensor.clone(), len(grad_tensor)
        original_shape = grad_tensor.shape
        # flatten
        pc_grads = pc_grads.view(num_of_tasks, -1)
        grad_tensor = grad_tensor.view(num_of_tasks, -1)

        for g_i in range(num_of_tasks):
            task_index = list(range(num_of_tasks))
            random.shuffle(task_index)
            for g_j in task_index:
                dot_product = pc_grads[g_i].dot(grad_tensor[g_j])
                if dot_product < 0:  # conflict
                    pc_grads[g_i] -= (dot_product / (grad_tensor[g_j].norm() ** 2)) * grad_tensor[g_j]  # project
                    self._conflicting_gradient_count += 1
                else:
                    self._nonconflicting_gradient_count += 1
        return pc_grads.view(original_shape)

    def pcgrad_online(self, grad_tensor: torch.tensor) -> torch.tensor:
        """Perform pcgrad (online) algorithm.

        This algorithm is called every time we compute a new set of gradients (after each sub-batch is processed).
        Therefore, the grad_list is of length 1 or 2 only.
        """
        assert len(grad_tensor) == 2
        p = grad_tensor[0]
        g = grad_tensor[-1]

        p = p.view(-1)
        g = g.view(-1)

        dot_product = p.dot(g)
        if dot_product < 0:
            p = p - (dot_product / (g.norm() ** 2)) * g  # deconflict p and g
            self._conflicting_gradient_count += 1
        else:
            self._nonconflicting_gradient_count += 1

        p += g
        return p.view(grad_tensor[0].shape)

    def aggregate_gradients_online(self) -> Dict[str, torch.tensor]:
        """Aggregate the current overall gradient (stored in accumulator) with a (possibly conflicting) new gradient.

        This method is called by pcgrad_online only.
        Therefore, we have at most 2 gradients (the current overall gradient & a (possibly conflicting) new gradient).
        """
        conflicting_grads = self.accumulator.get_gradients()
        length = len(conflicting_grads[list(conflicting_grads.keys())[0]])
        conflicting_grads = [{k: v[i, ...] for k, v in conflicting_grads.items()} for i in range(length)]
        current_overall_grad: Dict[str, torch.Tensor] = {}

        # single grad cannot conflict
        if length == 1:
            return conflicting_grads[0]
        elif length == 2:
            keys = list(conflicting_grads[0].keys())
            # update layer-wise the final_grad dictionary
            for layer_key in keys:
                list_of_st_grads = [st_grad[layer_key] for st_grad in conflicting_grads]
                current_overall_grad.update(
                    {layer_key: self.find_nonconflicting_grad(torch.stack(list_of_st_grads, dim=0))}
                )
            return current_overall_grad
        else:
            raise Exception

    def update(self, gradients: Dict[str, torch.tensor], scaling_weight: float) -> None:
        """
        Update the gradients of the accumulator.

        Add the new gradient to the accumulator.
        If we use an online deconflicting method, here is where we call it.
        """
        self.accumulator.update(gradients=gradients, weight=scaling_weight)
        if self.aggregation_method == AggregationMethod.PCGRAD_ONLINE:
            self.accumulator.set_gradients(gradients=self.aggregate_gradients_online())

    def get_conflicting_gradients_ratio(self) -> Optional[float]:
        """Get the ratio of conflicting gradients.

        To ensure that this aggregation_method get called if and only if we used deconflicting methods (e.g. pcgrad), it
        raises an exception if no conflicting or nonflicting gradients were counted.
        """
        if self.aggregation_method == AggregationMethod.MEAN:
            raise Exception
        if self._conflicting_gradient_count + self._nonconflicting_gradient_count == 0:
            raise Exception
        return self._conflicting_gradient_count / (
            self._conflicting_gradient_count + self._nonconflicting_gradient_count
        )

from torch import nn
import torch.nn.functional as F


class CheckMonotonicityModule(nn.Module):
    def __init__(
        self,
        layer, 
        layer_idx, 
        lm_head, 
        final_layer_norm,
        plot_monotonicity
    ):
        super().__init__()

        self.layer = layer
        self.layer_idx = layer_idx
        self.lm_head = lm_head
        self.final_layer_norm = final_layer_norm
        self.plot_monotonicity = plot_monotonicity
        self.sentence = None

    def forward(self, *args, **kwargs):
        outputs = self.layer(
            *args,
            **kwargs
        )

        hidden_states = outputs[0]

        # at each layer, take the three most likely tokens
        # plot them with three points at that x=layer_idx
        # connect points across different x if it's the same
        # token

        # other things to plot:
        # - fraction of tokens that do not change after the first/
        #   second/third/etc layer
        # - fraction of tokens that decrease in prob again

        probs = F.softmax(
            self.lm_head(
                self.final_layer_norm(
                    hidden_states
                )
            ), dim=-1
        ).squeeze()

        most_likely_tokens = probs.topk(3, dim=-1)

        self.plot_monotonicity.save_most_likely_tokens(most_likely_tokens)

        if self.layer_idx == 11:
            self.plot_monotonicity.plot_most_likely_tokens()

        return outputs

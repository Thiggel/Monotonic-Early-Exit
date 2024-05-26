import numpy as np
import torch

from transformers import AutoConfig


def recurrent_classifier(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    all_hidden_states: list[torch.Tensor] = None,
    all_softmax_values: list[torch.Tensor] = None,
    layer_index: int = None,
    should_reset: bool = False,
    threshold: float = None,
    cache: dict = None,
):
    assert hidden_states is not None
    assert classifier is not None
    
    if layer_index == 1 or should_reset:
        classifier.reset()

    try:
        preds = classifier(hidden_states)
    except Exception as e:
        classifier.reset()
        preds = classifier(hidden_states)

    probs = torch.softmax(preds, dim=-1)
    return probs[..., 1].squeeze(), None


def last_three_hiddens_classifier(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    all_hidden_states: list[torch.Tensor] = None,
    all_softmax_values: list[torch.Tensor] = None,
    layer_index: int = None,
    should_reset: bool = False,
    threshold: float = None,
    cache: dict = None,
):
    assert classifier is not None

    if all_hidden_states is None or len(all_hidden_states) < 3:
        return torch.zeros(hidden_states.shape[0])

    last_three_hiddens = torch.cat(all_hidden_states[-3:], dim=2)

    preds = classifier(last_three_hiddens)
    probs = torch.softmax(preds, dim=-1)
    return probs[..., 1].squeeze(), None

def hidden_state_saturation(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    all_hidden_states: list[torch.Tensor] = None,
    all_softmax_values: list[torch.Tensor] = None,
    layer_index: int = None,
    should_reset: bool = False,
    cache: dict = None,
):
    if all_hidden_states is None or len(all_hidden_states) < 2:
        return torch.zeros(hidden_states.shape[0])

    last_hidden = all_hidden_states[-1]
    second_last_hidden = all_hidden_states[-2]

    similarity = torch.nn.functional.cosine_similarity(last_hidden, second_last_hidden, dim=2).squeeze()

    return similarity


def last_three_top_prob_heuristic(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    all_hidden_states: list[torch.Tensor] = None,
    all_softmax_values: list[torch.Tensor] = None,
    layer_index: int = None,
    should_reset: bool = False,
    threshold: float = None,
    cache: dict = None,
):
    print(hidden_states.shape[0])
    if all_softmax_values is None:
        return torch.zeros(hidden_states.shape[0], device=hidden_states.device), cache

    if layer_index == 0 or cache is None:
        cache = {
            'last_top_probs': [],
            'increasing': torch.zeros(hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device)
        }

    current_top_prob = all_softmax_values[-1].max(dim=-1)[0]
    cache['last_top_probs'].append(current_top_prob)

    if len(cache['last_top_probs']) > 2:
        cache['last_top_probs'].pop(0)

    if len(cache['last_top_probs']) > 1:
        # Compare the last two top probabilities
        previous_top_prob = cache['last_top_probs'][-2]
        current_increasing = current_top_prob > previous_top_prob
        # Element-wise 'and' across batches
        increasing_for_3_layers = current_increasing & cache['increasing']
        cache['increasing'] = current_increasing

    else:
        increasing_for_3_layers = torch.zeros(hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device)

    if layer_index > 2:
        confidence = increasing_for_3_layers & (current_top_prob > threshold)
        if confidence.any():
            print("Early exit at layer:", layer_index)
            print("Increasing condition across three layers:", increasing_for_3_layers)
            print("Current top probabilities above threshold:", current_top_prob)
        else:
            if layer_index > 22:
                print("No early exit at layer:", layer_index)
                print("Increasing condition failed or threshold not met")
                print("Threshold:", threshold)
                print("Current top probabilities:", current_top_prob)
    else:
        confidence = torch.zeros(hidden_states.shape[0], dtype=torch.float, device=hidden_states.device)

    return confidence, cache

def softmax_confidence(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    all_hidden_states: list[torch.Tensor] = None,
    all_softmax_values: list[torch.Tensor] = None,
    layer_index: int = None,
    should_reset: bool = False,
    threshold: float = None,
    cache: dict = None,
):
    assert logits is not None
    probs = torch.softmax(logits, dim=-1)
    top_2 = torch.topk(probs, dim=-1, k=2)[0]

    conf = (top_2[..., 0] - top_2[..., 1]).squeeze()

    return conf, None


def meta_confidence(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    all_hidden_states: list[torch.Tensor] = None,
    all_softmax_values: list[torch.Tensor] = None,
    layer_index: int = None,
    should_reset: bool = False,
    threshold: float = None,
    cache: dict = None,
):
    assert hidden_states is not None
    assert classifier is not None
    
    preds = classifier(hidden_states)
    probs = torch.softmax(preds, dim=-1)
    return probs[..., 1].squeeze(),None

def get_confidence_class(key):

    _conf_class_map = {
        'softmax': softmax_confidence,
        'meta': meta_confidence,
        'recurrent_classifier': recurrent_classifier,
        'last_three_hiddens_classifier': last_three_hiddens_classifier,
        'last_three_top_prob_heuristic': last_three_top_prob_heuristic,
        'hidden_state_saturation': hidden_state_saturation,
    }

    if key in _conf_class_map:
        return _conf_class_map[key]
    else:
        raise ValueError('Invalid confidence measure: {}'.format(key))


def get_skip_mask(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    config: AutoConfig = None,
    pos_time: int = 1,
    adapt_threshold: float = None,
    return_conf=False,
    all_hidden_states: list[torch.Tensor] = None,
    all_softmax_values: list[torch.Tensor] = None,
    layer_index: int = None,
    should_reset: bool = False,
    cache: dict = None,
):
    assert config.exit_conf_type is not None or config.shallow2deep_conf_type is not None

    if config.exit_conf_type is not None:
        key = config.exit_conf_type
        if config.exit_position_temp is not None:
            # decays the confidence threshold with decoding time stp.        
            correct_by_pos = lambda i: config.exit_conf_threshold * np.exp(
                - config.exit_position_temp * i / config.max_answer_length
            ) / 10 + 9 * config.exit_conf_threshold / 10
            threshold = correct_by_pos(pos_time)
        else:
            threshold = config.exit_conf_threshold
    elif config.shallow2deep_conf_type is not None:
        key = config.shallow2deep_conf_type
        threshold = config.shallow2deep_conf_threshold if adapt_threshold is None else adapt_threshold

    conf_measure = get_confidence_class(key=key)    
    conf, cache = conf_measure(
        logits=logits, 
        hidden_states=hidden_states, 
        classifier=classifier,
        all_hidden_states=all_hidden_states,
        all_softmax_values=all_softmax_values,
        layer_index=layer_index,
        should_reset=should_reset,
        threshold=threshold,
        cache=cache
    )
    mask = torch.where(conf <= threshold, 0., 1.).bool()

    if cache is not None:
        return mask, cache
    if not return_conf:
        return mask  # Return the whole mask tensor
    else:
        return mask, conf

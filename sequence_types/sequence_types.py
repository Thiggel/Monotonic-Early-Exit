import torch
from tqdm.auto import tqdm
import numpy as np


def get_first_similar_state(similar_states_mask):
    """Get the index of the first similar state for each token."""
    idx = torch.arange(similar_states_mask.shape[1], 0, -1)
    return torch.argmax(similar_states_mask * idx, 1, keepdim=True)


def compute_hidden_state_similarities(
    dataset, trainer, tokenizer, max_examples=500, max_sequence_length=30
):
    """Generate sequences using the model and compute the hidden states similarities."""
    eval_dataloader = trainer.get_eval_dataloader(dataset)
    results = {}

    for example_id, batch in tqdm(enumerate(eval_dataloader)):
        
        past_key_values = None
        if example_id >= max_examples:
            break

        decoder_input_ids_list = []
        generated_ids = []
        hidden_satate_similaritites = []

        # Forward pass through the encoder
        encoder_outputs = trainer.model.encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        for n, decoder_token in enumerate(batch["decoder_input_ids"][0]):
            decoder_input_ids_list.append(decoder_token)

            # Stop generating if we reach the max sequence length or the EOS token
            if (
                n > max_sequence_length
                or decoder_token.item() == tokenizer.eos_token_id
            ):
                break

            # Forward pass
            with torch.no_grad():
                decoder_input_ids = torch.tensor(decoder_input_ids_list).unsqueeze(0)
                model_output = trainer.model(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            past_key_values = model_output.past_key_values

            # Get the generated token
            generated_id = model_output.logits[0, -1, :].argmax().item()

            # Do not include the EOS token in the generated sequence
            # and stop generating if we reach the EOS token
            if generated_id == tokenizer.eos_token_id:
                break
            generated_ids.append(generated_id)
            
            # Exctract the hiddens states of the last token and drop the embedding layer hidden state
            decoder_hidden_states = [h[:, -1, :] for h in model_output.decoder_hidden_states]
            decoder_hidden_states = decoder_hidden_states[1:]

            # Compute the cosine similarity between the last and all the previous hidden states
            h = torch.nn.functional.normalize(torch.cat(decoder_hidden_states), dim=1)
            hidden_satate_similaritites.append((h @ h.T)[-1][:-1].unsqueeze(0))

        if not hidden_satate_similaritites:
            continue
        # Concatenate all the hidden states similarities in a tensor
        hidden_satate_similaritites = torch.cat(hidden_satate_similaritites, axis=0)

        # Find state similarities bigger than a threshold
        similar_states_mask = torch.where(hidden_satate_similaritites >= 0.9, 1, 0)

        # Save the results
        results[example_id] = {
            "hidden_satate_similaritites": hidden_satate_similaritites,
            "generated_ids": generated_ids,
            "decoder_input_ids": torch.tensor(decoder_input_ids_list),
            "generated_answer": tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ),
            "n_similar_states": similar_states_mask.sum(axis=1),
            "mean_similarity": hidden_satate_similaritites.mean(axis=1),
            "first_similar_state": get_first_similar_state(similar_states_mask),
            "actual_answer": tokenizer.decode(
                batch["decoder_input_ids"][0], skip_special_tokens=True
            ),
            "strictly_monotonic": check_strict_monotonicity(hidden_satate_similaritites),
            "strictly_monotonic_after_4": check_strict_monotonicity(hidden_satate_similaritites, 4),
            "strictly_monotonic_after_8": check_strict_monotonicity(hidden_satate_similaritites, 8),
        }
    return results


def check_strict_monotonicity(hidden_state_similarities, start_layer=0):
    """Check if the hidden states similarities are strictly monotonic."""
    h = hidden_state_similarities[:, start_layer:]
    first_order_diff = torch.diff(h, prepend=torch.zeros(h.shape[0], 1))
    strictly_monotonic = (first_order_diff > 0).count_nonzero(dim=1) == h.shape[-1]
    return strictly_monotonic


def find_sequence_types(results, tokenizer, topk=10):
    """Find the easy and hard sequences."""
    
    easy_sequences = []
    hard_sequences = []

    # Find the examples with the highest and lowest mean similarity
    max_means = torch.tensor([results[i]["mean_similarity"].max().item() for i in results])
    argmax_means = torch.tensor([results[i]["mean_similarity"].argmax().item() for i in results])
    min_means = torch.tensor([results[i]["mean_similarity"].min().item() for i in results])
    argmin_means = torch.tensor([results[i]["mean_similarity"].argmin().item() for i in results])

    # Find the examples the highest and lowest mean similarity
    easy_token_examples = torch.argsort(max_means, descending=True)[:topk]
    hard_token_examples = torch.argsort(min_means)[:topk]

    easy_token_tuples = [(i, argmax_means[i].item()) for i in easy_token_examples.tolist()]
    hard_token_tuples = [(i, argmin_means[i].item()) for i in hard_token_examples.tolist()]

    # Easy and hard tokens
    easy_tokens = [
        tokenizer.decode(results[e]["decoder_input_ids"][i+1])
        for e, i in easy_token_tuples
    ]
    hard_tokens = [
        tokenizer.decode(results[e]["decoder_input_ids"][i+1])
        for e, i in hard_token_tuples
    ]

    # Find the sequences with the highest and lowest number of similar states
    easy_seq = tokenizer.batch_decode(
        [
            results[e]["decoder_input_ids"][:i+1]
            for e, i in easy_token_tuples
        ]
    )
    hard_seq = tokenizer.batch_decode(
        [
            results[e]["decoder_input_ids"][:i+1]
            for e, i in hard_token_tuples
        ]
    )

    for n, (e, i) in enumerate(easy_token_tuples):
        easy_sequences.append(
            {
                "example_id": e,
                "token_id": i,
                "hidden_state_similarities": results[e]['hidden_satate_similaritites'][i],
                "n_similar_states": results[e]["n_similar_states"][i].item(),
                "first_similar_state": results[e]["first_similar_state"][i].item(),
                "token_to_predict": easy_tokens[n],
                "input_sequence": easy_seq[n],
                "strictly_monotonic": results[e]["strictly_monotonic"][i].item(),
                "strictly_monotonic_after_4": results[e]["strictly_monotonic_after_4"][i].item(),
                "strictly_monotonic_after_8": results[e]["strictly_monotonic_after_8"][i].item(),
            }
        )

    for n, (e, i) in enumerate(hard_token_tuples):
        hard_sequences.append(
            {
                "example_id": e,
                "token_id": i,
                "hidden_state_similarities": results[e]['hidden_satate_similaritites'][i],
                "n_similar_states": results[e]["n_similar_states"][i].item(),
                "first_similar_state": results[e]["first_similar_state"][i].item(),
                "token_to_predict": hard_tokens[n],
                "input_sequence": hard_seq[n],
                "strictly_monotonic": results[e]["strictly_monotonic"][i].item(),
                "strictly_monotonic_after_4": results[e]["strictly_monotonic_after_4"][i].item(),
                "strictly_monotonic_after_8": results[e]["strictly_monotonic_after_8"][i].item(),
            }
        )

    return easy_sequences, hard_sequences

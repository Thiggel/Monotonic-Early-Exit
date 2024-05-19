import matplotlib.pyplot as plt


def plot_hidden_state_similarities(sequences, similarity_thereshold=0.9):
    # Plot the hidden state similarities
    plt.figure(figsize=(15, 5))
    plt.axhline(similarity_thereshold, color="r", linestyle="--")
    for sequence in sequences: 
        plt.plot(sequence["hidden_state_similarities"], label=f"{sequence['input_sequence']} -> {sequence['token_to_predict']}")
    plt.legend()
    plt.xlabel("Layer")
    plt.ylabel("Similarity with the last hidden state")
    plt.title("Similarities between the last and every other hidden state")

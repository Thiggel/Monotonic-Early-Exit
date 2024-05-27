import matplotlib.pyplot as plt


def plot_hidden_state_similarities(sequences, similarity_thereshold=0.9, save_path=None, colors=None, save_format='png'):
    # Plot the hidden state similarities
    figure = plt.gcf()
    ax = figure.gca()

    # Set figsize
    figure.set_figheight(5)
    figure.set_figwidth(15)

    dark_grey = '#202020'

    # Customize the color of the ticks
    ax.tick_params(axis='x', colors=dark_grey)
    ax.tick_params(axis='y', colors=dark_grey)

    # Plot the lines
    ax.axhline(similarity_thereshold, color="r", linestyle="--", alpha=0.5)
    if colors is None:
        colors = [None] * len(sequences)
    for color, sequence in zip(colors, sequences):
        ax.plot(
            sequence["hidden_state_similarities"],
            label=f"{sequence['input_sequence']} -> {sequence['token_to_predict']}",
            color=color,
        )
    plt.legend()

    # Set font size of ticks
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Similarity with the last hidden state")

    # Set font size of labels
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

    # ax.set_title("Similarities between the last and every other hidden state")
    # Set title size
    ax.title.set_size(18)

    # Change text color
    for text in ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label, ax.title]:
        text.set_color(dark_grey)

    # Save figure
    if save_path:
        plt.savefig(save_path+f".{save_format}", format=save_format, bbox_inches='tight', dpi=200)

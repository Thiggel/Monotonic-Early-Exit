import html
import re
from check_monotonicity.WordToColor import WordToColor


class PlotMonotonicity:
    def __init__(
        self,
        tokenizer,
        encoder_sentence_saver,
        decoder_sentence_saver,
    ):
        self.tokenizer = tokenizer
        self.encoder_sentence_saver = encoder_sentence_saver
        self.decoder_sentence_saver = decoder_sentence_saver
        self.most_likely_tokens = []

    def save_most_likely_tokens(self, most_likely_tokens):
        self.most_likely_tokens.append(most_likely_tokens)

    def plot_most_likely_tokens(self):
        # at each layer, take the three most likely tokens
        # plot them with three points at that x=layer_idx
        # connect points across different x if it's the same
        # token
        batch_size = len(self.most_likely_tokens[0].indices)
        
        x_values = [[] for _ in range(batch_size)]
        y_values = [[] for _ in range(batch_size)]
        labels = [[] for _ in range(batch_size)]
        titles = [None for _ in range(batch_size)]

        for step, most_likely_tokens in enumerate(self.most_likely_tokens):

            for sample_idx, (token_indices, token_probs) in enumerate(zip(
                most_likely_tokens.indices,
                most_likely_tokens.values
            )):
                for idx, token in enumerate(token_indices):
                    word = self.tokenizer.decode(token)
                    prob = token_probs[idx]

                    encoder_tokens = self.encoder_sentence_saver.sentence[
                        sample_idx
                    ]
                    decoder_tokens = self.decoder_sentence_saver.sentence[
                        sample_idx
                    ]
                    tokens = encoder_tokens + decoder_tokens
                    prefix = '...' if len(tokens) > 20 else ''
                    sentence = prefix + self.tokenizer.decode(tokens[-40:])

                    titles[sample_idx] = html.unescape(
                        re.sub(r'<[^>]*?>', '', sentence)
                    )

                    # Append x, y, and label values
                    if prob > 0.1:
                        x_values[sample_idx].append(step)  # Layer index as x-value
                        y_values[sample_idx].append(prob)  # Probability as y-value
                        labels[sample_idx].append(word)  # Token as label

        word_to_color = WordToColor()

        for x, y, labs, title in zip(x_values, y_values, labels, titles):
            plt.figure(figsize=(10, 6))  # Adjust figure size as needed
            # Annotate each point with its label
            for i, label in enumerate(labs):
                color = word_to_color.get_color(label)
                # find index of the previous point with the same label
                is_new_label = True
                for j in range(i - 1, -1, -1):
                    if labs[j] == label:
                        is_new_label = False
                        # Connect points with the same label
                        plt.plot(
                            [x[j], x[i]],
                            [y[j], y[i]],
                            c=color
                        )
                        break

                if is_new_label:
                    plt.annotate(
                        label,
                        (x[i], y[i] + 0.02),
                        ha='center',
                        va='bottom',
                    )

                plt.scatter(x[i], y[i], c=color)

            plt.xlabel('Layer Index')
            plt.ylabel('Probability')
            plt.title(title, wrap=True)

            plt.ylim(0, 1.25)
            # don't show y-axis ticks above 1.0
            plt.yticks(np.arange(0, 1.1, 0.5))

            plt.show()

        self.most_likely_tokens = []

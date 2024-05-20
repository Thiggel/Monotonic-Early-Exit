import torch
import html
import numpy as np
import re
from check_monotonicity.WordToColor import WordToColor
from transformers import PreTrainedTokenizerBase
import matplotlib.pyplot as plt
import os
from .SentenceSaver import SentenceSaver
from .RecordMonotonicity import RecordMonotonicity


class PlotMonotonicity:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        encoder_sentence_saver: SentenceSaver,
        decoder_sentence_saver: SentenceSaver,
        record_monotonicity: RecordMonotonicity,
        model_name_or_path: str,
        do_plot: bool = True
    ):
        """
        Args:
            tokenizer: The tokenizer
            encoder_sentence_saver: The SentenceSaver for the encoder
            decoder_sentence_saver: The SentenceSaver for the decoder
            record_monotonicity: The RecordMonotonicity instance
            model_name_or_path: The model name or path
            do_plot: Whether to plot the monotonicity using matplotlib or just
            record the monotonicity for a table
        """
        self.tokenizer = tokenizer
        self.encoder_sentence_saver = encoder_sentence_saver
        self.decoder_sentence_saver = decoder_sentence_saver
        self.record_monotonicity = record_monotonicity
        self.most_likely_tokens = []
        self.do_plot = do_plot

        self.directory = 'monotonicity_plots/' + model_name_or_path
        self.create_dir_if_not_exists(self.directory)

    def save_most_likely_tokens(self, most_likely_tokens):
        self.most_likely_tokens.append(most_likely_tokens)

    def create_dir_if_not_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def remove_non_letters(self,input_string):
        return re.sub(r'[^a-zA-Z]', '', input_string)

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
                    tokens = torch.cat([encoder_tokens, decoder_tokens]).tolist()

                    max_length = 20
                    prefix = '...' if len(tokens) > max_length else ''
                    sentence = prefix + self.tokenizer.decode(tokens[-max_length:])

                    titles[sample_idx] = html.unescape(
                        re.sub(r'<[^>]*?>', '', sentence)
                    )

                    prob = prob.item() if len(prob.shape) == 0 else prob[0].item()
                    # Append x, y, and label values
                    if prob > 0.1:
                        x_values[sample_idx].append(step)  # Layer index as x-value
                        y_values[sample_idx].append(prob)  # Probability as y-value
                        labels[sample_idx].append(word)  # Token as label

        word_to_color = WordToColor()

        for x, y, labs, title in zip(x_values, y_values, labels, titles):
            # Record the monotonicity of the tokens
            # for a nice table later
            self.record_monotonicity.record(x, y, labs)

            if not self.do_plot:
                continue

            plt.figure(figsize=(10, 6)) 
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
                        (x[i], y[i] - 0.03),
                        ha='center',
                        va='top',
                    )

                plt.scatter(x[i], y[i], c=color)

            plt.xlabel('Layer Index')
            plt.ylabel('Probability')
            plt.title(title, wrap=True, pad=20)

            plt.ylim(0, 1.25)
            # don't show y-axis ticks above 1.0
            plt.yticks(np.arange(0, 1.1, 0.5))

            fig = plt.gcf()
            ax = fig.gca()

            white_blue = '#F3F9F9'
            ax.set_facecolor(white_blue)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            dark_grey = '#202020'

            # Customize the color of the ticks
            ax.tick_params(axis='x', colors=dark_grey)
            ax.tick_params(axis='y', colors=dark_grey)

            for text in ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label, ax.title]:
                text.set_color(dark_grey)

            filename = self.remove_non_letters(title.lower())

            plt.savefig(
                f'{self.directory}/{filename}.pdf',
                format='pdf',
                bbox_inches='tight'
            )
            plt.close()

        self.most_likely_tokens = []

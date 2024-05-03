class RecordMonotonicity:
    def __init__(
        self,
        num_layers: int = 12,
    ):
        # The number of tokens encountered (to calculate fractions)
        self.num_tokens = 0

        # The number of tokens that do not change in probability
        # after each layer
        self.num_tokens_no_change = [0] * num_layers

        # The number of tokens that decrease in probability
        # after being at their maximum
        self.num_tokens_decrease = 0

        # The number of switches between first and second position
        # at each layer
        self.num_switches = [0] * num_layers

        self.num_layers = num_layers

    def print_results(self):
        print('Fraction of tokens that do not change in probability after each layer:')
        for layer_idx, num_no_change in enumerate(self.num_tokens_no_change):
            print(f'Layer {layer_idx}: {num_no_change / self.num_tokens:.2f}')

        print(f'Fraction of tokens that decrease in probability after being at their maximum: {self.num_tokens_decrease / self.num_tokens:.2f}')

        print('Fraction of switches between first and second position at each layer:')
        for layer_idx, num_switches in enumerate(self.num_switches):
            print(f'Layer {layer_idx}: {num_switches / self.num_tokens:.2f}')

    def record(
        self,
        x: list[int],
        y: list[float],
        labels: list[str]
    ):
        """
        Args:
            x: a list of x values (layer 0 through 11)
            y: a list of y values (probabilities)
            labels: a list of labels (tokens), one for each x, y pair
        """
        sorted_data = self.sort_scatter_data(x, y, labels)
        top1_labels = self.get_top1_labels(sorted_data)

        self.num_tokens += 1

        for label_idx in range(1, len(top1_labels)):
            if top1_labels[label_idx] != top1_labels[label_idx - 1]:
                self.num_switches[label_idx] += 1

        for label, y_values in sorted_data.items():
            has_decreased = False

            for x, y in enumerate(y_values):

                # Check if the token has not decreased in probability
                # until layer x
                if all(y >= y_prec for y_prec in y_values[:x]):
                    self.num_tokens_no_change[x] += 1

                # Check if the token has decreased in probability
                else:
                    has_decreased = True

            if has_decreased:
                self.num_tokens_decrease += 1

    def get_top1_labels(
        self,
        sorted_data: dict[str, list[float]]
    ) -> list[str]:
        """
        Args:
            sorted_data: a dictionary with unique labels as keys and a list of length 12
            with the corresponding y values as values
        Returns:
            a list of labels that are in the top 1 probability at each layer
        """
        top1_labels = []

        for x in range(12):
            top1_label = max(
                sorted_data.keys(),
                key=lambda label: sorted_data[label][x]
            )

            top1_labels.append(top1_label)

        return top1_labels


    def sort_scatter_data(
        self,
        x: list[int],
        y: list[float],
        labels: list[str]
    ) -> dict[str, list[float]]:
        """
        Args:
            x: a list of x values (layer 0 through 11)
            y: a list of y values (probabilities)
            labels: a list of labels (tokens), one for each x, y pair
        Returns:
            a dictionary with unique labels as keys and a list of length 12
            with the corresponding y values as values
        """
        label_to_y = {}

        for i, label in enumerate(labels):
            if label not in label_to_y:
                label_to_y[label] = [0.0] * 12

            label_to_y[label][x[i]] = y[i]

        return label_to_y

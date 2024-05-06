class RecordMonotonicity:
    def __init__(
        self,
        num_layers: int = 12,
    ):
        # The number of tokens encountered (to calculate fractions)
        self.num_tokens = 0

        # number of predicted tokens that are monotonically increasing
        # in probability after layer x
        self.num_mono_incr_probs_predicted = [0] * num_layers

        # number of tokens where the top1 prediction does not change after layer x
        self.num_tokens_no_change_top1 = [0] * num_layers

        self.num_layers = num_layers

    def print_results(self):
        print('Fraction of tokens that are monotonic in probability after layer x:')
        for layer_idx, num_monot in enumerate(self.num_mono_incr_probs_predicted):
            print(f'Layer {layer_idx}: {num_monot / self.num_tokens:.2f}')

        print()


        print('Fraction of tokens where the top1 prediction does not change after layer x:')
        for layer_idx, num_no_change in enumerate(self.num_tokens_no_change_top1):
            print(f'Layer {layer_idx}: {num_no_change / self.num_tokens:.2f}')

        print()

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
        
        def get_top1(labs):
            return labs[0] if len(labs) > 0 else None

        for label_idx in range(1, len(top1_labels)):
            label = get_top1(top1_labels[label_idx])
            if label != None and all([label == get_top1(top3) for top3 in top1_labels[label_idx:]]):
                self.num_tokens_no_change_top1[label_idx] += 1

        print()


        # get the predicted token (has max probability at last layer)
        max_label = max(sorted_data.keys(), key=lambda label: sorted_data[label][-1])
        
        # check whether the y_values for max_label are monotonically increasing
        def is_monotonically_increasing(list_of_values):
            return all(x <= y for x, y in zip(list_of_values, list_of_values[1:]))

        for layer_idx in range(self.num_layers):
            if is_monotonically_increasing(sorted_data[max_label][layer_idx:]):
                self.num_mono_incr_probs_predicted[layer_idx] += 1

        self.print_results()


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
            ) if sorted_data and len(sorted_data) > 0 else []

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

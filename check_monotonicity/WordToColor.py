class WordToColor:
    def __init__(self):
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.current_color = 0

        self.word_to_color = {}

    def get_color(self, word):
        if word not in self.word_to_color:
            self.word_to_color[word] = self.colors[self.current_color]
            self.current_color += 1
            self.current_color %= len(self.colors)

        return self.word_to_color[word]


class WordToColor:
    def __init__(self):
        self.colors = [
            '#7EAF4D',
            '#447512',
            '#0B6383',
            '#4793AF',
            '#FFC470',
            '#DD5746',
            '#8B322C',
        ]

        self.current_color = 0

        self.word_to_color = {}

    def get_color(self, word):
        if word not in self.word_to_color:
            self.word_to_color[word] = self.colors[self.current_color]
            self.current_color += 1
            self.current_color %= len(self.colors)

        return self.word_to_color[word]


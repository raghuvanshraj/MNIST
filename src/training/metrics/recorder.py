class MetricsRecorder(object):

    def __init__(self, name):
        self.name = name
        self.update_count = 0
        self.curr_loss = 0.
        self.curr_accuracy = 0.
        self.losses = []
        self.accuracies = []

    def record(self, loss, accuracy):
        self.curr_loss += loss
        self.curr_accuracy += accuracy
        self.update_count += 1

    def clear(self):
        self.update_count = 0.
        self.curr_loss = 0.
        self.curr_accuracy = 0.

    def update(self):
        self.losses.append(self.curr_loss / self.update_count)
        self.accuracies.append(self.curr_accuracy / self.update_count)
        self.clear()

    def print(self):
        print(f'{self.name}:\tloss: {self.losses[-1]}\t accuracy: {self.accuracies[-1]}')

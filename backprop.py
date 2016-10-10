class BPTTUpdater(training.StandardUpdater):
  def __init__(self, train_iter, optimizer, bprop_len, device):
    super(BPTTUpdater, self).__init__(train_iter, optimizer, device=device)
    self.bprop_len = bprop_len

  def update_core(self):
      loss = 0
      train_iter = self.get_iterator('main')
      optimizer = self.get_optimizer('main')

      for i in range(self.bprop_len):
        batch = train_iter.__next__()

        x, t = self.converter(batch, self.device)

        loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

      optimizer.target.zerograds()  # Initialize the parameter gradients
      loss.backward()  # Backprop
      loss.unchain_backward()  # Truncate the graph
      optimizer.update()  # Update the parameters

g = L.Linear(2, 1)

x = variable.Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = f(x)
z = g(y)

z.grad = np.ones((2, 1), dtype=np.float32)
x.grad
y.grad
f.W.grad
f.b.grad
g.W.grad
g.b.grad
f.cleargrads()
g.cleargrads()
z.backward()

x.grad
y.grad
f.W.grad
f.b.grad
g.W.grad
g.b.grad
#---------------------
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import variable

f = L.Linear(4, 3)
g = L.Linear(3, 2)
o = L.Linear(2, 1)

x = variable.Variable(np.array([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=np.float32))
y = f(x)
z = g(y)
t = o(z)

t.grad = np.ones((2, 1), dtype=np.float32)
x.grad
y.grad
z.grad
t.grad
f.W.grad
f.b.grad
g.W.grad
g.b.grad
o.W.grad
o.b.grad
#f.cleargrads()
#g.cleargrads()
o.cleargrads()
t.backward()

x.grad
y.grad
z.grad
t.grad
f.W.grad
f.b.grad
g.W.grad
g.b.grad
o.W.grad
o.b.grad
#------------------------
f = L.Linear(3, 2)

x = variable.Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = f(x)

y.grad = np.ones((2, 2), dtype=np.float32)
f.cleargrads()
y.backward()

x.__dict__
y.__dict__

#pattern2
g.cleargrads()
z.grad = np.ones((2, 1), dtype=np.float32)
z.backward()

x.__dict__
y.__dict__
z.__dict__


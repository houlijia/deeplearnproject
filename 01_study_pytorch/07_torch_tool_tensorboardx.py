import numpy as np
from tensorboardX import SummaryWriter
# create a summary writer
writer = SummaryWriter("log")

for i in range(100):
    writer.add_scalar("a", i,global_step=i)
    writer.add_scalar("b", i,global_step=i)

# close the writer
writer.close()
# view the log in tensorboard
# open a terminal and type: tensorboard --logdir ./log

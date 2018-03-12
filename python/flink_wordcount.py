from flink.plan.Environment import get_environment
from flink.functions.GroupReduceFunction import GroupReduceFunction

env = get_environment()
env.set_parallelism(4)
data = env.from_elements("Who's there?", "I think I hear them. Stand, ho! Who's there?")

class Adder(GroupReduceFunction):
  def reduce(self, iterator, collector):
    count, word = iterator.next()
    count += sum([x[0] for x in iterator])
    collector.collect((count, word))
    
data \
  .flat_map(lambda x, c: [(1, word) for word in x.lower().split()]) \
  .group_by(1) \
  .reduce_group(Adder(), combinable=True) \
  .output()

env.execute(local=True)

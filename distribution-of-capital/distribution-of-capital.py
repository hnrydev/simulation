import mesa

# Toy "wealth transfer" ABM: at each step, each person with positive wealth
# gives one unit to a randomly chosen other person. Total wealth is conserved
# (only moves between agents), but the distribution spreads — a simple
# illustration of how random exchanges can concentrate or spread holdings
# even without creating new units.


class Person(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.wealth = 1

    def step(self):
        if self.wealth > 0:
            others = [a for a in self.model.agents if a is not self]
            other = self.random.choice(others)
            other.wealth += 1
            self.wealth -= 1


class Economy(mesa.Model):
    def __init__(self, num_people=50):
        super().__init__()
        self.num_agents = num_people
        Person.create_agents(model=self, n=num_people)

    def step(self):
        # Randomize activation order each tick so no agent has a fixed advantage.
        self.agents.shuffle_do("step")


# Run it
model = Economy(100)
for _ in range(50):
    model.step()

wealths = [agent.wealth for agent in model.agents]
print("Average wealth:", sum(wealths) / len(wealths))
print("Richest person has:", max(wealths))

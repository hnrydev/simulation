import mesa

# Two-sided matching "market": n proposers and n receivers, each with a strict
# ranking over everyone on the other side. Gale-Shapley (men-optimal version
# here: side 0 proposes) yields a stable matching — no pair both prefer each
# other to their assigned partners. We report how well each side does on
# average in terms of rank of their assigned partner (1 = top choice).


def gale_shapley_men_optimal(men_prefs, women_prefs):
    """
    men_prefs[m]: list of woman indices for man m, best first.
    women_prefs[w]: list of man indices for woman w, best first.
    Returns dict mapping woman_index -> man_index (stable matching).
    """
    n = len(men_prefs)
    woman_rank = []
    for w in range(n):
        rank = {}
        for rank_pos, m in enumerate(women_prefs[w]):
            rank[m] = rank_pos
        woman_rank.append(rank)

    next_proposal = [0] * n
    free_men = list(range(n))
    woman_partner = [None] * n  # woman w matched to man index or None

    while free_men:
        m = free_men.pop()
        w = men_prefs[m][next_proposal[m]]
        next_proposal[m] += 1

        if woman_partner[w] is None:
            woman_partner[w] = m
        else:
            m_cur = woman_partner[w]
            # Lower rank index = more preferred
            if woman_rank[w][m] < woman_rank[w][m_cur]:
                woman_partner[w] = m
                free_men.append(m_cur)
            else:
                free_men.append(m)

    return {w: woman_partner[w] for w in range(n)}


class Dater(mesa.Agent):
    def __init__(self, model, side, idx):
        super().__init__(model)
        # side 0 = proposers (algorithm treats them as "men"), 1 = receivers ("women")
        self.side = side
        self.idx = idx
        self.prefs = []  # indices on the other side, best match first
        self.partner_idx = None


class DatingMarket(mesa.Model):
    def __init__(self, n_each=25, rng=None):
        super().__init__(rng=rng if rng is not None else 42)
        self.n = n_each
        sides = [0] * n_each + [1] * n_each
        idxs = list(range(n_each)) * 2
        Dater.create_agents(model=self, n=2 * n_each, side=sides, idx=idxs)

        proposers = sorted(
            [a for a in self.agents if a.side == 0], key=lambda a: a.idx
        )
        receivers = sorted(
            [a for a in self.agents if a.side == 1], key=lambda a: a.idx
        )

        opp_p = list(range(n_each))
        opp_r = list(range(n_each))
        for a in proposers:
            self.random.shuffle(opp_p)
            a.prefs = list(opp_p)
        for a in receivers:
            self.random.shuffle(opp_r)
            a.prefs = list(opp_r)

        men_prefs = [proposers[i].prefs for i in range(n_each)]
        women_prefs = [receivers[j].prefs for j in range(n_each)]
        match_w_to_m = gale_shapley_men_optimal(men_prefs, women_prefs)

        for w_idx, m_idx in match_w_to_m.items():
            proposers[m_idx].partner_idx = w_idx
            receivers[w_idx].partner_idx = m_idx

    def step(self):
        # One-shot matching is computed in __init__; no dynamics unless you extend.
        pass


def rank_of_partner(agent):
    """1-based rank of matched partner in this agent's preference list."""
    if agent.partner_idx is None:
        return None
    return agent.prefs.index(agent.partner_idx) + 1


model = DatingMarket(30, rng=42)
proposers = sorted([a for a in model.agents if a.side == 0], key=lambda a: a.idx)
receivers = sorted([a for a in model.agents if a.side == 1], key=lambda a: a.idx)
n = model.n

ranks_m = [rank_of_partner(a) for a in proposers]
ranks_w = [rank_of_partner(a) for a in receivers]

print("Stable matching (Gale-Shapley, proposers optimal).")
print(
    "Mean rank of partner for proposers (1 = top choice):",
    sum(ranks_m) / len(ranks_m),
)
print(
    "Mean rank of partner for receivers (1 = top choice):",
    sum(ranks_w) / len(ranks_w),
)

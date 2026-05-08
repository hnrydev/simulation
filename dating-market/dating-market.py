import math
from collections import deque

import mesa

# Stylized heterosexual dating market (ABM). Informed by common themes in mating
# research - e.g. parental-investment logic (Trivers, 1972) as one reason *on
# average* higher minimum investment in offspring is modeled as higher female
# choosiness in many species; human mate-preference *averages* in large
# surveys (e.g. Buss, 1989 and follow-ons) where men more heavily weight
# physical attractiveness and women more heavily weight status/resource cues
# *on average* - with huge within-sex variance, culture, cohort, and mating
# context not represented here.
#
# This is a toy model for intuition, not a claim that all individuals follow these
# rules, nor that the magnitudes are empirically calibrated. Prefer reading
# outputs as comparative statics ("when parameter X rises, what happens?")
# rather than as quantitative predictions about real dating markets.

# --- Gale-Shapley (full information, stable matching) -------------------------
# Real dating is search under incomplete information and noise; GS is kept as a
# benchmark: what happens if everyone knows everyone and runs optimal proposals.


def gale_shapley_men_optimal(men_prefs, women_prefs):
    """
    men_prefs[m]: woman indices for man m, best first.
    women_prefs[w]: man indices for woman w, best first.
    Returns dict woman_index -> man_index.
    """
    n = len(men_prefs)
    woman_rank = []
    for w in range(n):
        woman_rank.append({m: pos for pos, m in enumerate(women_prefs[w])})

    next_proposal = [0] * n
    free_men = deque(range(n))
    woman_partner = [None] * n

    while free_men:
        m = free_men.popleft()
        w = men_prefs[m][next_proposal[m]]
        next_proposal[m] += 1

        if woman_partner[w] is None:
            woman_partner[w] = m
        else:
            m_cur = woman_partner[w]
            if woman_rank[w][m] < woman_rank[w][m_cur]:
                woman_partner[w] = m
                free_men.append(m_cur)
            else:
                free_men.append(m)

    return {w: woman_partner[w] for w in range(n)}


# --- Trait-based utilities ----------------------------------------------------


def utility_man_rates_woman(_m, w, w_attr_m, w_res_m):
    """How much the man values this woman (linear in her traits; no idiosyncratic taste)."""
    return w_attr_m * w.attractiveness + w_res_m * w.resources


def utility_woman_rates_man(_w, m, w_attr_f, w_res_f):
    """How much the woman values this man (linear in his traits)."""
    return w_attr_f * m.attractiveness + w_res_f * m.resources


def quantile_reservation(utilities, q):
    """Reservation utility at quantile q in [0, 1] (higher q = pickier)."""
    if not utilities:
        return 0.0
    s = sorted(utilities)
    idx = min(int(q * (len(s) - 1)), len(s) - 1)
    return float(s[idx])


class Person(mesa.Agent):
    def __init__(self, model, sex, idx, attractiveness, resources):
        super().__init__(model)
        self.sex = sex  # "M" or "F"
        self.idx = idx
        self.attractiveness = float(attractiveness)
        self.resources = float(resources)
        self.partner_idx = None
        self.prefs = []  # opposite-sex indices, best first (by utility)
        self.reservation_utility = 0.0


class DatingMarket(mesa.Model):
    """
    Parameters default to a *stylized* sex difference: men weight women's
    attractiveness more than their resources; women weight men's resources
    more than attractiveness - matching the *direction* of many cross-cultural
    survey averages, not calibrated effect sizes.
    """

    def __init__(
        self,
        n_each=40,
        rng=None,
        # Utility weights (must be non-negative)
        male_weight_on_female_attractiveness=0.78,
        male_weight_on_female_resources=0.22,
        female_weight_on_male_attractiveness=0.35,
        female_weight_on_male_resources=0.65,
        # Reservation quantiles: higher = accept only higher utility partners
        male_reservation_quantile=0.38,
        female_reservation_quantile=0.52,
        # Logistic noise on accept/reject (0 = deterministic threshold rule)
        acceptance_temperature=0.08,
        # Idiosyncratic taste: within-sex heterogeneity in who finds whom appealing
        male_taste_sd=0.045,
        female_taste_sd=0.045,
    ):
        super().__init__(rng=rng if rng is not None else 42)
        self.n = n_each
        self.w_attr_m = male_weight_on_female_attractiveness
        self.w_res_m = male_weight_on_female_resources
        self.w_attr_f = female_weight_on_male_attractiveness
        self.w_res_f = female_weight_on_male_resources
        self.q_res_m = male_reservation_quantile
        self.q_res_f = female_reservation_quantile
        self.accept_temp = max(0.0, float(acceptance_temperature))
        self._male_taste_sd = max(0.0, float(male_taste_sd))
        self._female_taste_sd = max(0.0, float(female_taste_sd))

        attractiveness_seq = [self.random.random() for _ in range(n_each)]
        resources_seq = [self.random.random() for _ in range(n_each)]
        # Mild positive correlation (halo / life outcomes); optional realism tweak
        for i in range(n_each):
            blend = 0.65 * attractiveness_seq[i] + 0.35 * resources_seq[i]
            attractiveness_seq[i] = 0.5 * attractiveness_seq[i] + 0.5 * blend
            resources_seq[i] = 0.5 * resources_seq[i] + 0.5 * blend

        men_idx = list(range(n_each))
        women_idx = list(range(n_each))
        Person.create_agents(
            model=self,
            n=n_each,
            sex=["M"] * n_each,
            idx=men_idx,
            attractiveness=attractiveness_seq,
            resources=resources_seq,
        )
        women_attrs = [self.random.random() for _ in range(n_each)]
        women_res = [self.random.random() for _ in range(n_each)]
        for i in range(n_each):
            blend = 0.65 * women_attrs[i] + 0.35 * women_res[i]
            women_attrs[i] = 0.5 * women_attrs[i] + 0.5 * blend
            women_res[i] = 0.5 * women_res[i] + 0.5 * blend
        Person.create_agents(
            model=self,
            n=n_each,
            sex=["F"] * n_each,
            idx=women_idx,
            attractiveness=women_attrs,
            resources=women_res,
        )

        self.men = sorted(
            [a for a in self.agents if a.sex == "M"], key=lambda a: a.idx
        )
        self.women = sorted(
            [a for a in self.agents if a.sex == "F"], key=lambda a: a.idx
        )

        # Pair-specific taste shocks so rankings differ within each sex (survey
        # heterogeneity; avoids everyone sharing one global "league table").
        self._taste_mw = [
            [
                self.random.gauss(0, self._male_taste_sd)
                for _ in range(n_each)
            ]
            for _ in range(n_each)
        ]
        self._taste_wm = [
            [
                self.random.gauss(0, self._female_taste_sd)
                for _ in range(n_each)
            ]
            for _ in range(n_each)
        ]

        self._build_preferences_and_reservations()

    def _utility(self, agent, partner):
        if agent.sex == "M" and partner.sex == "F":
            base = utility_man_rates_woman(
                agent, partner, self.w_attr_m, self.w_res_m
            )
            return base + self._taste_mw[agent.idx][partner.idx]
        if agent.sex == "F" and partner.sex == "M":
            base = utility_woman_rates_man(
                agent, partner, self.w_attr_f, self.w_res_f
            )
            return base + self._taste_wm[agent.idx][partner.idx]
        raise ValueError("Utility only defined for male-female pairs.")

    def _build_preferences_and_reservations(self):
        for m in self.men:
            scored = [
                (self._utility(m, w), w.idx) for w in self.women
            ]
            scored.sort(key=lambda t: (-t[0], t[1]))
            m.prefs = [idx for _, idx in scored]
            utils = [self._utility(m, w) for w in self.women]
            m.reservation_utility = quantile_reservation(utils, self.q_res_m)

        for w in self.women:
            scored = [
                (self._utility(w, m), m.idx) for m in self.men
            ]
            scored.sort(key=lambda t: (-t[0], t[1]))
            w.prefs = [idx for _, idx in scored]
            utils = [self._utility(w, m) for m in self.men]
            w.reservation_utility = quantile_reservation(utils, self.q_res_f)

    def _logistic_accept(self, utility, threshold):
        if self.accept_temp == 0.0:
            return utility >= threshold
        # Smooth acceptance: more likely to accept when utility exceeds threshold
        x = (utility - threshold) / self.accept_temp
        p = 1.0 / (1.0 + math.exp(-x))
        return self.random.random() < p

    def step(self):
        """One round of random serial encounters between unmatched M/F pairs."""
        singles_m = [m for m in self.men if m.partner_idx is None]
        singles_f = [w for w in self.women if w.partner_idx is None]
        if not singles_m or not singles_f:
            return

        self.random.shuffle(singles_m)
        self.random.shuffle(singles_f)
        k = min(len(singles_m), len(singles_f))

        for t in range(k):
            m = singles_m[t]
            w = singles_f[t]
            um = self._utility(m, w)
            uw = self._utility(w, m)
            if self._logistic_accept(um, m.reservation_utility) and self._logistic_accept(
                uw, w.reservation_utility
            ):
                m.partner_idx = w.idx
                w.partner_idx = m.idx

    def run_search(self, n_rounds):
        for _ in range(n_rounds):
            self.step()

    def apply_gale_shapley_benchmark(self):
        """Assign partners by men-optimal stable matching on strict prefs."""
        men_prefs = [m.prefs for m in self.men]
        women_prefs = [w.prefs for w in self.women]
        match_w_to_m = gale_shapley_men_optimal(men_prefs, women_prefs)
        for w in self.women:
            w.partner_idx = None
        for m in self.men:
            m.partner_idx = None
        for w_idx, m_idx in match_w_to_m.items():
            self.men[m_idx].partner_idx = w_idx
            self.women[w_idx].partner_idx = m_idx


def rank_of_partner(agent):
    """1-based rank of current partner in agent.prefs (full-information list)."""
    if agent.partner_idx is None:
        return None
    return agent.prefs.index(agent.partner_idx) + 1


def pearson_corr(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (vx * vy) ** 0.5


def report(model, label):
    men = model.men
    women = model.women
    n = model.n
    pairs = [
        (m, women[m.partner_idx]) for m in men if m.partner_idx is not None
    ]
    ranks_m = [rank_of_partner(m) for m in men if m.partner_idx is not None]
    ranks_w = [rank_of_partner(w) for w in women if w.partner_idx is not None]

    print(f"\n=== {label} ===")
    print(f"Matched pairs: {len(pairs)} / {n}")
    if ranks_m:
        print(
            "Mean preference rank of partner for men (1 = top):",
            sum(ranks_m) / len(ranks_m),
        )
    if ranks_w:
        print(
            "Mean preference rank of partner for women (1 = top):",
            sum(ranks_w) / len(ranks_w),
        )
    if len(pairs) >= 2:
        a_m = [m.attractiveness for m, w in pairs]
        a_f = [w.attractiveness for m, w in pairs]
        r_m = [m.resources for m, w in pairs]
        r_f = [w.resources for m, w in pairs]
        print(
            "Assortment r (attractiveness M vs F):",
            round(pearson_corr(a_m, a_f), 3),
        )
        print("Assortment r (resources M vs F):", round(pearson_corr(r_m, r_f), 3))


if __name__ == "__main__":
    n = 60
    rng = 42
    model = DatingMarket(
        n_each=n,
        rng=rng,
        male_reservation_quantile=0.36,
        female_reservation_quantile=0.55,
        acceptance_temperature=0.06,
    )

    # Search: random encounters; need enough rounds for most to find someone
    model.run_search(n_rounds=80 * n)
    report(model, "Encounter-based search (stylized)")

    # Reset and run full-information stable benchmark on same traits/prefs
    model2 = DatingMarket(
        n_each=n,
        rng=rng,
        male_reservation_quantile=0.36,
        female_reservation_quantile=0.55,
        acceptance_temperature=0.06,
    )
    model2.apply_gale_shapley_benchmark()
    report(model2, "Gale-Shapley benchmark (full info, men propose)")

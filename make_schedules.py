from __future__ import division
import sys
import os.path as op
from string import letters
import numpy as np
from numpy.random import permutation
from pandas import DataFrame
import tools


def main(arglist):

    p = tools.Params("gape_sched")

    # Make a certain number of schedules
    # We'll ID each one with a letter and then
    # scramble the run - letter association across subjs
    all_ids = list(letters[:p.total_schedules]) + ["train_a", "train_b", "train_c"]
    for id in all_ids:
        df = build_run_schedule(p)
        fname = op.join("sched", "schedule_%s.csv" % id)
        df.to_csv(fname, index_label="trial")


def build_run_schedule(p):

    # First get the category level order
    n_macroblocks = p.n_blocks / 3
    assert n_macroblocks - int(n_macroblocks) == 0
    n_macroblocks = int(n_macroblocks)
    block_cat_sched = [range(3) for i in range(n_macroblocks)]
    block_cat_sched = map(permutation, block_cat_sched)
    block_cat_sched = np.concatenate(block_cat_sched)

    # Set up the output lists
    itis = []
    blocks = []
    ori_as = []
    ori_bs = []
    cat_ids = []
    ev_types = []
    oddballs = []

    # Set up the seeds to get permuted
    iti_choices = p.iti_options * 2
    block_events = (["catch"] * p.catch_per_block +
                    ["seq"] * p.seq_per_block)

    # Set up the oddballs properly
    total_seqs = p.seq_per_block * p.n_blocks
    macro_seqs = total_seqs / n_macroblocks
    assert macro_seqs - int(macro_seqs) == 0
    macro_seqs = int(macro_seqs)
    macro_oddballs = p.oddball_prob * macro_seqs
    assert macro_oddballs - int(macro_oddballs) == 0
    macro_oddballs = int(macro_oddballs)
    oddball_sched = [(macro_oddballs * [1] +
                     (macro_seqs - macro_oddballs) * [0])
                     for i in range(n_macroblocks)]
    oddball_sched = map(permutation, oddball_sched)
    oddball_sched = np.concatenate(oddball_sched)
    oddball_scheds = np.split(oddball_sched, p.n_blocks)

    # Build the schedule
    for i, block_cat in enumerate(block_cat_sched):

        with_rest = not (i + 1) % p.rest_every_n

        event_per_block = 1 + p.seq_per_block + p.catch_per_block
        if with_rest:
            event_per_block += 1
        

        # First the easy bits
        blocks.extend([i] * event_per_block)
        itis.extend(permutation(iti_choices))
        cat_ids.extend([block_cat] * event_per_block)

        # Now the stim orientations
        ori_a, ori_b = permutation(p.stim_orients)[:2]
        ori_as.extend([ori_a] * event_per_block)
        ori_bs.extend([ori_b] * event_per_block)

        # Next the event types
        this_block = permutation(block_events)
        ev_types.append("demo")
        ev_types.extend(this_block)
        if with_rest:
            ev_types.append("rest")
            itis.append(0)

        # Next the oddballs
        odd = oddball_scheds[i].tolist()
        block_odd = [0]
        for type in this_block:
            if type == "seq":
                block_odd.append(odd.pop())
            else:
                block_odd.append(0)
        if with_rest:
            block_odd.append(0)
        oddballs.extend(block_odd)


    # Put it all together and return
    return DataFrame(dict(ev_type=ev_types,
                          cat_id=cat_ids,
                          block=blocks,
                          iti=itis,
                          ori_a=ori_as,
                          ori_b=ori_bs,
                          oddball=oddballs))

if __name__ == "__main__":
    main(sys.argv[1:])

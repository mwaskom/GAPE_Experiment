from __future__ import division
import sys
import os.path as op
from textwrap import dedent
from string import letters
from pandas import read_csv
import numpy as np
from numpy.random import RandomState, multinomial, randint, uniform
from psychopy import visual, core, event
import psychopy.monitors.calibTools as calib
import tools
from tools import draw_all, check_quit, wait_check_quit


def run_experiment(arglist):

    # Get the experiment paramters
    p = tools.Params("gape")
    p.set_by_cmdline(arglist)

    # Sequence categories
    cat_list = [[0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 0]]
    cat_names = ["alternated", "paired", "reflected"]

    # Get this run's schedule in a manner that is consistent
    # within and random between subjects
    if p.train:
        letter = letters[p.run - 1]
        p.sched_id = "train_%s" % letter
        sched_file = "sched/schedule_%s.csv" % p.sched_id

    else:
        state = RandomState(abs(hash(p.subject)))
        choices = list(letters[:p.total_schedules])
        p.sched_id = state.permutation(choices)[p.run - 1]
        sched_file = "sched/schedule_%s.csv" % p.sched_id

    # Read in this run's schedule
    s = read_csv(sched_file)

    # Max the screen brightness
    tools.max_brightness(p.monitor_name)

    # Open up the stimulus window
    calib.monitorFolder = "./calib"
    mon = calib.Monitor(p.monitor_name)
    m = tools.WindowInfo(p, mon)
    win = visual.Window(**m.window_kwargs)

    # Set up the stimulus objects
    fix = visual.PatchStim(win, tex=None, mask="circle",
                           color=p.fix_color, size=p.fix_size)
    a_fix = visual.PatchStim(win, tex=None, mask="circle",
                             color=p.fix_antic_color, size=p.fix_size)
    r_fix = visual.PatchStim(win, tex=None, mask="circle",
                             color=p.fix_resp_color, size=p.fix_size)
    d_fix = visual.PatchStim(win, tex=None, mask="circle",
                             color=p.fix_demo_color, size=p.fix_size)
    c_fix = visual.PatchStim(win, tex=None, mask="circle",
                             color=p.fix_catch_color, size=p.fix_size)
    b_fix = visual.PatchStim(win, tex=None, mask="circle",
                             color=p.fix_break_color, size=p.fix_size)
    halo = visual.PatchStim(win, tex=None, mask=p.demo_halo_mask,
                            opacity=p.demo_halo_opacity,
                            color=p.demo_halo_color,
                            size=p.demo_halo_size)
    grate = visual.PatchStim(win, "sin", p.stim_mask, size=p.stim_size,
                             contrast=p.stim_contrast, sf=p.stim_sf,
                             opacity=p.stim_opacity)
    disk = visual.PatchStim(win, tex=None, mask=p.stim_mask,
                            color=win.color, size=p.stim_disk_ratio)
    stims = [grate, disk, fix]

    # Set up some timing variables
    running_time = 0
    antic_secs = p.tr
    demo_secs = 4 * p.demo_stim_dur + 3 * p.demo_stim_isi + p.tr
    seq_secs = p.tr + 4 * p.stim_dur + 3 * p.stim_isi
    catch_secs = p.tr
    rest_secs = p.rest_trs * p.tr

    # Draw the instructions and wait to go
    instruct = dedent("""
    Watch the sample sequence and say if the target sequences match

    Blue dot: sample sequence
    Red dot: get ready
    Orange dot: relax
    Green dot: say if sequence matched the sample
    Button 1: same    Button 2: different

    Grey dot: quick break


    Experimenter: Press space to prep for scan""")  # TODO
    # Draw the instructions and wait to go
    tools.WaitText(win, instruct, height=.7)(check_keys=["space"])

    # Possibly wait for the scanner
    if p.fmri:
        tools.wait_for_trigger(win, p)

    # Start a data file and write the params to it
    f, fname = tools.start_data_file(p.subject, p.experiment_name,
                                     p.run, train=p.train)
    p.to_text_header(f)

    # Save run params to JSON
    save_name = op.join("./data", op.splitext(fname)[0])
    p.to_json(save_name)

    # Write the datafile header
    header = ["trial", "block",
              "cat_id", "cat_name",
              "event_type",
              "event_sched", "event_time",
              "ori_a", "ori_b",
              "oddball", "odd_item", "odd_orient",
              "iti", "response", "rt", "acc"]
    tools.save_data(f, *header)

    # Start a clock and flush the event buffer
    exp_clock = core.Clock()
    trial_clock = core.Clock()
    event.clearEvents()

    # Main experiment loop
    # --------------------
    try:

        # Dummy scans
        fix.draw()
        win.flip()
        dummy_secs = p.dummy_trs * p.tr
        running_time += dummy_secs
        wait_check_quit(dummy_secs, p.quit_keys)

        for t in s.trial:

            cat_seq = cat_list[s.cat_id[t]]
            block_ori_list = np.array([s.ori_a[t], s.ori_b[t]])[cat_seq]

            # Set up some defaults for variables that aren't always set
            oddball_seq = [0, 0, 0 ,0]
            odd_item, odd_ori = -1, -1
            acc, response, resp_rt = -1, -1, -1

            # Possibly rest and then bail out of the rest of the loop
            if s.ev_type[t] == "rest":
                if p.train and not p.fmri:
                    b_fix.draw()
                    win.flip()
                    wait_check_quit(2)
                    before = exp_clock.getTime()
                    msg = "Quick break! Press space to continue."
                    tools.WaitText(win, msg, height=.7)(check_keys=["space"])
                    b_fix.draw()
                    win.flip()
                    wait_check_quit(2)
                    after = exp_clock.getTime()
                    rest_time = after - before
                    running_time += rest_time
                    continue
                else:
                    b_fix.draw()
                    win.flip()
                    wait_check_quit(rest_secs)
                    running_time += rest_secs
                    continue
 
            # Otherwise, we always get an anticipation
            if p.antic_fix_dur <= p.tr:  # possibly problematic
                fix.draw()
                win.flip()
                core.wait(p.tr - p.antic_fix_dur)
            if s.ev_type[t] == "demo":
                stim = d_fix
            else:
                stim = a_fix
            end_time = running_time + p.antic_fix_dur
            tools.precise_wait(win, exp_clock, end_time, stim)
            running_time += antic_secs

            # The event is about to happen so stamp that time
            event_sched = running_time
            event_time = exp_clock.getTime()

            # Demo sequence
            if s.ev_type[t] == "demo":

                for i, ori in enumerate(block_ori_list):
                    # Draw each stim
                    grate.setOri(ori)
                    halo.draw()
                    draw_all(*stims)
                    d_fix.draw()
                    win.flip()
                    core.wait(p.demo_stim_dur)

                    # Short isi fix
                    if i < 3:
                        d_fix.draw()
                        win.flip()
                        core.wait(p.demo_stim_isi)
                    check_quit()

                # Demo always has >1 TR fixation
                fix.draw()
                win.flip()
                wait_check_quit(p.tr)

                # Update timing
                running_time += demo_secs

            # Proper test sequence
            if s.ev_type[t] == "seq":

                # If this is an oddball, figure out where
                if s.oddball[t]:
                    oddball_seq = multinomial(1, [.25] * 4).tolist()
                    odd_item = oddball_seq.index(1)

                # Iterate through each element in the sequence
                for i, ori in enumerate(block_ori_list):

                    # Set the grating attributes
                    if oddball_seq[i]:
                        ori_choices = [o for o in p.stim_orients
                                       if not o == ori]
                        odd_ori = ori_choices[randint(3)]
                        grate.setOri(odd_ori)
                    else:
                        grate.setOri(ori)
                    grate.setPhase(uniform())

                    # Draw the grating set
                    draw_all(*stims)
                    win.flip()
                    core.wait(p.stim_dur)

                    # ISI Fix (on all but last stim)
                    if i < 3:
                        fix.draw()
                        win.flip()
                        core.wait(p.stim_isi)
                    check_quit()

                # Response fixation
                r_fix.draw()
                trial_clock.reset()
                event.clearEvents()
                win.flip()
                acc, response, resp_rt = wait_get_response(p,
                                                           trial_clock,
                                                           s.oddball[t],
                                                           p.resp_dur)

                # Update timing
                running_time += seq_secs

            # Catch trial
            if s.ev_type[t] == "catch":
                c_fix.draw()
                win.flip()
                wait_check_quit(p.tr)
                running_time += catch_secs

            # Save data to the datafile
            data = [t, s.block[t],
                    s.cat_id[t], cat_names[s.cat_id[t]],
                    s.ev_type[t],
                    event_sched, event_time,
                    s.ori_a[t], s.ori_b[t],
                    s.oddball[t],
                    odd_item, odd_ori, s.iti[t],
                    response, resp_rt, acc]
            tools.save_data(f, *data)

            # ITI interval
            # Go by screen refreshes for precise timing
            this_iti = s.iti[t] * p.tr
            end_time = running_time + this_iti
            tools.precise_wait(win, exp_clock, end_time, fix)
            running_time += this_iti

            
    finally:
        # Clean up
        f.close()
        win.close()

    # Good execution, print out some info
    try:
        data_file = op.join("data", fname)
        with open(data_file, "r") as fid:
            lines = fid.readlines()
            n_comments = len([l for l in lines if l.startswith("#")])
        df = read_csv(data_file, skiprows=n_comments, na_values=["-1"])

        info = dict()
        time_error = df.event_sched - df.event_time
        info["run"] = p.run
        info["acc"] = df.acc.mean()
        info["mean_rt"] =  df.rt.mean()
        info["missed_resp"] = (df.response == 0).sum()
        info["time_error_mean"] = abs(time_error).mean()
        info["time_error_max"] = max(time_error)

        print dedent("""Performance summary for run %(run)d:

        Accuracy: %(acc).3f
        Mean RT: %(mean_rt).3f
        Missed responses: %(missed_resp)d

        Mean timing error: %(time_error_mean).4f
        Max timing error: %(time_error_max).4f
        """ % info)

    except Exception as err:
        print "Could not read data file for summary"
        print err


def wait_get_response(p, clock, oddball, wait_time):
    """Get response info specific to this experiment."""
    check_clock = core.Clock()
    good_resp = False
    corr, response, resp_rt = 0, 0, -1
    while not good_resp:
        keys = event.getKeys(timeStamped=clock)
        for key, stamp in keys:
            if key in p.quit_keys:
                print "Subject quit execution"
                core.quit()
            elif key in p.match_keys:
                corr = 0 if oddball else 1
                response = 1
                resp_rt = stamp
                good_resp = True
                break
            elif key in p.nonmatch_keys:
                corr = 1 if oddball else 0
                response = 2
                resp_rt = stamp
                good_resp = True
                break
            event.clearEvents()
        # Possibly exit with nothing
        if check_clock.getTime() >= wait_time:
            return corr, response, resp_rt
    # Wait the rest of the time
    core.wait(wait_time - resp_rt)
    return corr, response, resp_rt


if __name__ == "__main__":
    run_experiment(sys.argv[1:])

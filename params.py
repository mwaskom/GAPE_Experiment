gape = dict(

    experiment_name="gape",

    # Display setup
    monitor_name='mlw-mbpro',
    monitor_units="deg",
    full_screen=True,
    screen_number=1,

    # Fixation
    fix_size=.2,
    fix_color='#FFFFFF',
    fix_demo_color='#1E90FF',
    fix_antic_color='#FF2400',
    fix_resp_color='#7CFC00',
    fix_catch_color='#F46F1B',
    fix_break_color='#BBBBBB',

    # Gratings
    stim_size=7,
    stim_mask="gauss",
    stim_contrast=1,
    stim_opacity=1,
    stim_sf=2,
    stim_disk_ratio=8,

    # Demo halo
    demo_halo_size=8,
    demo_halo_color="#1E90FF",
    demo_halo_opacity=.5,
    demo_halo_mask="gauss",

    # Response settings
    quit_keys=["escape", "q"],
    match_keys=['2', 'comma'],
    nonmatch_keys=['3', 'period'],

    # Timing
    tr=2,
    antic_fix_dur=2,
    stim_dur=2. / 7,
    stim_isi=2. / 7,
    resp_dur=2,
    demo_stim_dur=4. / 7,
    demo_stim_isi=4. / 7,
    dummy_trs = 4,
    rest_trs=6,

    )

gape_sched = dict(

    # Event counts for make_schedule.py 
    n_cat=3,
    n_blocks=9,
    stim_orients=[0, 45, 90, 135],
    seq_per_block=3,
    catch_per_block=2,
    max_repeat=2,
    n_search=1000,
    iti_options=[0, 1, 2],
    oddball_prob=.25,
    total_schedules=12,
    rest_every_n=3,

    )
gape.update(gape_sched)


def add_cmdline_params(parser):

    parser.add_argument("-train", action="store_true")
    return parser

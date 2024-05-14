import argparse
import sys
import os
from pathlib import Path
from copy import deepcopy
import socket
import time

import numpy as np

from matplotlib import pyplot as plt

from astropy import units as u

import pint.toa
from pint.models import model_builder as mb

from APT import APT


def main(argv=None):

    # read in arguments from the command line

    """required = parfile, timfile"""
    """optional = starting points, param ranges"""
    parser = argparse.ArgumentParser(
        description="PINT tool for agorithmically timing pulsars."
    )

    parser.add_argument("parfile", help="par file to read model from")
    parser.add_argument("timfile", help="tim file to read toas from")
    parser.add_argument(
        "--starting_points",
        help="mask array to apply to chose the starting points, clusters or mjds",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--maskfile",
        help="csv file of bool array for fit points",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n_pred",
        help="Number of predictive models that should be calculated",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--ledge_multiplier",
        help="scale factor for how far to plot predictive models to the left of fit points",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--redge_multiplier",
        help="scale factor for how far to plot predictive models to the right of fit points",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--RAJ_lim",
        help="minimum time span before Right Ascension (RAJ) can be fit for",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--DECJ_lim",
        help="minimum time span before Declination (DECJ) can be fit for",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--F1_lim",
        help="minimum time span before Spindown (F1) can be fit for (default = time for F1 to change residuals by 0.35phase)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--Ftest_lim",
        help="Upper limit for successful Ftest values",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--check_bad_points",
        help="whether the algorithm should attempt to identify and ignore bad data",
        type=str,
        default="True",
    )
    parser.add_argument(
        "--plot_bad_points",
        help="Whether to actively plot the polynomial fit on a bad point. This will interrupt the program and require manual closing",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--check_bp_min_diff",
        help="minimum residual difference to count as a questionable point to check",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--check_bp_max_resid",
        help="maximum polynomial fit residual to exclude a bad data point",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--check_bp_n_clusters",
        help="how many clusters ahead of the questionable group to fit to confirm a bad data point",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--try_poly_extrap",
        help="whether to try to speed up the process by fitting ahead where polyfit confirms a clear trend",
        type=str,
        default="True",
    )
    parser.add_argument(
        "--plot_poly_extrap",
        help="Whether to plot the polynomial fits during the extrapolation attempts. This will interrupt the program and require manual closing",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--pe_min_span",
        help="minimum span (days) before allowing polynomial extrapolation attempts",
        type=float,
        default=30,
    )
    parser.add_argument(
        "--pe_max_resid",
        help="maximum acceptable goodness of fit for polyfit to allow the polynomial extrapolation to succeed",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--span1_c",
        help="coefficient for first polynomial extrapolation span (i.e. try polyfit on current span * span1_c)",
        type=float,
        default=1.3,
    )
    parser.add_argument(
        "--span2_c",
        help="coefficient for second polynomial extrapolation span (i.e. try polyfit on current span * span2_c)",
        type=float,
        default=1.8,
    )
    parser.add_argument(
        "--span3_c",
        help="coefficient for third polynomial extrapolation span (i.e. try polyfit on current span * span3_c)",
        type=float,
        default=2.4,
    )
    parser.add_argument(
        "--max_wrap",
        help="how many phase wraps in each direction to try",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--plot_final",
        help="whether to plot the final residuals at the end of each attempt",
        type=str,
        default="True",
    )
    parser.add_argument(
        "--data_path",
        help="where to store data",
        type=str,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--parfile_compare",
        help="par file to compare solution to",
        type=str,
        default="",
    )
    parser.add_argument(
        "--chisq_cutoff",
        help="",
        type=float,
        default=10,
    )

    args = parser.parse_args(argv)
    # interpret strings as booleans
    args.check_bad_points = args.check_bad_points.lower()[0] == "t"
    args.try_poly_extrap = args.try_poly_extrap.lower()[0] == "t"
    args.plot_poly_extrap = args.plot_poly_extrap.lower()[0] == "t"
    args.plot_bad_points = args.plot_bad_points.lower()[0] == "t"
    args.plot_final = args.plot_final.lower()[0] == "t"

    # if given starting points from command line, check if ints (group numbers) or floats (mjd values)
    start_type = None
    start = None
    if args.starting_points != None:
        start = args.starting_points.split(",")
        try:
            start = [int(i) for i in start]
            start_type = "clusters"
        except:
            start = [float(i) for i in start]
            start_type = "mjds"

    """start main program"""
    # construct the filenames
    # datadir = os.path.dirname(os.path.abspath(str(__file__))) # replacing these lines with the following lines
    # allows APT to be run in the directory that the command was ran on
    # parfile = os.path.join(datadir, args.parfile)
    # timfile = os.path.join(datadir, args.timfile)
    parfile = Path(args.parfile)
    timfile = Path(args.timfile)
    original_path = Path.cwd()
    data_path = Path(args.data_path)

    #### FIXME When fulled implemented, DELETE the following lines
    if socket.gethostname()[0] == "J":
        data_path = Path.cwd()
    else:
        data_path = Path("/data1/people/jdtaylor")
    ####
    os.chdir(data_path)

    # read in the toas
    t = pint.toa.get_TOAs(timfile)
    sys_name = str(mb.get_model(parfile).PSR.value)

    # check that there is a directory to save the algorithm state in
    if not Path("alg_saves").exists():
        Path("alg_saves").mkdir()

    # checks there is a directory specific to the system in alg_saves
    if not Path(f"alg_saves/{sys_name}").exists():
        Path(f"alg_saves/{sys_name}").mkdir()

    # set F1_lim if one not given
    APT.set_F1_lim(args, parfile)

    masknumber = -1
    for mask in APT.starting_points(t, start_type):
        print(mask)
        # raise Exception("Quit")
        masknumber += 1
        # starting_points returns a list of boolean arrays, each a mask for the base toas. Iterating through all of them give different pairs of starting points
        # read in the initial model
        m = mb.get_model(parfile)

        # check for extraneous parameters such as DM which will ruin fit
        if "DM" in m.params and getattr(m, "DM").frozen == False:
            print(
                "WARNING: APT only fits for F0, RAJ, DECJ, and F1, not DM. Please turn off DM in the parfile, and try again."
            )
            break

        # TODO: add this in when 0.8 comes out and get_params_dict is available
        # for parameter in m.params:
        #    print(parameter)
        #    if (
        #        parameter not in ["FO", "RAJ", "DECJ", "F1"]
        #    ):
        #        print("WARNING: APT only fits for F0, RAJ, DECJ, and F1. All other parameters should be turned off. Turning off parameter ", parameter)
        #        getattr(m, parameter).frozen = True

        # Read in the TOAs and compute their pulse numbers
        t = pint.toa.get_TOAs(timfile)

        # Print a summary of the TOAs that we have
        t.print_summary()

        # check has TZR params
        try:
            m.TZRMJD
            m.TZRSITE
            m.TZRFRQ
        except:
            print("Error: Must include TZR parameters in parfile")
            return -1

        if args.starting_points != None or args.maskfile != None:
            mask = APT.readin_starting_points(mask, t, start_type, start, args)

        # apply the starting mask and print the group(s) the starting points are part of
        t.select(mask)
        print("Starting clusters:\n", t.get_clusters())

        # save starting TOAs to print out at end if successful
        starting_TOAs = deepcopy(t)

        # for first iteration, last model, toas, and starting points is just the base ones read in
        last_model = deepcopy(m)
        last_t = deepcopy(t)
        last_mask = deepcopy(mask)

        # toas as read in from timfile, should only be modified with deletions
        base_TOAs = pint.toa.get_TOAs(timfile)

        cont = True
        iteration = 0
        bad_mjds = []
        # if given a maskfile (csv), read in the iteration we are on from the maskfile filename
        if args.maskfile != None:
            iteration = int(
                args.maskfile[-8:].strip(
                    "qwertyuioplkjhgfdsazxcvbnmQWERTYUIOPLKJHGFDSAMNBVCXZ._"
                )
            )

        while cont:
            # main loop of the algorithm, continues until all toas have been included in fit
            iteration += 1
            skip_phases = False

            # fit the toas with the given model as a baseline
            print("Fitting...")
            f = pint.fitter.WLSFitter(t, m)
            print("BEFORE:", f.get_fitparams())
            print(f.fit_toas())

            print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
            print("RMS in phase is", f.resids.phase_resids.std())
            print("RMS in time is", f.resids.time_resids.std().to(u.us))
            print("\n Best model is:")
            print(f.model.as_parfile())

            # calculate random models and residuals
            full_clusters, selected, rs_mean, f_toas, rss, rmods = (
                APT.calc_random_models(base_TOAs, f, t, args)
            )

            # define t_others
            t_others = deepcopy(base_TOAs)

            # calculate the group closest to the fit toas, pass deepcopies to prevent unintended pass by reference
            closest_group, dist = APT.get_closest_cluster(
                deepcopy(t_others), deepcopy(t), deepcopy(base_TOAs)
            )
            print("closest group:", closest_group)

            if closest_group is None:
                # end the program
                # print the final model and toas
                # save the latest model as a new parfile (all done at end of code)
                cont = False
                continue

            # right now t_others is just all the toas, so can use as all
            # redefine a as the mask giving the fit toas plus the closest group of toas
            mask = np.logical_or(mask, t_others.get_clusters() == closest_group)

            # define t_others as the current fit toas pluse the closest group
            t_others.select(mask)
            # t_others is now the fit toas plus the to be added group of toas, t is just the fit toas

            # calculate difference in resids between current fit group and closest group
            selected_closest, diff = APT.calc_resid_diff(
                closest_group, full_clusters, base_TOAs, f, selected
            )
            minmjd, maxmjd = (min(t_others.get_mjds()), max(t_others.get_mjds()))
            span = maxmjd - minmjd

            nclusters = max(t_others.get_clusters())

            # if difference in phase is >0.15 and check_bad_points is True, check if the TOA is a bad data point
            if (
                np.abs(diff) > args.check_bp_min_diff
                and args.check_bad_points == True
                and nclusters > 10
            ):
                skip_phases, t_others, mask, bad_mjds = APT.bad_points(
                    dist,
                    t,
                    closest_group,
                    args,
                    full_clusters,
                    base_TOAs,
                    m,
                    sys_name,
                    iteration,
                    t_others,
                    mask,
                    skip_phases,
                    bad_mjds,
                    data_path,
                    original_path,
                )

            # if difference in phase is >0.15, and not a bad point, try phase wraps to see if point fits better wrapped
            if np.abs(diff) > args.check_bp_min_diff and skip_phases is False:
                f_phases = []
                t_phases = []
                t_others_phases = []
                m_phases = []
                chi2_phases = []

                # try every phase wrap from -max_wrap to +max_wrap
                for wrap in range(-args.max_wrap, args.max_wrap + 1):
                    print("\nTrying phase wrap:", wrap)
                    # copy models to appropriate lists --> use index -1 because current object will always be the one just appended to array

                    # append the current fitter and toas to the appropriate lists
                    f_phases.append(deepcopy(f))
                    t_phases.append(deepcopy(base_TOAs))

                    # add the phase wrap to the closest group
                    APT.add_phase_wrap(t_phases[-1], f.model, selected_closest, wrap)

                    # append the wrapped toas to t_others and select the fit toas and closest group as normal
                    t_others_phases.append(deepcopy(t_phases[-1]))
                    t_others_phases[-1].select(mask)

                    # plot data
                    APT.plot_wraps(
                        f,
                        t_others_phases,
                        rmods,
                        f_toas,
                        rss,
                        t_phases,
                        m,
                        iteration,
                        wrap,
                        sys_name,
                        data_path,
                        original_path,
                    )

                    # repeat model selection with phase wrap. f.model should be same as f_phases[-1].model (all f_phases[n] should be the same)
                    chi2_ext_phase = [
                        pint.residuals.Residuals(
                            t_others_phases[-1],
                            rmods[i],
                            track_mode="use_pulse_numbers",
                        ).chi2_reduced
                        for i in range(len(rmods))
                    ]
                    chi2_dict_phase = dict(zip(chi2_ext_phase, rmods))

                    # append 0model to dict so it can also be a possibility
                    chi2_dict_phase[
                        pint.residuals.Residuals(
                            t_others_phases[-1], f.model, track_mode="use_pulse_numbers"
                        ).chi2_reduced
                    ] = f.model
                    min_chi2_phase = sorted(chi2_dict_phase.keys())[0]

                    # m_phases is list of best models from each phase wrap
                    m_phases.append(chi2_dict_phase[min_chi2_phase])

                    # mask = current t plus closest group, defined above
                    t_phases[-1].select(mask)

                    # fit toas with new model
                    f_phases[-1] = pint.fitter.WLSFitter(t_phases[-1], m_phases[-1])
                    f_phases[-1].fit_toas()

                    # do Ftests with phase wraps
                    m_phases[-1] = APT.do_Ftests_phases(
                        m_phases, t_phases, f_phases, args
                    )

                    # current best fit chi2 (extended points and actually fit for with maybe new param)
                    f_phases[-1] = pint.fitter.WLSFitter(t_phases[-1], m_phases[-1])
                    f_phases[-1].fit_toas()
                    chi2_phases.append(
                        pint.residuals.Residuals(
                            t_phases[-1],
                            f_phases[-1].model,
                            track_mode="use_pulse_numbers",
                        ).chi2
                    )

                # have run over all phase wraps
                # compare chi2 to see which is best and use that one's f, m, and t as the "correct" f, m, and t
                print("Comparing phase wraps")
                print(
                    np.column_stack(
                        (list(range(-args.max_wrap, args.max_wrap + 1)), chi2_phases)
                    )
                )

                i_phase = np.argmin(chi2_phases)
                print(
                    f"Phase wrap {list(range(-args.max_wrap, args.max_wrap + 1))[i_phase]} won with chi2 {chi2_phases[i_phase]}."
                )

                f = deepcopy(f_phases[i_phase])
                m = deepcopy(m_phases[i_phase])
                t = deepcopy(t_phases[i_phase])

                # fit toas just in case
                f.fit_toas()
                print("Current Fit Params:", f.get_fitparams().keys())
                # END INDENT FOR RESID > 0.15

            # if not resid > 0.15, run as normal
            else:
                # t is the current fit toas, t_others is the current fit toas plus the closest group, and a is the same as t_others

                minmjd, maxmjd = (min(t_others.get_mjds()), max(t_others.get_mjds()))
                print("Current Fit TOAs span is", maxmjd - minmjd)

                # do polynomial extrapolation check
                if (maxmjd - minmjd) > args.pe_min_span * u.d and args.try_poly_extrap:
                    try:
                        t_others, mask = APT.poly_extrap(
                            minmjd,
                            maxmjd,
                            args,
                            dist,
                            base_TOAs,
                            t_others,
                            full_clusters,
                            m,
                            mask,
                        )
                    except Exception as e:
                        print(
                            f"an error occurred while trying to do polynomial extrapolation. Continuing on. ({e})"
                        )

                chi2_summary = []
                chi2_ext_summary = []

                # calculate chi2 and reduced chi2 for base model
                model0 = deepcopy(f.model)
                chi2_summary.append(f.resids.chi2)
                chi2_ext_summary.append(
                    pint.residuals.Residuals(t_others, f.model).chi2
                )

                fig, ax = plt.subplots(constrained_layout=True)

                # calculate chi2 and reduced chi2 for the random models
                for i in range(len(rmods)):
                    chi2_summary.append(pint.residuals.Residuals(t, rmods[i]).chi2)
                    chi2_ext_summary.append(
                        pint.residuals.Residuals(t_others, rmods[i]).chi2
                    )
                    ax.plot(f_toas, rss[i], "-k", alpha=0.6)

                # print summary of chi squared values
                print("RANDOM MODEL SUMMARY:")
                print(f"chi2 median on fit TOAs: {np.median(chi2_summary)}")
                print(
                    f"chi2 median on fit TOAs plus closest group: {np.median(chi2_ext_summary)}"
                )
                print(f"chi2 stdev on fit TOAs: {np.std(chi2_summary)}")
                print(
                    f"chi2 stdev on fit TOAs plus closest group: {np.std(chi2_ext_summary)}"
                )

                print(f"Current Fit Params: {f.get_fitparams().keys()}")
                print(f"nTOAs (fit): {t.ntoas}")

                t = deepcopy(base_TOAs)
                # t is now a copy of the base TOAs (aka all the toas)
                print("nTOAS (total):", t.ntoas)

                # plot data
                plot_plain(
                    f,
                    t_others,
                    rmods,
                    f_toas,
                    rss,
                    t,
                    m,
                    iteration,
                    sys_name,
                    fig,
                    ax,
                    data_path,
                    original_path,
                )

                # get next model by comparing chi2 for t_others
                chi2_ext = [
                    pint.residuals.Residuals(t_others, rmods[i]).chi2_reduced
                    for i in range(len(rmods))
                ]
                chi2_dict = dict(zip(chi2_ext, rmods))

                # append 0model to dict so it can also be a possibility
                chi2_dict[pint.residuals.Residuals(t_others, f.model).chi2_reduced] = (
                    f.model
                )
                min_chi2 = sorted(chi2_dict.keys())[0]

                # the model with the smallest chi2 is chosen as the new best fit model
                m = chi2_dict[min_chi2]

                # mask = current t plus closest group, defined above
                t.select(mask)

                # do Ftests
                m = APT.do_Ftests(t, m, args)

                # current best fit chi2 (extended points and actually fit for with maybe new param)
                f = pint.fitter.WLSFitter(t, m)
                f.fit_toas()
                chi2_new_ext = pint.residuals.Residuals(t, f.model).chi2
                # END INDENT FOR ELSE (RESID < 0.35)

            # fit toas just in case
            f.fit_toas()

            # save current state in par, tim, and csv files
            last_model, last_t, last_mask = APT.save_state(
                m, t, mask, sys_name, iteration, base_TOAs, data_path, original_path
            )
            """for each iteration, save picture, model, toas, and a"""

        # try fitting with any remaining unfit parameters included and see if the fit is better for it
        m_plus = deepcopy(m)
        getattr(m_plus, "RAJ").frozen = False
        getattr(m_plus, "DECJ").frozen = False
        getattr(m_plus, "F1").frozen = False
        f_plus = pint.fitter.WLSFitter(t, m_plus)
        f_plus.fit_toas()

        # residuals
        r = pint.residuals.Residuals(t, f.model)
        r_plus = pint.residuals.Residuals(t, f_plus.model)
        if r_plus.chi2 <= r.chi2:
            f = deepcopy(f_plus)

        # save final model as .fin file
        print("Final Model:\n", f.model.as_parfile())

        # save as .fin
        fin_name = f.model.PSR.value + ".fin"
        with open(fin_name, "w") as finfile:
            finfile.write(f.model.as_parfile())

        # plot final residuals if plot_final True
        xt = t.get_mjds()
        fig, ax = plt.subplots()
        twinx = ax.twinx()
        ax.errorbar(
            xt.value,
            pint.residuals.Residuals(t, f.model).time_resids.to(u.us).value,
            t.get_errors().to(u.us).value,
            fmt=".b",
            label="post-fit (time)",
        )
        twinx.errorbar(
            xt.value,
            pint.residuals.Residuals(t, f.model).phase_resids,
            t.get_errors().to(u.us).value * float(f.model.F0.value) / 1e6,
            fmt=".b",
            label="post-fit (phase)",
        )
        ax.set_title(f"{m.PSR.value} Final Post-Fit Timing Residuals")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Residual (us)")
        twinx.set_ylabel("Residual (phase)", labelpad=15)
        span = (0.5 / float(f.model.F0.value)) * (10**6)
        plt.grid()

        time_end_main = time.monotonic()
        if args.plot_final:
            plt.show()

        fig.savefig(f"./alg_saves/{sys_name}/{sys_name}_final.png", bbox_inches="tight")
        plt.clf()

        # if success, stop trying and end program
        if pint.residuals.Residuals(t, f.model).chi2_reduced < float(args.chisq_cutoff):
            print(
                "SUCCESS! A solution was found with reduced chi2 of",
                pint.residuals.Residuals(t, f.model).chi2_reduced,
                "after",
                iteration,
                "iterations",
            )
            if args.parfile_compare:
                identical_solution = APT.solution_compare(
                    args.parfile_compare, f"{f.model.PSR.value}.fin", timfile
                )
                print(
                    f"\nThe .fin solution and comparison solution ARE {['NOT', ''][identical_solution]} identical.\n"
                )
            print(f"The input parameters for this fit were:\n {args}")
            print(
                f"\nThe final fit parameters are: {[key for key in f.get_fitparams().keys()]}"
            )
            print(f"starting points (clusters):\n {starting_TOAs.get_clusters()}")
            print(f"starting points (MJDs): {starting_TOAs.get_mjds()}")
            print(f"TOAs Removed (MJD): {bad_mjds}")
            break

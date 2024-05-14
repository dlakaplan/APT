#!/usr/bin/env python
# -W ignore::FutureWarning -W ignore::UserWarning -W ignore:DeprecationWarning
import pint.toa
import pint.models
import pint.fitter
import pint.residuals
import pint.utils
import pint.models.model_builder as mb
import pint.random_models
from pint.phase import Phase
from copy import deepcopy
from collections import OrderedDict
from astropy import log
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import operator
import time
from pathlib import Path
import socket

__all__ = ["main"]


def add_phase_wrap(toas, model, selected, phase):
    """
    Add a phase wrap to selected points in the TOAs object

    Turn on pulse number tracking in the model, if it isn't already

    :param selected: boolean array to apply to toas, True = selected toa
    :param phase: phase diffeence to be added, i.e.  -0.5, +2, etc.
    """
    # Check if pulse numbers are in table already, if not, make the column
    # if "pn" not in toas.table.colnames:
    if "pulse_number" not in toas.table.colnames:
        toas.compute_pulse_numbers(model)
    if "delta_pulse_number" not in toas.table.colnames:
        toas.table["delta_pulse_number"] = np.zeros(len(toas.get_mjds()))

    # add phase wrap
    toas.table["delta_pulse_number"][selected] += phase


def starting_points(toas, start_type):
    """
    Choose which TOAs to start the fit at based on highest density

    :param toas: TOAs object of all TOAs
    :return list of boolean arrays to mask starting TOAs:
    """
    # if starting points are given, return a list with one element so only runs the starting points once
    if start_type != None:
        return [0]

    # initialze TOAs object, TOA times, TOA densities, and lists to store TOA scores and masks
    t = deepcopy(toas)
    mjd_values = t.get_mjds().value
    dts = np.fabs(mjd_values - mjd_values[:, np.newaxis]) + np.eye(len(mjd_values))

    score_list = list((1.0 / dts).sum(axis=1))
    mask_list = []

    # while there are scores to be evaulated, create mask for highest scoring TOA, \
    # find closest TOA to be in a fittable pair, and remove duplicates from mask_list
    while np.any(score_list):

        hsi = np.argmax(score_list)
        score_list[hsi] = 0

        mask = list(np.zeros(len(mjd_values), dtype=bool))
        mask[hsi] = True

        if hsi == 0:
            mask[hsi + 1] = True

        elif hsi == len(mjd_values) - 1:
            mask[hsi - 1] = True

        elif (mjd_values[hsi] - mjd_values[hsi - 1]) >= (
            mjd_values[hsi + 1] - mjd_values[hsi]
        ):
            mask[hsi + 1] = True

        else:
            mask[hsi - 1] = True

        # remove duplicates
        if mask not in mask_list:
            mask_list.append(mask)

    # TODO: make 5 a settable param?
    return mask_list[:5]


def get_closest_cluster(all_toas, fit_toas, base_TOAs):
    """
    find the closest group of TOAs to the given toa(s)

    :param all_toas: TOAs object of all TOAs
    :param fit_toas: TOAs object of subset of TOAs that have already been fit
    :param base_TOAs: TOAs object of unedited TOAs as read from the timfile
    :return
    """

    fit_mjds = fit_toas.get_mjds()
    d_left = d_right = None

    # find distance to closest toa to the fit toas on the left \
    # (unless fit toas includes the overall leftmost toa, in which case d_left remains None)
    if min(fit_mjds) != min(all_toas.get_mjds()):
        all_toas.select(all_toas.get_mjds() < min(fit_mjds))
        left_dict = {min(fit_mjds) - mjd: mjd for mjd in all_toas.get_mjds()}
        d_left = min(left_dict.keys())

    # reset all_toas
    all_toas = deepcopy(base_TOAs)

    # find distance to closest toa to the fit toas on the right \
    # (unless fit toas includes the overall rightmost toa, in which case d_right remains None)
    if max(fit_mjds) != max(all_toas.get_mjds()):
        all_toas.select(all_toas.get_mjds() > max(fit_mjds))
        right_dict = {mjd - max(fit_mjds): mjd for mjd in all_toas.get_mjds()}
        d_right = min(right_dict.keys())

    # reset all_toas
    all_toas = deepcopy(base_TOAs)

    # return group number of closest group and distance to group (- = left, + = right), or None, None if all clusters have been included
    if d_left == None and d_right == None:
        print("all clusters have been included")
        return None, None

    elif d_left == None or (d_right != None and d_right <= d_left):
        all_toas.select(all_toas.get_mjds() == right_dict[d_right])
        return all_toas.table["clusters"][0], d_right

    else:
        all_toas.select(all_toas.get_mjds() == left_dict[d_left])
        return all_toas.table["clusters"][0], -d_left


def Ftest_param(r_model, fitter, param_name):
    """
    do an Ftest comparing a model with and without a particular parameter added

    Note: this is NOT a general use function - it is specific to this code and cannot be easily adapted to other scripts
    :param r_model: timing model to be compared
    :param fitter: fitter object containing the toas to compare on
    :param param_name: name of the timing model parameter to be compared
    :return
    """
    # read in model and toas
    m_plus_p = deepcopy(r_model)
    toas = deepcopy(fitter.toas)

    # set given parameter to unfrozen
    getattr(m_plus_p, param_name).frozen = False

    # make a fitter object with the chosen parameter unfrozen and fit the toas using the model with the extra parameter
    f_plus_p = pint.fitter.WLSFitter(toas, m_plus_p)
    f_plus_p.fit_toas()

    # calculate the residuals for the fit with (m_plus_p_rs) and without (m_rs) the extra parameter
    m_rs = pint.residuals.Residuals(toas, fitter.model)
    m_plus_p_rs = pint.residuals.Residuals(toas, f_plus_p.model)

    # calculate the Ftest, comparing the chi2 and degrees of freedom of the two models
    Ftest_p = pint.utils.FTest(
        float(m_rs.chi2), m_rs.dof, float(m_plus_p_rs.chi2), m_plus_p_rs.dof
    )
    # The Ftest determines how likely (from 0. to 1.) that improvement due to the new parameter is due to chance and not necessity
    # Ftests close to zero mean the parameter addition is necessary, close to 1 the addition is unnecessary,
    # and NaN means the fit got worse when the parameter was added

    # if the Ftest returns NaN (fit got worse), iterate the fit until it improves to a max of 3 iterations.
    # It may have gotten stuck in a local minima
    counter = 0

    while not Ftest_p and counter < 3:
        counter += 1

        f_plus_p.fit_toas()
        m_plus_p_rs = pint.residuals.Residuals(toas, f_plus_p.model)

        # recalculate the Ftest
        Ftest_p = pint.utils.FTest(
            float(m_rs.chi2), m_rs.dof, float(m_plus_p_rs.chi2), m_plus_p_rs.dof
        )

    # print the Ftest for the parameter and return the value of the Ftest
    print("Ftest" + param_name + ":", Ftest_p)
    return Ftest_p


def Ftest_param_phases(r_model, fitter, param_name):
    """
    do an Ftest comparing a model with and without a particular parameter added for a phase wrapped model (use track_mode='use_pulse_numbers')

    Note: this is NOT a general use function - it is specific to this code and cannot be easily adapted to other scripts
    :param r_model: timing model to be compared
    :param fitter: fitter object containing the toas to compare on
    :param param_name: name of the timing model parameter to be compared
    :return
    """
    # read in model and toas
    m_plus_p = deepcopy(r_model)
    toas = deepcopy(fitter.toas)

    # set given parameter to unfrozen
    getattr(m_plus_p, param_name).frozen = False

    # make a fitter object with the chosen parameter unfrozen and fit the toas using the model with the extra parameter
    f_plus_p = pint.fitter.WLSFitter(toas, m_plus_p)
    f_plus_p.fit_toas()

    # calculate the residuals for the fit with (m_plus_p_rs) and without (m_rs) the extra parameter
    m_rs = pint.residuals.Residuals(toas, fitter.model, track_mode="use_pulse_numbers")
    m_plus_p_rs = pint.residuals.Residuals(
        toas, f_plus_p.model, track_mode="use_pulse_numbers"
    )

    # calculate the Ftest, comparing the chi2 and degrees of freedom of the two models
    Ftest_p = pint.utils.FTest(
        float(m_rs.chi2), m_rs.dof, float(m_plus_p_rs.chi2), m_plus_p_rs.dof
    )
    # The Ftest determines how likely (from 0. to 1.) that improvement due to the new parameter is due to chance and not necessity
    # Ftests close to zero mean the parameter addition is necessary, close to 1 the addition is unnecessary,
    # and NaN means the fit got worse when the parameter was added

    # if the Ftest returns NaN (fit got worse), iterate the fit until it improves to a max of 10 iterations.
    # It may have gotten stuck in a local minima
    counter = 0
    while not Ftest_p and counter < 3:
        counter += 1

        f_plus_p.fit_toas()
        m_plus_p_rs = pint.residuals.Residuals(
            toas, f_plus_p.model, track_mode="use_pulse_numbers"
        )

        # recalculate the Ftest
        Ftest_p = pint.utils.FTest(
            float(m_rs.chi2), m_rs.dof, float(m_plus_p_rs.chi2), m_plus_p_rs.dof
        )

    # print the Ftest for the parameter and return the value of the Ftest
    print("Ftest" + param_name + ":", Ftest_p)
    return Ftest_p


def set_F1_lim(args, parfile):
    # if F1_lim not specified in command line, calculate the minimum span based on general F0-F1 relations from P-Pdot diagram

    if args.F1_lim == None:
        # for slow pulsars, allow F1 to be up to 1e-12 Hz/s, for medium pulsars, 1e-13 Hz/s, otherwise, 1e-14 Hz/s (recycled pulsars)
        F0 = mb.get_model(parfile).F0.value

        if F0 < 10:
            F1 = 10**-12

        elif 10 < F0 < 100:
            F1 = 10**-13

        else:
            F1 = 10**-14

        # rearranged equation [delta-phase = (F1*span^2)/2], span in seconds.
        # calculates span (in days) for delta-phase to reach 0.35 due to F1
        args.F1_lim = np.sqrt(0.35 * 2 / F1) / 86400.0


def readin_starting_points(mask, t, start_type, start, args):
    # if given starting points from command line, replace calculated starting points with given starting points (group numbers or mjd values)

    clusters = t.get_clusters()

    if start_type == "clusters":
        mask = np.logical_or(clusters == start[0], clusters == start[1])

    elif start_type == "mjds":
        # TODO: program crashes if no MJDs in the range given
        mask = np.logical_and(
            t.get_mjds() > start[0] * u.d, t.get_mjds() < start[1] * u.d
        )

    # can read in toas from a maskfile (csv) or a saved boolean array
    if args.maskfile != None:
        mask_read = open(args.maskfile, "r")
        data = csv.reader(mask_read)
        mask = [bool(int(row[0])) for row in data]

    return mask


def calc_resid_diff(closest_group, full_clusters, base_TOAs, f, selected):
    # create mask array for closest group (without fit toas)
    selected_closest = [
        True if group == closest_group else False for group in full_clusters
    ]

    # calculate phase resid of last toa of the fit toas and first toa of the closest group.
    last_fit_toa_phase = pint.residuals.Residuals(base_TOAs, f.model).phase_resids[
        selected
    ][-1]
    first_new_toa_phase = pint.residuals.Residuals(base_TOAs, f.model).phase_resids[
        selected_closest
    ][0]

    # Use difference of edge points as difference between clusters as a whole
    diff = first_new_toa_phase - last_fit_toa_phase

    return selected_closest, diff


def bad_points(
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
):
    # try polyfit on next n data, and if works (has resids < 0.001), just ignore it as a bad data group, and fit the next n data clusters instead

    if dist > 0:
        # mask next group to the right
        try_mask = [
            (
                True
                if group in t.get_clusters()
                or group
                in np.arange(
                    closest_group + 1, closest_group + 1 + args.check_bp_n_clusters
                )
                else False
            )
            for group in full_clusters
        ]

    else:
        # mask next group to the left
        try_mask = [
            (
                True
                if group in t.get_clusters()
                or group
                in np.arange(closest_group - args.check_bp_n_clusters, closest_group)
                else False
            )
            for group in full_clusters
        ]

    # try_t is the current subset of TOAs and the next args.check_bp_n_clusters (default = 3) clusters after the closest group, but WITHOUT the closest group included
    try_t = deepcopy(base_TOAs)
    try_t.select(try_mask)
    try_resids = np.float64(pint.residuals.Residuals(try_t, m).phase_resids)
    try_mjds = np.float64(try_t.get_mjds())

    # try fitting the current subset and the next few clusters with a polynomial while ignoring the closest group
    p, resids, q1, q2, q3 = np.polyfit(
        (try_mjds / u.d).to_value(u.dimensionless_unscaled), try_resids, 3, full=True
    )

    if resids.size == 0:
        # means residuals were perfectly 0, which only happens if there isn't enough data to do a proper fit
        resids = [0.0]
        print("phase resids was empty")

    print("Bad Point Check residuals (phase)", resids)

    # right now, bad_group_t is the current subset, plus the closest group, plus the next args.check_bp_n_clusters clusters. This is for plotting purposes
    bad_group_t = deepcopy(base_TOAs)
    bad_group_t.select(bad_group_t.get_mjds() >= min(try_t.get_mjds()))
    bad_group_t.select(bad_group_t.get_mjds() <= max(try_t.get_mjds()))
    bad_group_r = pint.residuals.Residuals(bad_group_t, m).phase_resids

    # define the index of the possibly bad data group
    index = np.where(bad_group_t.get_clusters() == closest_group)

    x = np.arange(min(try_mjds) / u.d, max(try_mjds) / u.d, 2)
    y = p[0] * x**3 + p[1] * x**2 + p[2] * x + p[3]

    plt.plot(try_mjds, try_resids, "b.")
    plt.plot(bad_group_t.get_mjds()[index], bad_group_r[index], "r.")
    plt.plot(x, y, "g-")
    plt.grid()
    plt.xlabel("MJD")
    plt.ylabel("phase resids")
    plt.title("Checking Bad Point")
    if args.plot_bad_points == True:
        plt.show()

    else:
        plt.savefig(f"./alg_saves/{sys_name}/{sys_name}_{iteration:03d}_B.png")
        plt.clf()

    if (
        # if the residuals of the polynomial fit are less than the threshold value (meaning the fit without the bad data group was very good), ignore the bad group
        resids[0]
        < args.check_bp_max_resid
    ):
        print("Ignoring Bad Data Point, will not attempt phase wraps this iteration")
        bad_mjds.append(bad_group_t.get_mjds()[index])
        t_others = deepcopy(try_t)
        mask = [
            True if group in t_others.get_clusters() else False
            for group in full_clusters
        ]
        skip_phases = True

    return skip_phases, t_others, mask, bad_mjds


def poly_extrap(
    minmjd, maxmjd, args, dist, base_TOAs, t_others, full_clusters, m, mask
):
    # polynomial extrapolation script, calls poly_extrap1-3, returns t_others and mask with added possible points
    resids, try_span1, try_t = poly_extrap1(
        minmjd, maxmjd, args, dist, base_TOAs, t_others, full_clusters, m
    )

    if resids[0] < args.pe_max_resid:
        # go ahead and fit on all those days
        # try with even bigger span
        resids2, try_span2, try_t2 = poly_extrap2(
            minmjd, maxmjd, args, dist, base_TOAs, t_others, full_clusters, m
        )

        if resids2[0] < args.pe_max_resid:
            # go ahead and fit on all those days
            # try with even bigger span
            resids3, try_span3, try_t3 = poly_extrap3(
                minmjd, maxmjd, args, dist, base_TOAs, t_others, full_clusters, m
            )

            if resids3[0] < args.pe_max_resid:
                print("Fitting points from", minmjd, "to", minmjd + try_span3)
                t_others = deepcopy(try_t3)
                mask = [
                    True if group in t_others.get_clusters() else False
                    for group in full_clusters
                ]

            else:
                print("Fitting points from", minmjd, "to", minmjd + try_span2)
                t_others = deepcopy(try_t2)
                mask = [
                    True if group in t_others.get_clusters() else False
                    for group in full_clusters
                ]

        else:
            # and repeat all above until get bad resids, then do else and the below
            print("Fitting points from", minmjd, "to", minmjd + try_span1)
            t_others = deepcopy(try_t)
            mask = [
                True if group in t_others.get_clusters() else False
                for group in full_clusters
            ]

    # END INDENT OF IF_ELSEs
    return t_others, mask


def poly_extrap1(minmjd, maxmjd, args, dist, base_TOAs, t_others, full_clusters, m):
    # function to calculate poly_extrap at first level

    try_span1 = args.span1_c * (maxmjd - minmjd)
    print("Trying polynomial extrapolation on span", try_span1)

    new_t = deepcopy(base_TOAs)

    if dist > 0:
        # next data is to the right
        new_t.select(new_t.get_mjds() > maxmjd)
        new_t.select(new_t.get_mjds() < minmjd + try_span1)

    else:
        # next data is to the left
        new_t.select(new_t.get_mjds() < minmjd)
        new_t.select(new_t.get_mjds() > maxmjd - try_span1)

    # try_t now includes all the TOAs to be fit by polyfit but are not included in t_others
    try_mask = [
        (
            True
            if group in t_others.get_clusters() or group in new_t.get_clusters()
            else False
        )
        for group in full_clusters
    ]
    try_t = deepcopy(base_TOAs)
    try_t.select(try_mask)
    try_resids = np.float64(pint.residuals.Residuals(try_t, m).phase_resids)
    try_mjds = np.float64(try_t.get_mjds())

    p, resids, q1, q2, q3 = np.polyfit(
        (try_mjds / u.d).to_value(u.dimensionless_unscaled), try_resids, 3, full=True
    )

    if resids.size == 0:
        # shouldnt happen if make it wait until more than a week of data
        resids = [0.0]
        print("resids was empty")

    print("PE residuals span 1 (phase)", resids)

    if args.plot_poly_extrap == True:
        x = np.arange(min(try_mjds) / u.d, max(try_mjds) / u.d, 2)
        y = p[0] * x**3 + p[1] * x**2 + p[2] * x + p[3]
        plt.plot(try_mjds, try_resids, "b.")
        plt.plot(x, y, "g-")
        plt.grid()
        plt.xlabel("MJD")
        plt.ylabel("phase resids")
        plt.show()

    return resids, try_span1, try_t


def poly_extrap2(minmjd, maxmjd, args, dist, base_TOAs, t_others, full_clusters, m):
    # function to calculate poly_extrap at second level

    try_span2 = args.span2_c * (maxmjd - minmjd)
    print("Trying polynomial extrapolation on span", try_span2)

    new_t2 = deepcopy(base_TOAs)

    if dist > 0:
        # next data is to the right
        new_t2.select(new_t2.get_mjds() > maxmjd)
        new_t2.select(new_t2.get_mjds() < minmjd + try_span2)

    else:
        # next data is to the left
        new_t2.select(new_t2.get_mjds() < minmjd)
        new_t2.select(new_t2.get_mjds() > maxmjd - try_span2)

    # try_t now includes all the TOAs to be fit by polyfit but are not included in t_others
    try_mask2 = [
        (
            True
            if group in t_others.get_clusters() or group in new_t2.get_clusters()
            else False
        )
        for group in full_clusters
    ]
    try_t2 = deepcopy(base_TOAs)
    try_t2.select(try_mask2)
    try_resids2 = np.float64(pint.residuals.Residuals(try_t2, m).phase_resids)
    try_mjds2 = np.float64(try_t2.get_mjds())

    p, resids2, q1, q2, q3 = np.polyfit(
        (try_mjds2 / u.d).to_value(u.dimensionless_unscaled), try_resids2, 3, full=True
    )

    if resids2.size == 0:
        # shouldnt happen if make it wait until more than a week of data
        resids2 = [0.0]
        print("resids was empty")

    print("PE residuals span 2 (phase)", resids2)

    if args.plot_poly_extrap == True:
        x = np.arange(min(try_mjds2) / u.d, max(try_mjds2) / u.d, 2)
        y = p[0] * x**3 + p[1] * x**2 + p[2] * x + p[3]
        plt.plot(try_mjds2, try_resids2, "b.")
        plt.plot(x, y, "k-")
        plt.grid()
        plt.xlabel("MJD")
        plt.ylabel("phase resids")
        plt.show()

    return resids2, try_span2, try_t2


def poly_extrap3(minmjd, maxmjd, args, dist, base_TOAs, t_others, full_clusters, m):
    # function to calculate poly_extrap at third and final level

    try_span3 = args.span3_c * (maxmjd - minmjd)
    print("Trying polynomial extrapolation on span", try_span3)

    new_t3 = deepcopy(base_TOAs)

    if dist > 0:
        # next data is to the right
        new_t3.select(new_t3.get_mjds() > maxmjd)
        new_t3.select(new_t3.get_mjds() < minmjd + try_span3)

    else:
        # next data is to the left
        new_t3.select(new_t3.get_mjds() < minmjd)
        new_t3.select(new_t3.get_mjds() > maxmjd - try_span3)

    # try_t now includes all the TOAs to be fit by polyfit but are not included in t_others
    try_mask3 = [
        (
            True
            if group in t_others.get_clusters() or group in new_t3.get_clusters()
            else False
        )
        for group in full_clusters
    ]
    try_t3 = deepcopy(base_TOAs)
    try_t3.select(try_mask3)
    try_resids3 = np.float64(pint.residuals.Residuals(try_t3, m).phase_resids)
    try_mjds3 = np.float64(try_t3.get_mjds())

    p, resids3, q1, q2, q3 = np.polyfit(
        (try_mjds3 / u.d).to_value(u.dimensionless_unscaled), try_resids3, 3, full=True
    )

    if resids3.size == 0:
        # shouldnt happen if make it wait until more than a week of data
        resids3 = [0.0]
        print("resids was empty")

    print("PE residuals span 3 (phase)", resids3)

    if args.plot_poly_extrap == True:
        x = np.arange(min(try_mjds3) / u.d, max(try_mjds3) / u.d, 2)
        y = p[0] * x**3 + p[1] * x**2 + p[2] * x + p[3]
        plt.plot(try_mjds3, try_resids3, "b.")
        plt.plot(x, y, "m-")
        plt.grid()
        plt.xlabel("MJD")
        plt.ylabel("phase resids")
        plt.show()

    return resids3, try_span3, try_t3


def plot_wraps(
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
):
    # plot with phase wrap

    chi2_summary = []
    chi2_ext_summary = []

    model0 = deepcopy(f.model)

    chi2_summary.append(f.resids.chi2)
    chi2_ext_summary.append(
        pint.residuals.Residuals(
            t_others_phases[-1], f.model, track_mode="use_pulse_numbers"
        ).chi2
    )

    fig, ax = plt.subplots(constrained_layout=True)

    # t_phases[-1] is full toas with phase wrap
    # t_others_phases[-1] is selected toas plus closest group with phase wrap
    for i in range(len(rmods)):
        chi2_summary.append(
            pint.residuals.Residuals(
                t_phases[-1], rmods[i], track_mode="use_pulse_numbers"
            ).chi2
        )
        chi2_ext_summary.append(
            pint.residuals.Residuals(
                t_others_phases[-1], rmods[i], track_mode="use_pulse_numbers"
            ).chi2
        )
        ax.plot(f_toas, rss[i], "-k", alpha=0.6)

    # print summary of chi squared values
    print("RANDOM MODEL SUMMARY:")
    print("chi2 median on fit TOAs:", np.median(chi2_summary))
    print("chi2 median on fit TOAs plus closest group:", np.median(chi2_ext_summary))
    print("chi2 stdev on fit TOAs:", np.std(chi2_summary))
    print("chi2 stdev on fit TOAs plus closest group:", np.std(chi2_ext_summary))

    # plot post fit residuals with error bars
    xt = t_phases[-1].get_mjds()
    ax.errorbar(
        xt.value,
        pint.residuals.Residuals(t_phases[-1], model0, track_mode="use_pulse_numbers")
        .time_resids.to(u.us)
        .value,
        t_phases[-1].get_errors().to(u.us).value,
        fmt=".b",
        label="post-fit",
    )

    # string of fit params for plot title
    fitparams = ""
    for param in f.get_fitparams().keys():
        fitparams += str(param) + " "

    # notate pulsar name, iteration number, phase wrap, and parameters that have been fit
    plt.title(
        f"{m.PSR.value} Post-Fit Residuals {iteration} P{wrap} | fit params: {fitparams}"
    )
    ax.set_xlabel("MJD")
    ax.set_ylabel("Residual (us)")
    r = (
        pint.residuals.Residuals(t_phases[-1], model0, track_mode="use_pulse_numbers")
        .time_resids.to(u.us)
        .value
    )

    # set the y limits to just above and below the highest and lowest points
    yrange = abs(max(r) - min(r))
    ax.set_ylim(max(r) + 0.1 * yrange, min(r) - 0.1 * yrange)
    width = max(f_toas).value - min(f_toas).value

    # if the random lines are within the minimum and maximum toas, scale to the edges of the random models
    if (min(f_toas).value - 0.1 * width) < (min(xt).value - 20) or (
        max(f_toas).value + 0.1 * width
    ) > (max(xt).value + 20):
        ax.set_xlim(min(xt).value - 20, max(xt).value + 20)

    # otherwise scale to include all the toas
    else:
        ax.set_xlim(min(f_toas).value - 0.1 * width, max(f_toas).value + 0.1 * width)

    plt.grid()

    def us_to_phase(x):
        return (x / (10**6)) * f.model.F0.value

    def phase_to_us(y):
        return (y / f.model.F0.value) * (10**6)

    # include a secondary axis for phase
    secaxy = ax.secondary_yaxis("right", functions=(us_to_phase, phase_to_us))
    secaxy.set_ylabel("residuals (phase)")

    # save the image in alg_saves with the iteration and wrap number
    plt.savefig(f"./alg_saves/{sys_name}/{sys_name}_{iteration:03d}_P{wrap:03d}.png")
    plt.close()


def plot_plain(
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
):
    # plot post fit residuals with error bars

    model0 = deepcopy(f.model)

    xt = t.get_mjds()
    ax.errorbar(
        xt.value,
        pint.residuals.Residuals(t, model0).time_resids.to(u.us).value,
        t.get_errors().to(u.us).value,
        fmt=".b",
        label="post-fit",
    )

    # string of fit parameters for plot title
    fitparams = ""
    for param in f.get_fitparams().keys():
        fitparams += str(param) + " "

    # notate the pulsar name, iteration, and fit parameters
    plt.title(f"{m.PSR.value} Post-Fit Residuals {iteration} | fit params: {fitparams}")
    ax.set_xlabel("MJD")
    ax.set_ylabel("Residual (us)")
    r = pint.residuals.Residuals(t, model0).time_resids.to(u.us).value

    # set the y limit to be just above and below the max and min points
    yrange = abs(max(r) - min(r))
    ax.set_ylim(max(r) + 0.1 * yrange, min(r) - 0.1 * yrange)

    # scale to the edges of the points or the edges of the random models, whichever is smaller
    width = max(f_toas).value - min(f_toas).value
    if (min(f_toas).value - 0.1 * width) < (min(xt).value - 20) or (
        max(f_toas).value + 0.1 * width
    ) > (max(xt).value + 20):
        ax.set_xlim(min(xt).value - 20, max(xt).value + 20)

    else:
        ax.set_xlim(min(f_toas).value - 0.1 * width, max(f_toas).value + 0.1 * width)

    plt.grid()

    def us_to_phase(x):
        return (x / (10**6)) * f.model.F0.value

    def phase_to_us(y):
        return (y / f.model.F0.value) * (10**6)

    # include secondary axis to show phase
    secaxy = ax.secondary_yaxis("right", functions=(us_to_phase, phase_to_us))
    secaxy.set_ylabel("residuals (phase)")

    plt.savefig(f"./alg_saves/{sys_name}/{sys_name}_{iteration:03d}.png")
    plt.close()


def do_Ftests(t, m, args):
    # perform Ftests on all necessary parameters

    # fit toas with new model
    f = pint.fitter.WLSFitter(t, m)
    f.fit_toas()

    # calculate the span of fit toas for comparison to minimum parameter spans
    span = f.toas.get_mjds().max() - f.toas.get_mjds().min()
    print("Current fit TOAs span:", span)

    Ftests = dict()
    f_params = []
    # TODO: need to take into account if param isn't setup in model yet

    # make list of already fit parameters
    for param in m.params:
        if getattr(m, param).frozen == False:
            f_params.append(param)

    # if span is longer than minimum parameter span and parameter hasn't been added yet, do Ftest to see if parameter should be added
    if "RAJ" not in f_params and span > args.RAJ_lim * u.d:
        Ftest_R = Ftest_param(m, f, "RAJ")
        Ftests[Ftest_R] = "RAJ"

    if "DECJ" not in f_params and span > args.DECJ_lim * u.d:
        Ftest_D = Ftest_param(m, f, "DECJ")
        Ftests[Ftest_D] = "DECJ"

    if "F1" not in f_params and span > args.F1_lim * u.d:
        Ftest_F = Ftest_param(m, f, "F1")
        Ftests[Ftest_F] = "F1"

    # remove possible boolean elements from Ftest returning False if chi2 increases
    Ftests_keys = [key for key in Ftests.keys() if type(key) != bool]

    # if no Ftests performed, continue on without change
    if not bool(Ftests_keys):
        if span > 100 * u.d:
            print("F0, RAJ, DECJ, and F1 have all been added")

    # if smallest Ftest of those calculated is less than the given limit, add that parameter to the model. Otherwise add no parameters
    elif min(Ftests_keys) < args.Ftest_lim:
        add_param = Ftests[min(Ftests_keys)]
        print("adding param ", add_param, " with Ftest ", min(Ftests_keys))
        getattr(m, add_param).frozen = False

    return m


def do_Ftests_phases(m_phases, t_phases, f_phases, args):
    # calculate Ftests for a model with phase wraps

    # calculate the span of the fit toas to compare to minimum spans for parameters
    span = f_phases[-1].toas.get_mjds().max() - f_phases[-1].toas.get_mjds().min()
    print("Current Fit TOA Span:", span)

    Ftests_phase = dict()
    f_params_phase = []
    # TODO: need to take into account if param isn't setup in model yet

    # make a list of all the fit params
    for param in m_phases[-1].params:
        if getattr(m_phases[-1], param).frozen == False:
            f_params_phase.append(param)

    # if a given parameter has not already been fit (in fit_params) and span > minimum fitting span for that param, do an Ftest for that param
    if "RAJ" not in f_params_phase and span > args.RAJ_lim * u.d:
        Ftest_R_phase = Ftest_param_phases(m_phases[-1], f_phases[-1], "RAJ")
        Ftests_phase[Ftest_R_phase] = "RAJ"

    if "DECJ" not in f_params_phase and span > args.DECJ_lim * u.d:
        Ftest_D_phase = Ftest_param_phases(m_phases[-1], f_phases[-1], "DECJ")
        Ftests_phase[Ftest_D_phase] = "DECJ"

    if "F1" not in f_params_phase and span > args.F1_lim * u.d:
        Ftest_F_phase = Ftest_param_phases(m_phases[-1], f_phases[-1], "F1")
        Ftests_phase[Ftest_F_phase] = "F1"

    # remove possible boolean elements from Ftest returning False if chi2 increases
    Ftests_phase_keys = [key for key in Ftests_phase.keys() if type(key) != bool]

    # if nothing in the Ftests list, continue to next step. Print message if long enough span that all params should be added
    if not Ftests_phase_keys:
        if span > 100 * u.d:
            print("F1, RAJ, DECJ, and F1 have been added.")

    # whichever parameter's Ftest is smallest and less than the Ftest limit gets added to the model. Else no parameter gets added
    elif min(Ftests_phase_keys) < args.Ftest_lim:
        add_param = Ftests_phase[min(Ftests_phase_keys)]
        print("adding param ", add_param, " with Ftest ", min(Ftests_phase_keys))
        getattr(m_phases[-1], add_param).frozen = False

    return m_phases[-1]


def calc_random_models(base_TOAs, f, t, args):
    # calculate the random models
    print("\n" * 5, end="#" * 40)
    print()
    print("clusters" in t.table.columns)
    base_TOAs.table["clusters"] = base_TOAs.get_clusters()
    full_clusters = base_TOAs.table["clusters"]
    t.table["clusters"] = t.get_clusters()

    # create a mask which produces the current subset of toas
    selected = [
        True if cluster in t.table["clusters"] else False for cluster in full_clusters
    ]

    base_TOAs_copy = deepcopy(base_TOAs)

    # compute the pulse numbers and set delta_pulse_numbers to zero so plots correctly
    base_TOAs_copy.compute_pulse_numbers(f.model)
    base_TOAs_copy.table["delta_pulse_number"] = np.zeros(
        len(base_TOAs_copy.get_mjds())
    )

    # calculate the average phase resid of the fit toas
    rs_mean = (
        pint.residuals.Residuals(base_TOAs_copy, f.model).phase_resids[selected].mean()
    )

    # produce several (n_pred) random models given the fitter object and mean residual.
    # return the random models, their residuals, and evenly spaced toas to plot against
    f_toas, rss, rmods = pint.random_models.random_models(
        f,
        rs_mean,
        iter=args.n_pred,
        ledge_multiplier=args.ledge_multiplier,
        redge_multiplier=args.redge_multiplier,
    )

    return full_clusters, selected, rs_mean, f_toas.get_mjds(), rss, rmods


def save_state(m, t, mask, sys_name, iteration, base_TOAs, data_path, original_path):
    # save the system state

    last_model = deepcopy(m)
    last_t = deepcopy(t)
    last_mask = deepcopy(mask)

    # write these to a par, tim and txt file to be saved and reloaded
    par_pntr = open(
        "./alg_saves/" + sys_name + "/" + sys_name + "_" + str(iteration) + ".par", "w"
    )
    mask_pntr = open(
        "./alg_saves/" + sys_name + "/" + sys_name + "_" + str(iteration) + ".csv", "w"
    )
    par_pntr.write(m.as_parfile())

    mask_string = ""
    for item in mask:
        mask_string += str(int(item)) + "\n"

    mask_pntr.write(mask_string)  # list to string

    base_TOAs.write_TOA_file(
        "./alg_saves/" + sys_name + "/" + sys_name + "_" + str(iteration) + ".tim",
        format="TEMPO2",
    )

    par_pntr.close()
    mask_pntr.close()

    return last_model, last_t, last_mask


def solution_compare(parfile1: Path, parfile2: Path, timfile: Path) -> bool:
    """
    Compares two solutions to see if identical solution, where identical means they identify
    the same pulse number for each TOA.
    """
    m1, t1 = pint.models.get_model_and_toas(parfile1, timfile)
    t1.compute_pulse_numbers(m1)
    pn1 = t1.table["pulse_number"]

    m2, t2 = pint.models.get_model_and_toas(parfile2, timfile)
    t2.compute_pulse_numbers(m2)
    pn2 = t2.table["pulse_number"]

    return np.array_equal(pn1, pn2)

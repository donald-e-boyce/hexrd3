"""Grain fitting functions"""

from enum import IntEnum

import numpy as np
from scipy import optimize

from hexrd import matrixutil as mutil
from hexrd.transforms import xfcapi
from hexrd import constants
from hexrd import rotations

from hexrd.xrdutil import (
    apply_correction_to_wavelength,
    extract_detector_transformation,
)


class ReturnValue(IntEnum):
    RESIDUAL_SUMABS = 1
    CHISQ_ONLY = 2
    CHISQ_PLUS = 3


return_value_flag = None

epsf = np.finfo(float).eps  # ~2.2e-16
sqrt_epsf = np.sqrt(epsf)  # ~1.5e-8

bVec_ref = constants.beam_vec
eta_ref = constants.eta_vec
vInv_ref = np.r_[1., 1., 1., 0., 0., 0.]


# for grain parameters
gFlag_ref = np.ones(12, dtype=bool)
gScl_ref = np.ones(12, dtype=bool)

np.set_printoptions(precision=4, suppress=True)



def fitGrain(gFull, instrument, reflections_dict,
             bMat, wavelength,
             gFlag=gFlag_ref, gScl=gScl_ref,
             omePeriod=None,
             factor=0.1, xtol=sqrt_epsf, ftol=sqrt_epsf):
    """
    Pefrorm least-squares optimization of grain parameters.

    Parameters
    ----------

    Raises
    ------
    RuntimeError

    Returns
    -------

    """
    # FIXME: will currently fail if omePeriod is specifed
    if omePeriod is not None:
        # xyo_det[:, 2] = rotations.mapAngle(xyo_det[:, 2], omePeriod)
        raise RuntimeError

    gFit = gFull[gFlag]

    # objFuncFitGrain can run *significantly* faster if we convert the
    # results to use a dictionary instead of lists or numpy arrays.
    # Do that conversion here, if necessary.
    new_reflections_dict = {}
    for det_key, results in reflections_dict.items():
        if not isinstance(results, (list, np.ndarray)) or len(results) == 0:
            # Maybe it's already a dict...
            new_reflections_dict[det_key] = results
            continue

        if isinstance(results, list):
            hkls = np.atleast_2d(
                np.vstack([x[2] for x in results])
            ).T
            meas_xyo = np.atleast_2d(
                np.vstack([np.r_[x[7], x[6][-1]] for x in results])
            )
        else:
            hkls = np.atleast_2d(results[:, 2:5]).T
            meas_xyo = np.atleast_2d(results[:, [15, 16, 12]])

        new_reflections_dict[det_key] = {
            'hkls': hkls,
            'meas_xyo': meas_xyo,
        }

    fitArgs = (gFull, gFlag, instrument, new_reflections_dict,
               bMat, wavelength, omePeriod)
    results = optimize.leastsq(
        objFuncFitGrain, gFit, args=fitArgs,
        diag=1./gScl[gFlag].flatten(),
        factor=0.1, xtol=xtol, ftol=ftol,
        full_output=True
    )

    gFit_opt = results[0]

    # debug: Fix this !
    fvec = results[2]['fvec']
    npts = len(fvec) // 3
    fvec_2d = fvec.reshape(npts, 3)
    diff_vecs_xy = fvec_2d[:, :2]
    diff_ome = fvec_2d[:, 2]
    diff_ome_deg = np.degrees(diff_ome)
    rms = np.sqrt(np.sum(fvec * fvec) /npts)
    rms_dxy = np.sqrt(np.sum(diff_vecs_xy ** 2)/npts)
    rms_distxy = np.sqrt(np.sum(dist_xy ** 2)/npts)
    rms_dom = np.sqrt(np.sum(diff_ome ** 2)/npts)
    # gidstr = f"gid: {grainid}:"

    retval = gFull
    retval[gFlag] = gFit_opt
    return retval


def objFuncFitGrain(gFit, gFull, gFlag,
                    instrument,
                    reflections_dict,
                    bMat, wavelength,
                    omePeriod,
                    simOnly=False,
                    return_value_flag=return_value_flag):
    """Least squares residual for grain fitting


    PARAMETERS
    ----------
    gFit :

    gFull :

    gFlag :

    instrument : HEDMInstrument instance
        the HEDM instrument
    reflections_dict : dict
        dictionary of reflections for each panel
    bMat : array()
        matrix for converting ... to spatial coordinates
    wavelength : float
        wavelength of beam
    omePeriod :

    simOnly : bool, default=False
        simulate only
    return_value_flag : int
        indicating what to return

    RETURNS
    -------
    retval :
        depends on return flag

    RAISES
    ------
    RuntimeError

    """
    bVec = instrument.beam_vector
    eVec = instrument.eta_vector
    tVec_s = instrument.tvec
    energy_correction = instrument.energy_correction

    # fill out parameters
    gFull[gFlag] = gFit

    # map parameters to functional arrays
    rMat_c = xfcapi.make_rmat_of_expmap(gFull[:3])
    tVec_c = gFull[3:6].reshape(3, 1)
    cen_r = np.sqrt(tVec_c[0] ** 2 + tVec_c[2] ** 2)

    vInv_s = gFull[6:]
    vMat_s = mutil.vecMVToSymm(vInv_s)  # NOTE: Inverse of V from F = V * R

    # Apply an energy correction according to grain position
    corrected_wavelength = apply_correction_to_wavelength(
        wavelength,
        energy_correction,
        tVec_s,
        tVec_c,
    )

    # loop over instrument panels
    # CAVEAT: keeping track of key ordering in the "detectors" attribute of
    # instrument here because I am not sure if instatiating them using
    # dict.fromkeys() preserves the same order if using iteration...
    # <JVB 2017-10-31>
    calc_omes_dict = dict.fromkeys(instrument.detectors, [])
    calc_xy_dict = dict.fromkeys(instrument.detectors)
    meas_xyo_all = []
    det_keys_ordered = []
    for det_key, panel in instrument.detectors.items():
        det_keys_ordered.append(det_key)

        rMat_d, tVec_d, chi, tVec_s = extract_detector_transformation(
            instrument.detector_parameters[det_key])

        results = reflections_dict[det_key]

        if not isinstance(results, dict) and len(results) == 0:
            continue

        """
        extract data from results list fields:
          refl_id, gvec_id, hkl, sum_int, max_int, pred_ang, meas_ang, meas_xy

        or array from spots tables:
          0:5    ID    PID    H    K    L
          5:7    sum(int)    max(int)
          7:10   pred tth    pred eta    pred ome
          10:13  meas tth    meas eta    meas ome
          13:15  pred X    pred Y
          15:17  meas X    meas Y
        """
        if isinstance(results, list):
            # WARNING: hkls and derived vectors below must be columnwise;
            # strictly necessary??? change affected APIs instead?
            # <JVB 2017-03-26>
            hkls = np.atleast_2d(
                np.vstack([x[2] for x in results])
            ).T

            meas_xyo = np.atleast_2d(
                np.vstack([np.r_[x[7], x[6][-1]] for x in results])
            )
        elif isinstance(results, np.ndarray):
            hkls = np.atleast_2d(results[:, 2:5]).T
            meas_xyo = np.atleast_2d(results[:, [15, 16, 12]])
        elif isinstance(results, dict):
            hkls = results['hkls']
            meas_xyo = results['meas_xyo']

        # distortion handling
        if panel.distortion is not None:
            meas_omes = meas_xyo[:, 2]
            xy_unwarped = panel.distortion.apply(meas_xyo[:, :2])
            meas_xyo = np.vstack([xy_unwarped.T, meas_omes]).T

        # append to meas_omes
        meas_xyo_all.append(meas_xyo)

        # G-vectors:
        #   1. calculate full g-vector components in CRYSTAL frame from B
        #   2. rotate into SAMPLE frame and apply stretch
        #   3. rotate back into CRYSTAL frame and normalize to unit magnitude
        # IDEA: make a function for this sequence of operations with option for
        # choosing ouput frame (i.e. CRYSTAL vs SAMPLE vs LAB)
        gVec_c = np.dot(bMat, hkls)
        gVec_s = np.dot(vMat_s, np.dot(rMat_c, gVec_c))
        gHat_c = mutil.unitVector(np.dot(rMat_c.T, gVec_s))

        # !!!: check that this operates on UNWARPED xy
        match_omes, calc_omes = matchOmegas(
            meas_xyo, hkls, chi, rMat_c, bMat, corrected_wavelength,
            vInv=vInv_s, beamVec=bVec, etaVec=eVec,
            omePeriod=omePeriod)

        # append to omes dict
        calc_omes_dict[det_key] = calc_omes

        # TODO: try Numba implementations
        rMat_s = xfcapi.make_sample_rmat(chi, calc_omes)
        calc_xy = xfcapi.gvec_to_xy(gHat_c.T,
                                    rMat_d, rMat_s, rMat_c,
                                    tVec_d, tVec_s, tVec_c,
                                    beam_vec=bVec)

        # append to xy dict
        calc_xy_dict[det_key] = calc_xy

    # stack results to concatenated arrays
    calc_omes_all = np.hstack([calc_omes_dict[k] for k in det_keys_ordered])
    tmp = []
    for k in det_keys_ordered:
        if calc_xy_dict[k] is not None:
            tmp.append(calc_xy_dict[k])
    calc_xy_all = np.vstack(tmp)
    meas_xyo_all = np.vstack(meas_xyo_all)

    npts = len(meas_xyo_all)
    if np.any(np.isnan(calc_xy)):
        raise RuntimeError(
            "infeasible pFull: may want to scale" +
            "back finite difference step size")

    # return values
    if simOnly:
        # return simulated values
        if return_value_flag in [None, 1]:
            retval = np.hstack([calc_xy_all, calc_omes_all.reshape(npts, 1)])
        else:
            rd = dict.fromkeys(det_keys_ordered)
            for det_key in det_keys_ordered:
                rd[det_key] = {'calc_xy': calc_xy_dict[det_key],
                               'calc_omes': calc_omes_dict[det_key]}
            retval = rd
    else:
        # return residual vector
        # IDEA: try angles instead of xys?
        diff_vecs_xy = calc_xy_all - meas_xyo_all[:, :2]
        diff_ome = rotations.angularDifference(
            calc_omes_all, meas_xyo_all[:, 2]
        )
        # DEBUGGING
        # Print stats on diff_ome and prompt to continue
        _tmp = np.degrees(diff_ome)
        _stats = np.array([_tmp.max(), _tmp.min(), _tmp.mean()])

        DEBUG = False
        OMEOFFSET = 0.0
        PENALTY = cen_r
        if DEBUG:
            retval = np.hstack([
                diff_vecs_xy,
                PENALTY * (diff_ome - OMEOFFSET).reshape(npts, 1)

            ]).flatten()
        else:
            retval = np.hstack(
                [diff_vecs_xy,
                 diff_ome.reshape(npts, 1)
                 ]
            ).flatten()

        # Standard chisq return value.
        resid_ssq = np.sum(retval ** 2)
        denom = 3 * npts - len(gFit) - 1.0
        if denom != 0:
            nu_fac = 1. / denom
        else:
            nu_fac = 1.
        chisq = nu_fac * resid_ssq

        # Other info.
        rms = np.sqrt(resid_ssq / npts)
        rmsxy = np.sqrt(np.sum(diff_vecs_xy ** 2 / npts))
        omestats = (diff_ome.max(), diff_ome.min(), diff_ome.mean())

        if return_value_flag == ReturnValue.RESIDUAL_SUMABS:
            # return scalar sum of squared residuals
            retval = sum(abs(retval))

        elif return_value_flag == ReturnValue.CHISQ_ONLY:
            # return DOF-normalized chisq
            retval = chisq

        elif return_value_flag == ReturnValue.CHISQ_PLUS:
            np.save("diff-ome", diff_ome)
            rms = np.sqrt(resid_ssq / npts)
            retval = (chisq, (npts, rms, rmsxy, np.array(omestats)))

    return retval


def matchOmegas(xyo_det, hkls_idx, chi, rMat_c, bMat, wavelength,
                vInv=vInv_ref, beamVec=bVec_ref, etaVec=eta_ref,
                omePeriod=None):
    """
    For a given list of (x, y, ome) points, outputs the index into the results
    from oscillAnglesOfHKLs, including the calculated omega values.
    """
    # get omegas for rMat_s calculation
    if omePeriod is not None:
        meas_omes = rotations.mapAngle(xyo_det[:, 2], omePeriod)
    else:
        meas_omes = xyo_det[:, 2]

    oangs0, oangs1 = xfcapi.oscill_angles_of_hkls(
            hkls_idx.T, chi, rMat_c, bMat, wavelength,
            v_inv=vInv,
            beam_vec=beamVec,
            eta_vec=etaVec)
    if np.any(np.isnan(oangs0)):
        # debugging
        # TODO: remove this
        # import pdb
        # pdb.set_trace()
        nanIdx = np.where(np.isnan(oangs0[:, 0]))[0]
        errorString = "Infeasible parameters for hkls:\n"
        for i in range(len(nanIdx)):
            errorString += "%d  %d  %d\n" % tuple(hkls_idx[:, nanIdx[i]])
        errorString += "you may need to deselect this hkl family."
        raise RuntimeError(errorString)
    else:
        # CAPI version gives vstacked angles... must be (2, nhkls)
        calc_omes = np.vstack([oangs0[:, 2], oangs1[:, 2]])
    if omePeriod is not None:
        calc_omes = np.vstack([rotations.mapAngle(oangs0[:, 2], omePeriod),
                               rotations.mapAngle(oangs1[:, 2], omePeriod)])
    # do angular difference
    diff_omes = rotations.angularDifference(
        np.tile(meas_omes, (2, 1)), calc_omes
    )
    match_omes = np.argsort(diff_omes, axis=0) == 0
    calc_omes = calc_omes.T.flatten()[match_omes.T.flatten()]

    return match_omes, calc_omes

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <pineappl_capi.h>

#include "orders.h"

#include "pineappl_maxproc.h"

/*
  fNLO mode of aMCatNLO

  This interface uses the generalized C API of PineAPPL 1.x
  (pineappl_channels_*, pineappl_grid_new2, pineappl_grid_fill2). The
  two beams are treated as two independent convolutions, so that
  asymmetric collisions (different hadrons/nuclei, different PDFs) and
  resolved-photon beams (PDG id 22, with a photon PDF) are supported.
  The convolution of beam i is described by the PDG id stored in
  heprup_.idbmup[i], and the partonic channels are given by the pairs
  (parton from beam 1, parton from beam 2) in appl_common_lumi_.
*/

// Declare grids
std::vector<pineappl_grid*> grid_obs;

// translates an index from the range [0, __amp_split_size) to the indices need by `fill_grid`
std::vector<std::vector<int>> translation_tables;

// Information defined at the generation (configuration) step, that does
// not vary event by event
extern "C" struct
{
    int amp_split_size; // Maximum number of coupling-combinations
    int qcdpower[__amp_split_size]; // Power of alpha_s for each amp_split
    int qedpower[__amp_split_size]; // Power of alpha for each amp_split
} appl_common_fixed_;

// Map of the PDF combinations from aMC@NLO - structure for each
// "subprocess" i, has some number nproc[i] pairs of parton
// combinations. To be used together with the info in appl_flavmap.
// The first entry of each pair is the parton of beam 1, the second one
// the parton of beam 2.
extern "C" struct
{
    int lumimap[__max_nproc__][__max_nproc__][2]; // (paired) subprocesses per combination
    int nproc[__max_nproc__]; // number of separate (pairwise) subprocesses for this combination
    int nlumi; // overall number of combinations ( 0 < nlumi <= __mxpdflumi__ )
} appl_common_lumi_;

// Event weights, kinematics, etc. that are different event by event
extern "C" struct
{
    double x1[4], x2[4];
    double muF2[4], muR2[4], muQES2[4];
    double W0[__amp_split_size][4], WR[__amp_split_size][4];
    double WF[__amp_split_size][4], WB[__amp_split_size][4];
    int flavmap[4];
} appl_common_weights_;

// Parameters of the grids.
// These parameters can optionally be singularly specified by the user,
// but if no specification is given, the code will use the default values.
// The x parameters given here are used for both beams, unless they are
// overridden beam by beam in appl_common_grid_beam_ (see below).
extern "C" struct
{
    double Q2min, Q2max;
    double xmin, xmax;
    int nQ2, Q2order;
    int nx, xorder;
} appl_common_grid_;

// Optional beam-by-beam parameters of the x interpolation (index 0 is
// beam 1, index 1 is beam 2). A non-positive value means "use the value
// in appl_common_grid_ (or the default if that is not set either)".
extern "C" struct
{
    double xmin[2], xmax[2];
    int nx[2], xorder[2];
} appl_common_grid_beam_;

// Parameters of the histograms
extern "C" struct
{
    double www_histo, norm_histo;
    double obs_histo;
    double obs_min, obs_max;
    double obs_bins[101];
    int obs_nbins;
    int itype_histo;
    int amp_pos;
    int obs_num;
} appl_common_histokin_;

// Event weight and cross section
extern "C" struct
{
    double event_weight, vegaswgt;
    double xsec12, xsec11, xsec20;
} appl_common_reco_;

extern "C" struct
{
    int idbmup[2];
    double ebmup[2];
    int pdfgup[2];
    int pdfsup[2];
    int idwtup;
    int nprup;
    double xsecup[100];
    double xerrup[100];
    double xmaxup[100];
    int lprup[100];
} heprup_;

namespace
{

// Default interpolation settings. They are the same that the legacy
// (pre-1.0) PineAPPL interface used, so grids stay comparable.
constexpr double default_q2min = 1.0e2;
constexpr double default_q2max = 1.0e8;
constexpr int default_nq2 = 40;
constexpr int default_q2order = 3;
constexpr double default_xmin = 2.0e-7;
constexpr double default_xmax = 1.0;
constexpr int default_nx = 50;
constexpr int default_xorder = 3;

// hbar^2 c^2 in GeV^2 pb, as used in the rest of aMC@NLO
constexpr double hbarc2_pb = 389379660.0;

// Interpolation ranges that were actually used to build a grid, together
// with counters of the fills that PineAPPL silently discards because
// they fall outside of these ranges.
struct GridInfo
{
    double q2min, q2max;
    double xmin[2], xmax[2];
    unsigned long n_fills = 0;
    unsigned long n_out_scale = 0;
    unsigned long n_out_x[2] = { 0, 0 };
    double sum_abs_wgt = 0.0;
    double sum_abs_wgt_lost = 0.0;
};

std::vector<GridInfo> grid_info;

// Returns the value requested for beam `ibeam`: the beam-specific one if
// set, otherwise the common one if set, otherwise the default.
double pick(double beam_value, double common_value, double default_value)
{
    if (beam_value > 0.0)
    {
        return beam_value;
    }
    if (common_value > 0.0)
    {
        return common_value;
    }
    return default_value;
}

int pick(int beam_value, int common_value, int default_value)
{
    if (beam_value > 0)
    {
        return beam_value;
    }
    if (common_value > 0)
    {
        return common_value;
    }
    return default_value;
}

// Index in the arrays {W0, WR, WF, WB} of the event-type itype
int itype_to_k(int itype)
{
    switch (itype)
    {
    case 1:
        return 0;

    case 2:
    case 3:
        return 1;

    case 4:
        return 2;

    case 5:
        return 3;

    default:
        break;
    }

    std::cout << "amcblast ERROR: Invalid value of itype = " << itype << std::endl;
    std::exit(-10);
}

std::string to_string_precise(double value)
{
    std::ostringstream stream;
    stream << std::setprecision(12) << value;
    return stream.str();
}

} // namespace

extern "C" void appl_init_()
{
    // Grid Initialization and definition of the observables.

    int lo_power = 9999;
    int nlo_power = 0;

    for (int i = 0; i != appl_common_fixed_.amp_split_size; ++i)
    {
        int sum = appl_common_fixed_.qcdpower[i] + appl_common_fixed_.qedpower[i];

        lo_power = std::min(lo_power, sum);
        nlo_power = std::max(nlo_power, sum);
    }

    // TODO: are there any situations is which there are NLOs but no LOs?

    // we assume that there is always at least one LO, and zero or one NLO
    assert((nlo_power == (lo_power + 2)) || (nlo_power == lo_power));

    // Each order is described by five numbers: the powers of alphas, alpha,
    // log(xi_R^2), log(xi_F^2) and log(xi_A^2) (fragmentation scale, which
    // is not used in aMC@NLO).
    std::vector<std::uint8_t> order_params;

    auto add_order = [&order_params](int qcd, int qed, int logxir, int logxif) {
        assert((qcd % 2) == 0 && (qed % 2) == 0);
        assert(qcd / 2 <= 255 && qed / 2 <= 255);

        order_params.push_back(static_cast<std::uint8_t>(qcd / 2));
        order_params.push_back(static_cast<std::uint8_t>(qed / 2));
        order_params.push_back(static_cast<std::uint8_t>(logxir));
        order_params.push_back(static_cast<std::uint8_t>(logxif));
        order_params.push_back(static_cast<std::uint8_t>(0));
    };

    translation_tables.emplace_back();
    translation_tables.back().reserve(appl_common_fixed_.amp_split_size);

    for (int i = 0; i != appl_common_fixed_.amp_split_size; ++i)
    {
        int const qcd = appl_common_fixed_.qcdpower[i];
        int const qed = appl_common_fixed_.qedpower[i];
        int const sum = qcd + qed;

        translation_tables.back().push_back(static_cast<int>(order_params.size() / 5));

        if (sum == lo_power)
        {
            // WB
            add_order(qcd, qed, 0, 0);
        }
        else if (sum == nlo_power)
        {
            // W0
            add_order(qcd, qed, 0, 0);
            // WR
            add_order(qcd, qed, 1, 0);
            // WF
            add_order(qcd, qed, 0, 1);
        }
    }

    // Set up the partonic channels. Every channel is a list of pairs of
    // PDG ids (parton of beam 1, parton of beam 2).
    constexpr std::size_t nb_convolutions = 2;
    auto* channels = pineappl_channels_new(nb_convolutions);

    // Loop over parton luminosities
    for (int ilumi = 0; ilumi < appl_common_lumi_.nlumi; ilumi++)
    {
        int nproc = appl_common_lumi_.nproc[ilumi];

        std::vector<int32_t> pdg_ids;
        pdg_ids.reserve(nb_convolutions * nproc);

        for (int iproc = 0; iproc != nproc; ++iproc)
        {
            int32_t a = appl_common_lumi_.lumimap[ilumi][iproc][0];
            int32_t b = appl_common_lumi_.lumimap[ilumi][iproc][1];

            // give the gluon a proper PDG id
            pdg_ids.push_back(a == 0 ? 21 : a);
            pdg_ids.push_back(b == 0 ? 21 : b);
        }

        pineappl_channels_add(channels, nproc, pdg_ids.data(), nullptr);
    }

    // Describe the two convolutions. The PDG id of each beam particle comes
    // from the Les Houches initial-state information: 2212 for a proton,
    // 22 for a (resolved) photon, the nuclear PDG id for an ion, etc.
    // Different ids for the two beams give an asymmetric grid, which is
    // then never symmetrized by PineAPPL.
    for (int ibeam = 0; ibeam != 2; ++ibeam)
    {
        if (heprup_.idbmup[ibeam] == 0)
        {
            std::cout << "amcblast ERROR: the PDG id of beam " << (ibeam + 1)
                      << " is not defined (idbmup = 0)" << std::endl;
            std::exit(-10);
        }
    }

    pineappl_conv convs[nb_convolutions] = {
        { PINEAPPL_CONV_TYPE_UNPOL_PDF, heprup_.idbmup[0] },
        { PINEAPPL_CONV_TYPE_UNPOL_PDF, heprup_.idbmup[1] },
    };

    // The kinematics of the grid: one scale and one momentum fraction for
    // each of the two convolutions. The order matches the `interps` array
    // below and the `ntuple` filled in `appl_fill_`.
    pineappl_kinematics kinematics[3] = {
        { PINEAPPL_KINEMATICS_SCALE, 0 },
        { PINEAPPL_KINEMATICS_X, 0 },
        { PINEAPPL_KINEMATICS_X, 1 },
    };

    // Define the settings for the interpolation in x and Q2.
    // If values larger than zero (i.e. set by the user) are found the default
    // settings are replaced with the new ones. The x settings can be given
    // separately for each beam.
    GridInfo info;

    info.q2min = pick(0.0, appl_common_grid_.Q2min, default_q2min);
    info.q2max = pick(0.0, appl_common_grid_.Q2max, default_q2max);
    int const nq2 = pick(0, appl_common_grid_.nQ2, default_nq2);
    int const q2order = pick(0, appl_common_grid_.Q2order, default_q2order);

    int nx[2], xorder[2];

    for (int ibeam = 0; ibeam != 2; ++ibeam)
    {
        info.xmin[ibeam] =
            pick(appl_common_grid_beam_.xmin[ibeam], appl_common_grid_.xmin, default_xmin);
        info.xmax[ibeam] =
            pick(appl_common_grid_beam_.xmax[ibeam], appl_common_grid_.xmax, default_xmax);
        nx[ibeam] = pick(appl_common_grid_beam_.nx[ibeam], appl_common_grid_.nx, default_nx);
        xorder[ibeam] =
            pick(appl_common_grid_beam_.xorder[ibeam], appl_common_grid_.xorder, default_xorder);
    }

    pineappl_interp interpolations[3] = {
        // scale
        { info.q2min, info.q2max, static_cast<std::size_t>(nq2),
          static_cast<std::size_t>(q2order), PINEAPPL_REWEIGHT_METH_NO_REWEIGHT,
          PINEAPPL_MAP_APPL_GRID_H0, PINEAPPL_INTERP_METH_LAGRANGE },
        // x of beam 1
        { info.xmin[0], info.xmax[0], static_cast<std::size_t>(nx[0]),
          static_cast<std::size_t>(xorder[0]), PINEAPPL_REWEIGHT_METH_APPL_GRID_X,
          PINEAPPL_MAP_APPL_GRID_F2, PINEAPPL_INTERP_METH_LAGRANGE },
        // x of beam 2
        { info.xmin[1], info.xmax[1], static_cast<std::size_t>(nx[1]),
          static_cast<std::size_t>(xorder[1]), PINEAPPL_REWEIGHT_METH_APPL_GRID_X,
          PINEAPPL_MAP_APPL_GRID_F2, PINEAPPL_INTERP_METH_LAGRANGE },
    };

    // The same (factorization) scale is used for the renormalization and
    // factorization scale; the logarithms of the scale variations are
    // stored in the orders. There is no fragmentation scale.
    pineappl_scale_func_form scale_mu = { PINEAPPL_SCALE_FUNC_FORM_SCALE, 0 };
    pineappl_scale_func_form no_scale_mu = { PINEAPPL_SCALE_FUNC_FORM_NO_SCALE, 0 };
    pineappl_scale_func_form mu_scales[3] = { scale_mu, scale_mu, no_scale_mu };

    // Create a grid with the binning given in the "obsbins[Nbins+1]" array
    grid_obs.push_back(pineappl_grid_new2(
        appl_common_histokin_.obs_nbins,
        appl_common_histokin_.obs_bins,
        order_params.size() / 5,
        order_params.data(),
        channels,
        PINEAPPL_PID_BASIS_PDG,
        convs,
        3,
        interpolations,
        kinematics,
        mu_scales
    ));

    // Record the beam energies in the grid: they are not part of the
    // convolution information, but are needed to interpret asymmetric
    // collisions
    pineappl_grid_set_metadata(grid_obs.back(), "mg5amc_ebeam_1",
        to_string_precise(heprup_.ebmup[0]).c_str());
    pineappl_grid_set_metadata(grid_obs.back(), "mg5amc_ebeam_2",
        to_string_precise(heprup_.ebmup[1]).c_str());

    grid_info.push_back(info);

    pineappl_channels_delete(channels);
}

extern "C" void appl_delete_itype_()
{
    // Check that itype ranges from 1 to 5.
    int k = itype_to_k(appl_common_histokin_.itype_histo);

    // this is the second index of the WB/R/F/0 arrays
    int index = appl_common_histokin_.amp_pos - 1;

    // aMC@NLO weights. Four grids, ordered as {W0,WR,WF,WB}.
    double(&W0)[4] = appl_common_weights_.W0[index];
    double(&WR)[4] = appl_common_weights_.WR[index];
    double(&WF)[4] = appl_common_weights_.WF[index];
    double(&WB)[4] = appl_common_weights_.WB[index];

    W0[k] = 0.0;
    WR[k] = 0.0;
    WF[k] = 0.0;
    WB[k] = 0.0;
}

extern "C" void appl_fill_()
{
    // Check that itype ranges from 1 to 5.
    int const k = itype_to_k(appl_common_histokin_.itype_histo);

    // this is the second index of the WB/R/F/0 arrays
    int index = appl_common_histokin_.amp_pos - 1;

    // aMC@NLO weights. Four grids, ordered as {W0,WR,WF,WB}.
    double(&W0)[4] = appl_common_weights_.W0[index];
    double(&WR)[4] = appl_common_weights_.WR[index];
    double(&WF)[4] = appl_common_weights_.WF[index];
    double(&WB)[4] = appl_common_weights_.WB[index];

    double ttol = 1e-100;
    double obs = appl_common_histokin_.obs_histo;

    // Histogram number
    int nh = appl_common_histokin_.obs_num - 1;

    // translate (index,nh) -> index of the PineAPPL grid
    std::size_t const grid_index = translation_tables.at(nh).at(index);

    double const x1 = appl_common_weights_.x1[k];
    double const x2 = appl_common_weights_.x2[k];
    double const scale2 = appl_common_weights_.muF2[k];
    int const flavmap = appl_common_weights_.flavmap[k];

    if (x1 < 0.0 || x1 > 1.0 || x2 < 0.0 || x2 > 1.0)
    {
        std::cout << "amcblast ERROR: Invalid value of x1 and/or x2 = " << x1 << " " << x2
                  << std::endl;
        std::exit(-10);
    }

    if (x1 == 0.0 && x2 == 0.0)
    {
        return;
    }

    if (flavmap < 1 || flavmap > appl_common_lumi_.nlumi)
    {
        std::cout << "amcblast ERROR: Invalid value of flavmap = " << flavmap
                  << " (number of channels: " << appl_common_lumi_.nlumi << ")" << std::endl;
        std::exit(-10);
    }

    // index of the channel in the grid
    std::size_t const ilumi = flavmap - 1;

    // order of the kinematic variables: { scale, x1, x2 }
    std::array<double, 3> const ntuple = { scale2, x1, x2 };

    // PineAPPL silently drops fills outside of the interpolation ranges:
    // keep track of them so that this can be reported at the end
    GridInfo& info = grid_info.at(nh);
    bool const out_scale = (scale2 < info.q2min) || (scale2 > info.q2max);
    bool const out_x1 = (x1 < info.xmin[0]) || (x1 > info.xmax[0]);
    bool const out_x2 = (x2 < info.xmin[1]) || (x2 > info.xmax[1]);

    double const abs_wgt = std::fabs(W0[k]) + std::fabs(WR[k]) + std::fabs(WF[k]) + std::fabs(WB[k]);

    info.n_fills += 1;
    info.sum_abs_wgt += abs_wgt;

    if (out_scale || out_x1 || out_x2)
    {
        info.n_out_scale += out_scale ? 1 : 0;
        info.n_out_x[0] += out_x1 ? 1 : 0;
        info.n_out_x[1] += out_x2 ? 1 : 0;
        info.sum_abs_wgt_lost += abs_wgt;
    }

    if (std::fabs(W0[k]) > ttol)
    {
        pineappl_grid_fill2(grid_obs[nh], grid_index + 0, obs, ilumi, ntuple.data(), W0[k]);
    }

    if (std::fabs(WR[k]) > ttol)
    {
        pineappl_grid_fill2(grid_obs[nh], grid_index + 1, obs, ilumi, ntuple.data(), WR[k]);
    }

    if (std::fabs(WF[k]) > ttol)
    {
        pineappl_grid_fill2(grid_obs[nh], grid_index + 2, obs, ilumi, ntuple.data(), WF[k]);
    }

    if (std::fabs(WB[k]) > ttol)
    {
        pineappl_grid_fill2(grid_obs[nh], grid_index, obs, ilumi, ntuple.data(), WB[k]);
    }
}

extern "C" void appl_term_()
{
    // Histogram number
    int const nh = appl_common_histokin_.obs_num - 1;

    GridInfo const& info = grid_info.at(nh);

    if (info.sum_abs_wgt_lost > 0.0 && info.sum_abs_wgt > 0.0)
    {
        std::cout << "amcblast WARNING: observable " << nh << ": "
                  << 100.0 * info.sum_abs_wgt_lost / info.sum_abs_wgt
                  << "% of the (absolute) weight lies outside of the interpolation ranges of the "
                     "grid and is NOT stored. Fills outside of the ranges: scale "
                  << info.n_out_scale << ", x1 " << info.n_out_x[0] << ", x2 " << info.n_out_x[1]
                  << " (out of " << info.n_fills << "). Ranges in use: Q2 in [" << info.q2min
                  << ", " << info.q2max << "], x1 in [" << info.xmin[0] << ", " << info.xmax[0]
                  << "], x2 in [" << info.xmin[1] << ", " << info.xmax[1]
                  << "]. Adjust appl_Q2min/appl_Q2max/appl_xmin/appl_xmax (or the appl_*_beam "
                     "variants) in analysis_begin."
                  << std::endl;
    }

    // convert between gs^2 and alphas, and normalize the grid by (hbarc)^2 and the number of runs
    pineappl_grid_scale_by_order(grid_obs[nh], 4.0 * std::acos(-1.0), 1.0, 1.0, 1.0,
        hbarc2_pb * appl_common_histokin_.norm_histo);

    // Write grid to file
    // The optimization is chosen explicitly.
    // - OPTIMIZE_NODES (detection of a static scale) is useless here, since the scale
    //   is dynamic, and it evaluates the inverse of the x-map on the nodes, which can
    //   abort inside PineAPPL.
    // - SYMMETRIZE_CHANNELS is NOT used: it merges the channel (a,b) into (b,a)
    //   assuming that the same PDF is used for both beams. This is not true for
    //   asymmetric collisions (for example p-A, where the two beams have different PDF
    //   sets even if both convolutions have the same PDG id), and the grid would give
    //   wrong results when convolved with a different PDF for each beam.
    pineappl_gof flags = PINEAPPL_GOF_OPTIMIZE_SUBGRID_TYPE | PINEAPPL_GOF_STRIP_EMPTY_ORDERS
        | PINEAPPL_GOF_MERGE_SAME_CHANNELS | PINEAPPL_GOF_STRIP_EMPTY_CHANNELS;
    pineappl_grid_optimize_using(grid_obs[nh], flags);
    pineappl_grid_write(grid_obs[nh], ("grid_obs_" + std::to_string(nh) + "_out.pineappl").c_str());

    pineappl_grid_delete(grid_obs[nh]);
}

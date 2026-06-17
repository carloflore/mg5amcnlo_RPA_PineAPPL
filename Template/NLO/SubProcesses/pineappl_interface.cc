#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
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
extern "C" struct
{
    double Q2min, Q2max;
    double xmin, xmax;
    int nQ2, Q2order;
    int nx, xorder;
} appl_common_grid_;

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

extern "C" void appl_init_()
{
    // Grid Initialization and definition of the observables.
    // Construct the input file name according to its position in the
    // vector "grid_obs".
    std::size_t const index = grid_obs.size();

    uint32_t lo_power = 9999;
    uint32_t nlo_power = 0;

    for (int i = 0; i != appl_common_fixed_.amp_split_size; ++i)
    {
        uint32_t sum = appl_common_fixed_.qcdpower[i] + appl_common_fixed_.qedpower[i];

        lo_power = std::min(lo_power, sum);
        nlo_power = std::max(nlo_power, sum);
    }

    // TODO: are there any situations is which there are NLOs but no LOs?

    // we assume that there is always at least one LO, and zero or one NLO
    assert((nlo_power == (lo_power + 2)) || (nlo_power == lo_power));

    // order params: 5 x uint8_t per order {alphas, alpha, logxir, logxif, logxia}
    std::vector<uint8_t> subgrid_params;

    translation_tables.emplace_back();
    translation_tables.back().reserve(appl_common_fixed_.amp_split_size);

    for (int i = 0; i != appl_common_fixed_.amp_split_size; ++i)
    {
        uint32_t const qcd = appl_common_fixed_.qcdpower[i];
        uint32_t const qed = appl_common_fixed_.qedpower[i];
        uint32_t const sum = qcd + qed;

        translation_tables.back().push_back(subgrid_params.size() / 5);

        if (sum == lo_power)
        {
            // WB
            subgrid_params.insert(subgrid_params.end(), { uint8_t(qcd / 2), uint8_t(qed / 2), 0, 0, 0 });
        }
        else if (sum == nlo_power)
        {
            // W0
            subgrid_params.insert(subgrid_params.end(), { uint8_t(qcd / 2), uint8_t(qed / 2), 0, 0, 0 });
            // WR
            subgrid_params.insert(subgrid_params.end(), { uint8_t(qcd / 2), uint8_t(qed / 2), 1, 0, 0 });
            // WF
            subgrid_params.insert(subgrid_params.end(), { uint8_t(qcd / 2), uint8_t(qed / 2), 0, 1, 0 });
        }
    }

    // Interpolation parameters: use user-specified values or PineAPPL defaults.
    double const q2_min   = appl_common_grid_.Q2min > 0.0 ? appl_common_grid_.Q2min : 1e2;
    double const q2_max   = appl_common_grid_.Q2max > 0.0 ? appl_common_grid_.Q2max : 1e8;
    std::size_t const q2_nodes = appl_common_grid_.nQ2     > 0 ? std::size_t(appl_common_grid_.nQ2)     : 40;
    std::size_t const q2_order = appl_common_grid_.Q2order > 0 ? std::size_t(appl_common_grid_.Q2order) : 3;
    double const x_min    = appl_common_grid_.xmin > 0.0 ? appl_common_grid_.xmin : 2e-7;
    double const x_max    = appl_common_grid_.xmax > 0.0 ? appl_common_grid_.xmax : 1.0;
    std::size_t const x_nodes  = appl_common_grid_.nx     > 0 ? std::size_t(appl_common_grid_.nx)     : 50;
    std::size_t const x_order  = appl_common_grid_.xorder > 0 ? std::size_t(appl_common_grid_.xorder) : 3;

    // Convolutions: unpolarized PDFs for each initial-state hadron
    pineappl_conv convs[2] = {
        { PINEAPPL_CONV_TYPE_UNPOL_PDF, heprup_.idbmup[0] },
        { PINEAPPL_CONV_TYPE_UNPOL_PDF, heprup_.idbmup[1] },
    };

    // Kinematic variables: one factorization scale + two momentum fractions
    pineappl_kinematics kinematics[3] = {
        { PINEAPPL_KINEMATICS_SCALE, {0} },
        { PINEAPPL_KINEMATICS_X,     {0} },
        { PINEAPPL_KINEMATICS_X,     {1} },
    };

    // Interpolation specs for each kinematic variable
    pineappl_interp interpolations[3] = {
        { q2_min, q2_max, q2_nodes, q2_order,
          PINEAPPL_REWEIGHT_METH_NO_REWEIGHT, PINEAPPL_MAP_APPL_GRID_H0, PINEAPPL_INTERP_METH_LAGRANGE },
        { x_min, x_max, x_nodes, x_order,
          PINEAPPL_REWEIGHT_METH_APPL_GRID_X, PINEAPPL_MAP_APPL_GRID_F2, PINEAPPL_INTERP_METH_LAGRANGE },
        { x_min, x_max, x_nodes, x_order,
          PINEAPPL_REWEIGHT_METH_APPL_GRID_X, PINEAPPL_MAP_APPL_GRID_F2, PINEAPPL_INTERP_METH_LAGRANGE },
    };

    // Scale functional forms: ren and fac use Scale(0); no fragmentation scale
    pineappl_scale_func_form mu_scales[3] = {
        { PINEAPPL_SCALE_FUNC_FORM_SCALE,    {0} },
        { PINEAPPL_SCALE_FUNC_FORM_SCALE,    {0} },
        { PINEAPPL_SCALE_FUNC_FORM_NO_SCALE, {0} },
    };

    // Set up the PDF channels
    auto* channels = pineappl_channels_new(2);

    // Loop over parton luminosities
    for (int ilumi = 0; ilumi < appl_common_lumi_.nlumi; ilumi++)
    {
        int nproc = appl_common_lumi_.nproc[ilumi];

        std::vector<int32_t> pdg_ids;
        pdg_ids.reserve(2 * nproc);

        for (int iproc = 0; iproc != nproc; ++iproc)
        {
            int32_t a = appl_common_lumi_.lumimap[ilumi][iproc][0];
            int32_t b = appl_common_lumi_.lumimap[ilumi][iproc][1];

            pdg_ids.push_back(a == 0 ? 21 : a);
            pdg_ids.push_back(b == 0 ? 21 : b);
        }

        pineappl_channels_add(channels, nproc, pdg_ids.data(), nullptr);
    }

    // Create a grid with the binning given in the "obsbins[Nbins+1]" array
    grid_obs.push_back(pineappl_grid_new2(
        appl_common_histokin_.obs_nbins,
        appl_common_histokin_.obs_bins,
        subgrid_params.size() / 5,
        subgrid_params.data(),
        channels,
        PINEAPPL_PID_BASIS_PDG,
        convs,
        3,
        interpolations,
        kinematics,
        mu_scales
    ));

    pineappl_channels_delete(channels);
}

extern "C" void appl_delete_itype_()
{
    // Check that itype ranges from 1 to 5.
    int itype = appl_common_histokin_.itype_histo;
    if ((itype < 1) || (itype > 5))
    {
        std::cout << "amcblast ERROR: Invalid value of itype = " << itype << std::endl;
        std::exit(-10);
    }

    // this is the second index of the WB/R/F/0 arrays
    int index = appl_common_histokin_.amp_pos - 1;

    // aMC@NLO weights. Four grids, ordered as {W0,WR,WF,WB}.
    double(&W0)[4] = appl_common_weights_.W0[index];
    double(&WR)[4] = appl_common_weights_.WR[index];
    double(&WF)[4] = appl_common_weights_.WF[index];
    double(&WB)[4] = appl_common_weights_.WB[index];

    int k;

    switch (itype)
    {
    case 1:
        k = 0;
        break;

    case 2:
    case 3:
        k = 1;
        break;

    case 4:
        k = 2;
        break;

    case 5:
        k = 3;
        break;

    default:
        assert( false );
    }

    W0[k] = 0.0;
    WR[k] = 0.0;
    WF[k] = 0.0;
    WB[k] = 0.0;
}

extern "C" void appl_fill_()
{
    // Check that itype ranges from 1 to 5.
    int itype = appl_common_histokin_.itype_histo;
    if ((itype < 1) || (itype > 5))
    {
        std::cout << "amcblast ERROR: Invalid value of itype = " << itype << std::endl;
        std::exit(-10);
    }

    // this is the second index of the WB/R/F/0 arrays
    int index = appl_common_histokin_.amp_pos - 1;

    // aMC@NLO weights. Four grids, ordered as {W0,WR,WF,WB}.
    double(&W0)[4] = appl_common_weights_.W0[index];
    double(&WR)[4] = appl_common_weights_.WR[index];
    double(&WF)[4] = appl_common_weights_.WF[index];
    double(&WB)[4] = appl_common_weights_.WB[index];

    int nlumi = appl_common_lumi_.nlumi;
    double ttol = 1e-100;
    double obs = appl_common_histokin_.obs_histo;

    // Histogram number
    int nh = appl_common_histokin_.obs_num - 1;

    // translate (index,nh) -> index of the PineAPPL grid
    int const grid_index = translation_tables.at(nh).at(index);

    int k;

    switch (itype)
    {
    case 1:
        k = 0;
        break;

    case 2:
    case 3:
        k = 1;
        break;

    case 4:
        k = 2;
        break;

    case 5:
        k = 3;
        break;

    default:
        assert( false );
    }

    double const x1 = appl_common_weights_.x1[k];
    double const x2 = appl_common_weights_.x2[k];
    double const scale2 = appl_common_weights_.muF2[k];
    int const ilumi = appl_common_weights_.flavmap[k] - 1;

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

    double const ntuple[3] = { scale2, x1, x2 };

    if (std::fabs(W0[k]) > ttol)
    {
        pineappl_grid_fill2(grid_obs[nh], grid_index + 0, obs, ilumi, ntuple, W0[k]);
    }

    if (std::fabs(WR[k]) > ttol)
    {
        pineappl_grid_fill2(grid_obs[nh], grid_index + 1, obs, ilumi, ntuple, WR[k]);
    }

    if (std::fabs(WF[k]) > ttol)
    {
        pineappl_grid_fill2(grid_obs[nh], grid_index + 2, obs, ilumi, ntuple, WF[k]);
    }

    if (std::fabs(WB[k]) > ttol)
    {
        pineappl_grid_fill2(grid_obs[nh], grid_index, obs, ilumi, ntuple, WB[k]);
    }
}

extern "C" void appl_term_()
{
    // Histogram number
    int const nh = appl_common_histokin_.obs_num - 1;

    // convert between gs^2 and alphas, and normalize the grid by (hbarc)^2 and the number of runs
    pineappl_grid_scale_by_order(grid_obs[nh], 4.0 * std::acos(-1.0), 1.0, 1.0, 1.0,
        389379660.0 * appl_common_histokin_.norm_histo);

    // Write grid to file
    pineappl_grid_optimize(grid_obs[nh]);
    pineappl_grid_write(grid_obs[nh], ("grid_obs_" + std::to_string(nh) + "_out.pineappl").c_str());

    pineappl_grid_delete(grid_obs[nh]);
}

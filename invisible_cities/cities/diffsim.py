"""
-----------------------------------------------------------------------
                            Diffsim
-----------------------------------------------------------------------
From Diffusion Simulation. This city reads energy deposits (hits) and simulates
the electron ionization, drift and diffusion. The input of this city is nexus 
output containing hits.
This city outputs:
    - Diff_electrons : Diffused electrons 
    - MC  info
    - Run info
    - Filters
"""

import os
import warnings
import numpy  as np
import tables as tb
import pandas as pd

from .components import city
from .components import print_every
from .components import copy_mc_info
from .components import collect
from .components import MC_hits_and_part_from_files
from .components import check_max_time

from ..core.configure import check_annotations
from ..core.configure import EventRangeType
from ..core.configure import OneOrManyFiles

from .. core     import tbl_functions as tbl
from .. dataflow import dataflow      as fl

from ..io.event_filter_io  import event_filter_writer
from ..io.dst_io           import df_writer

from ..detsim.simulate_electrons import generate_ionization_electrons
from ..detsim.simulate_electrons import drift_electrons
from ..detsim.simulate_electrons import diffuse_electrons
from ..detsim.simulate_electrons import distribute_hits_energy_among_electrons

from ..detsim.label_functions import select_main_track
from ..detsim.label_functions import label_track_and_other
from ..detsim.label_functions import label_blob_hits
from ..detsim.label_functions import small_blob_fix
from ..detsim.label_functions import get_extremes_label

@check_annotations
def filter_hits_after_max_time(max_time : float):
    """
    Function that filters and warns about delayed hits
    (hits at times larger that max_time configuration parameter)
    """
    def select_hits(x, y, z, energy, time, label, hit_id, hit_part_id, event_number):
        sel = ((time - min(time)) < max_time)
        if sel.all(): return x, y, z, energy, time, label, hit_id, hit_part_id
        else:
            warnings.warn(f"Delayed hits at event {event_number}")
            return x[sel], y[sel], z[sel], energy[sel], time[sel], label[sel], hit_id[sel], hit_part_id[sel]
    return select_hits


@check_annotations
def hits_selector(active_only: bool=True):
    """
    Filtering function that selects hits. (see :active_only: description)

    Parameters:
        :active_only: bool
            if True, returns hits in ACTIVE
            if False, returns hits in ACTIVE and BUFFER
    Returns:
        :select_hits: Callable
            function that select the hits depending on :active_only: parameter
    """
    def select_hits(x, y, z, energy, time, label, hit_id, hit_part_id, name, name_id):

        if label.dtype == np.int32:
            active = name_id[name == "ACTIVE"][0]
            buff   = name_id[name == "BUFFER"][0]
        else:
            active = 'ACTIVE'
            buff   = 'BUFFER'

        sel = (label == active)
        if not active_only:
            sel =  sel | (label == buff)
        return x[sel], y[sel], z[sel], energy[sel], time[sel], label[sel], hit_id[sel], hit_part_id[sel]
    return select_hits

@check_annotations
def binclass_creator(sig_creator : str):
    """
    Adds binary label to the events (0 - background / 1 - signal)
    The variable 'sig_creator' defines what we consider signal. 
    For double beta events, we use 'none', and for pair creation 'conv'.
    This is explained by how the nexus simulation stores information.
    The rest events are labelled as background.
    """
    def add_binclass(particle_id, particle_name, creator_proc, hit_part_id):
        # Pick the particles that created any MC hit
        part_df = pd.DataFrame({'particle_id' : particle_id, 'particle_name' : particle_name, 'creator_proc' : creator_proc})
        hit_part_df = pd.DataFrame({'particle_id' : hit_part_id}).drop_duplicates()
        part_df = hit_part_df.merge(part_df, how = 'left')
        # From all those, pick the e+/e-
        part_df = part_df[np.isin(part_df.particle_name, ['e+', 'e-'])]
        # Consider signal when there are 2 particles with the same signal creator process
        # Consider background otherwise
        return 1 if int(sum(part_df.creator_proc == sig_creator)) == 2 else 0

    return add_binclass

def hits_particle_df_creator():
    '''
    Creates the DataFrame with all the particles and hits info necessary for the labelling.
    '''
    def create_hits_particle_df(hit_part_id, hit_id, hit_energy, particle_id, particle_name, creator_proc):
        hits = pd.DataFrame({'particle_id' : hit_part_id, 'hit_id': hit_id, 'energy': hit_energy})
        part = pd.DataFrame({'particle_id' : particle_id, 'particle_name' : particle_name, 'creator_proc' : creator_proc})

        hits_part = pd.merge(hits, part, on = 'particle_id')
        return hits_part
    return create_hits_particle_df

@check_annotations
def segclass_creator(sig_creator : str, segclass_dct : dict, delta_ener_loss : float):
    '''
    Adds segmentation label to the hits of each event followign 'segclass_dct', that should
    contain the 3 classes: 'ohter', 'track' and 'blob'.

    Uses info from binary class and the merged hits and particle information for the event.
    Then, functions from label_functions are applied, and we get in return an array of 
    segmentation label.
    '''
    def add_segclass(hits_part, binclass):
        # Compute energy for each particle
        part = hits_part.groupby(['particle_id', 'particle_name', 'creator_proc']).agg({'energy':'sum'}).reset_index().rename(columns = {'energy':'track_ener'})

        # Select particles from main track and assign them their label
        main_track = select_main_track    (part, binclass, sig_creator)
        # From this we can get the extremes label
        ext = get_extremes_label(hits_part, main_track, binclass)
        label_part = label_track_and_other(part, main_track, segclass_dct)

        # Merge with hits info the segmentation label
        label_hits = hits_part.merge(label_part[['particle_id', 'track_ener', 'segclass']], how='outer', on='particle_id')

        label_hits = label_blob_hits(label_hits, delta_ener_loss, segclass_dct)
        label_hits = small_blob_fix (label_hits, main_track, segclass_dct)

        # Sort back again the hits with the same order as before
        label_hits = label_hits.sort_index()
        
        return label_hits.segclass.values, ext
    return add_segclass

def extlabel_creator(segclass_dct):
    '''
    Adds extreme label to voxels, and forces some of them to be blob class
    '''
    def add_extlabel(x, y, z, ext, bins, xbin, ybin, zbin, segbin, binclass):
        coords = ['x', 'y', 'z']
        ext_df = pd.DataFrame({'x':x, 'y':y, 'z':z, 'ext':ext})
        ext_df = ext_df[ext_df.ext != 0]
        for i in range(3): ext_df[coords[i] + 'bin'] = pd.cut(ext_df[coords[i]], bins[i], labels = np.arange(0, len(bins[i])-1)).astype('int')
        ext_df = ext_df.drop(coords, axis = 1).rename(columns={'xbin':'x', 'ybin':'y', 'zbin':'z'})
        ext_df = ext_df.groupby(coords).agg({'ext':'sum'}).reset_index() # if both extremes are in the same voxel, give them the sum of the labels

        vox_df = pd.DataFrame({'x':xbin, 'y':ybin, 'z':zbin, 'segclass':segbin})
        vox_df = vox_df.merge(ext_df, how = 'outer').fillna(0)
        vox_df['ext'] = vox_df['ext'].astype(int)
        # Make sure that certain extremes have blob label
        # add label 3 (both extremes in the same voxel), its always a blob voxel
        if binclass == 0:
            vox_df.loc[vox_df['ext'].isin([1, 3]), 'segclass'] = segclass_dct['blob']
        if binclass == 1:
            vox_df.loc[vox_df['ext'].isin([1, 2, 3]), 'segclass'] = segclass_dct['blob']
        return vox_df.segclass.values, vox_df.ext.values
    
    return add_extlabel

def decolabel_creator():
    '''
    Compares voxelized MC and diffused tracks to assign to the diffused voxels a label for deconvolution with NNs
    '''
    def add_decolabel(xdiff, ydiff, zdiff, xmc, ymc, zmc):
        diff = pd.DataFrame({'x':xdiff, 'y':ydiff, 'z':zdiff})
        norm = pd.DataFrame({'x':xmc, 'y':ymc, 'z':zmc})

        decolabel = (diff.merge(norm, how = 'left', indicator=True)._merge == 'both').astype('int').values
        return decolabel
    return add_decolabel

@check_annotations
def ielectron_simulator_diffsim(*, wi: float, fano_factor: float, lifetime: float,
                                transverse_diffusion: float, longitudinal_diffusion: float, drift_velocity:float,
                                el_gain: float, conde_policarpo_factor: float):
    """
    Function that simulates electron creation, drift, diffusion and photon generation at the EL

    Parameters: floats
        parameter names are self-descriptive.
    Returns:
        :simulate_ielectrons:
            function that returns the positions emission times and number of photons at the EL
    """
    def simulate_ielectrons(x, y, z, time, energy, label):
        nelectrons  = generate_ionization_electrons(energy, wi, fano_factor)
        nelectrons  = drift_electrons(z, nelectrons, lifetime, drift_velocity)
        dx, dy, dz  = diffuse_electrons(x, y, z, nelectrons, transverse_diffusion, longitudinal_diffusion)
        nelec_ener  = distribute_hits_energy_among_electrons(nelectrons, energy)
        nelec_label = np.repeat(label, nelectrons)
        dtimes = dz/drift_velocity + np.repeat(time, nelectrons)
        nphotons = np.random.normal(el_gain, np.sqrt(el_gain * conde_policarpo_factor), size=nelectrons.sum())
        nphotons = np.round(nphotons).astype(np.int32)
        return dx, dy, dz, nelec_ener, dtimes, nphotons, nelec_label
    return simulate_ielectrons

@check_annotations
def event_fiducial_selector(*, xlim: tuple, ylim: tuple, zlim: tuple):
    """
    Selects events based in the voxelization limits.
    """
    def select_fiducial_events(x, y, z):
        hits_out = np.any(x < xlim[0]) | np.any(x > xlim[1]) | \
                   np.any(y < ylim[0]) | np.any(y > ylim[1]) | \
                   np.any(z < zlim[0]) | np.any(z > zlim[1])
        return ~hits_out
    return select_fiducial_events

@check_annotations
def bins_creator(*, xlim : tuple, nbins_x : int,
                  ylim : tuple, nbins_y : int,
                  zlim : tuple, nbins_z : int):
    
    def create_bins():
        bins_x = np.linspace(xlim[0], xlim[1], nbins_x)
        bins_y = np.linspace(ylim[0], ylim[1], nbins_y)
        bins_z = np.linspace(zlim[0], zlim[1], nbins_z)
        return (bins_x, bins_y, bins_z)
    return create_bins

@check_annotations
def voxel_creator():
    """
    Aggregates all the diffussion electrons with (x, y, z, E, nphotons) in 
    voxels of certain size (total size / (nbins - 1)), adding the energy 
    and the number of photons (not used for pure MC).
    Also, it returns the x, y and z mean positions (once again, not used for MC
    voxelization, but used for the diffusion).
    # The class that deposited more energy within a voxel is assigned to
    # this voxel, but we add a weight for each class to give more importance
    # to certain classes using a dict with {class_number:weight}
    """
    def create_voxels(x, y, z, bins, energy, nphotons, segclass): 

        xbin = pd.cut(x, bins[0], labels = np.arange(0, len(bins[0])-1)).astype('int')
        ybin = pd.cut(y, bins[1], labels = np.arange(0, len(bins[1])-1)).astype('int')
        zbin = pd.cut(z, bins[2], labels = np.arange(0, len(bins[2])-1)).astype('int')

        df = pd.DataFrame({'xbin' : xbin, 'ybin' : ybin, 'zbin' : zbin, 'x' : x, 'y' : y, 'z' : z, 'energy' : energy, 'nphotons' : nphotons, 'segclass' : segclass}) 

        # Taking off the weighted energy thing because I'm adding the extremes label
        # if class_weights is not None:
        #     df['wgt_ene'] = df.apply(lambda row: row['energy'] * class_weights.get(row['segclass'], 1), axis=1)
        # else:
        #     df['wgt_ene'] = df['energy']
        
        def weighted_mean(column, weights):
            return (column * weights).sum() / weights.sum()

        mean_pos = lambda coord: weighted_mean(coord, df.loc[coord.index, 'energy'])

        # Get bins
        total_df = df.groupby(['xbin', 'ybin', 'zbin']).agg({'x':mean_pos, 'y':mean_pos, 'z':mean_pos, 'energy': 'sum', 'nphotons': 'sum'}).reset_index()
        class_df = df.groupby(['xbin', 'ybin', 'zbin', 'segclass']).agg({'energy': 'sum', 'energy':'sum', 'nphotons': 'sum'}).reset_index()

        max_ener_idx = class_df.groupby(['xbin', 'ybin', 'zbin'])['energy'].idxmax()
        class_winner_df = class_df.loc[max_ener_idx, ['xbin', 'ybin', 'zbin', 'segclass']]
        out = pd.merge(total_df, class_winner_df, on=['xbin', 'ybin', 'zbin'])

        xbin, ybin, zbin = out.xbin.values, out.ybin.values, out.zbin.values
        xmean, ymean, zmean = out.x.values, out.y.values, out.z.values
        ebin, phbin, segbin = out.energy.values, out.nphotons.values, out.segclass.values

        return xbin, ybin, zbin, xmean, ymean, zmean, ebin, phbin, segbin 
    
    return create_voxels

# Probably these 2 functions won't go here, or there's another function from IC that can do this easily, but I ignore it
def diff_df_creator():
    def create_diff_df(evt, x, y, z, xmean, ymean, zmean, energy, nphotons, binclass, segclass, decolabel, extlabel):
        return pd.DataFrame({'event'     : evt, 
                             'xbin'      : x, 
                             'ybin'      : y, 
                             'zbin'      : z, 
                             'x_mean'    : xmean,
                             'y_mean'    : ymean,
                             'z_mean'    : zmean,
                             'ebin'      : energy, 
                             'nphbin'    : nphotons, 
                             'binclass'  : binclass,
                             'segclass'  : segclass, 
                             'decolabel' : decolabel,
                             'extlabel'  : extlabel})
    return create_diff_df

def diff_writer(h5out):
    """
    For a given open table returns a writer for diffusion electrons dataframe
    """
    def write_diff(df):
        return df_writer(h5out              = h5out                         ,
                         df                 = df                            ,
                         group_name         = 'Diffsim'                     ,
                         table_name         = 'vox_diff'                    ,
                         descriptive_string = 'Voxelized diffused electrons',
                         columns_to_index   = ['event']                     )
    return write_diff

@city
def diffsim( *
            , files_in       : OneOrManyFiles
            , file_out       : str
            , event_range    : EventRangeType
            , print_mod      : int
            , compression    : str
            , detector_db    : str
            , run_number     : int
            , buffer_params  : dict
            , physics_params : dict
            , voxel_params   : dict
            , label_params   : dict
            , rate           : float
            ):

    buffer_params_  = buffer_params .copy()
    physics_params_ = physics_params.copy()

    buffer_params_["max_time"] = check_max_time(buffer_params_["max_time"], buffer_params_["length"])

    filter_delayed_hits = fl.map(filter_hits_after_max_time(buffer_params_["max_time"]),
                                 args = ('x', 'y', 'z', 'energy', 'time', 'label', 'hit_id', 'hit_part_id', 'event_number'),
                                 out  = ('x', 'y', 'z', 'energy', 'time', 'label', 'hit_id', 'hit_part_id'))

    select_s1_candidate_hits = fl.map(hits_selector(False),
                                item = ('x', 'y', 'z', 'energy', 'time', 'label', 'hit_id', 'hit_part_id', 'name', 'name_id'))

    select_active_hits = fl.map(hits_selector(True),
                                args = ('x', 'y', 'z', 'energy', 'time', 'label', 'hit_id', 'hit_part_id', 'name', 'name_id'),
                                out = ('x_a', 'y_a', 'z_a', 'energy_a', 'time_a', 'labels_a', 'hit_id_a', 'hit_part_id_a'))

    filter_events_no_active_hits = fl.map(lambda x:np.any(x),
                                          args= 'energy_a',
                                          out = 'passed_active')
    events_passed_active_hits = fl.count_filter(bool, args='passed_active')

    filter_events_out_fiducial_mc = fl.map(event_fiducial_selector(**{key: voxel_params[key] for key in ['xlim', 'ylim', 'zlim']}), 
                                        args = ('x_a', 'y_a', 'z_a'), 
                                        out = 'passed_fiducial_mc')
    fiducial_events_mc = fl.count_filter(bool, args='passed_fiducial_mc')

    assign_binclass = fl.map(binclass_creator(label_params['sig_creator']), 
                             args = ('particle_id', 'particle_name', 'creator_proc', 'hit_part_id'), 
                             out  = 'binclass')

    creates_hits_part_df = fl.map(hits_particle_df_creator(), 
                                  args = ('hit_part_id_a','hit_id_a', 'energy_a', 'particle_id', 'particle_name', 'creator_proc'), 
                                  out  = 'hits_part_df')
    
    assign_segclass = fl.map(segclass_creator(**label_params), 
                             args = ('hits_part_df', 'binclass'), 
                             out  = ('segclass_a', 'ext_a'))
    create_bins = fl.map(bins_creator(**voxel_params), 
                        args = (), 
                        out = 'bins')
    
    voxelize_mc = fl.map(voxel_creator(), 
                             args = ('x_a', 'y_a', 'z_a', 'bins', 'energy_a', 'time_a', 'segclass_a'), # using time here to fill the function with something
                             out = ('xbin_mc', 'ybin_mc', 'zbin_mc', '_', '_', '_', 'ebin_mc', '_', 'segbin_mc'))

    simulate_electrons = fl.map(ielectron_simulator_diffsim(**physics_params_),
                                args = ('x_a', 'y_a', 'z_a', 'time_a', 'energy_a', 'segclass_a'),
                                out  = ('x_ph', 'y_ph', 'z_ph', 'energy_ph', 'times_ph', 'nphotons', 'segclass_ph'))
    
    filter_events_out_fiducial = fl.map(event_fiducial_selector(**{key: voxel_params[key] for key in ['xlim', 'ylim', 'zlim']}), 
                                        args = ('x_ph', 'y_ph', 'z_ph'), 
                                        out = 'passed_fiducial')
    
    fiducial_events = fl.count_filter(bool, args='passed_fiducial')

    voxelize_events = fl.map(voxel_creator(), 
                             args = ('x_ph', 'y_ph', 'z_ph', 'bins', 'energy_ph', 'nphotons', 'segclass_ph'), 
                             out = ('x_bin', 'y_bin', 'z_bin', 'x_mean', 'y_mean', 'z_mean', 'e_bin', 'nph_bin', 'seg_bin'))
    
    create_extlabel = fl.map(extlabel_creator(label_params['segclass_dct']), 
                             args = ('x_a', 'y_a', 'z_a', 'ext_a', 'bins', 'x_bin', 'y_bin', 'z_bin', 'seg_bin', 'binclass'),
                             out = ('new_seg_bin', 'ext_bin'))
    
    create_decolabel = fl.map(decolabel_creator(), 
                              args = ('x_bin', 'y_bin', 'z_bin', 'xbin_mc', 'ybin_mc', 'zbin_mc'), 
                              out = ('deco_bin'))
    
    creates_diff_df = fl.map(diff_df_creator(), 
                             args = ('event_number', 'x_bin', 'y_bin', 'z_bin', 'x_mean', 'y_mean', 'z_mean', 'e_bin', 'nph_bin', 'binclass', 'new_seg_bin', 'deco_bin', 'ext_bin'), 
                             out = ('vox_diff_df'))

    count_photons = fl.map(lambda x: np.sum(x) > 0,
                           args= 'nphotons',
                           out = 'enough_photons')
    
    dark_events   = fl.count_filter(bool, args='enough_photons')

    event_count_in = fl.spy_count()
    evtnum_collect = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        write_nohits_filter      = fl.sink(event_filter_writer(h5out, "active_hits"), args=("event_number", "passed_active")     )
        write_fiducial_filter_mc = fl.sink(event_filter_writer(h5out,   "fid_ev_mc"), args=("event_number", "passed_fiducial_mc"))
        write_dark_evt_filter    = fl.sink(event_filter_writer(h5out, "dark_events"), args=("event_number", "enough_photons")    )
        write_fiducial_filter    = fl.sink(event_filter_writer(h5out,  "fid_events"), args=("event_number", "passed_fiducial")   )
        write_diff               = fl.sink(diff_writer        (h5out = h5out       ), args=("vox_diff_df")                       )

        result = fl.push(source= MC_hits_and_part_from_files(files_in, rate),
                         pipe  = fl.pipe( fl.slice(*event_range, close_all=True)
                                        , event_count_in.spy
                                        , print_every(print_mod)
                                        , filter_delayed_hits
                                        , select_s1_candidate_hits
                                        , select_active_hits
                                        , filter_events_no_active_hits
                                        , fl.branch(write_nohits_filter)
                                        , events_passed_active_hits.filter
                                        , filter_events_out_fiducial_mc
                                        , fl.branch(write_fiducial_filter_mc)
                                        , fiducial_events_mc.filter
                                        , assign_binclass
                                        , creates_hits_part_df
                                        , assign_segclass
                                        , create_bins
                                        , voxelize_mc
                                        , simulate_electrons
                                        , filter_events_out_fiducial
                                        , fl.branch(write_fiducial_filter)
                                        , fiducial_events.filter
                                        , voxelize_events
                                        , create_extlabel
                                        , create_decolabel
                                        , count_photons
                                        , fl.branch(write_dark_evt_filter)
                                        , dark_events.filter
                                        , creates_diff_df
                                        , fl.branch(write_diff)
                                        , "event_number"
                                        , evtnum_collect.sink),
                         result = dict(events_in     = event_count_in.future,
                                       evtnum_list   = evtnum_collect.future,
                                       dark_events   = dark_events   .future))

        copy_mc_info(files_in, h5out, result.evtnum_list,
                     detector_db, run_number)

        return result

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

from ..reco     import tbl_functions as tbl
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
    add_binclass = lambda creator_proc: 1 if int(sum(creator_proc == sig_creator)) == 2 else 0
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
        label_part = label_track_and_other(part, main_track, segclass_dct)

        # Merge with hits info the segmentation label
        label_hits = hits_part.merge(label_part[['particle_id', 'track_ener', 'segclass']], how='outer', on='particle_id')

        label_hits = label_blob_hits(label_hits, delta_ener_loss, segclass_dct)
        label_hits = small_blob_fix (label_hits, main_track, segclass_dct)

        # Sort back again the hits with the same order as before
        label_hits = label_hits.sort_index()
        return label_hits.segclass.values
    return add_segclass

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
def voxel_creator(*, xlim : tuple, nbins_x : int,
                     ylim : tuple, nbins_y : int,
                     zlim : tuple, nbins_z : int):
    """
    Aggregates all the diffussion electrons with (x, y, z, E, nphotons) in 
    voxels of certain size (total size / (nbins - 1)), adding the energy 
    and the number of photons.
    Also, the class that deposited more energy within a voxel is assigned to
    this voxel.
    """
    def create_voxels(x, y, z, energy, nphotons, segclass):
        bins_x = np.linspace(xlim[0], xlim[1], nbins_x)
        bins_y = np.linspace(ylim[0], ylim[1], nbins_y)
        bins_z = np.linspace(zlim[0], zlim[1], nbins_z)

        xbin = pd.cut(x, bins_x, labels = np.arange(0, len(bins_x)-1)).astype('int')
        ybin = pd.cut(y, bins_y, labels = np.arange(0, len(bins_y)-1)).astype('int')
        zbin = pd.cut(z, bins_z, labels = np.arange(0, len(bins_z)-1)).astype('int')

        df = pd.DataFrame({'xbin' : xbin, 'ybin' : ybin, 'zbin' : zbin,'energy' : energy, 'nphotons' : nphotons, 'segclass':segclass})
        class_df = df.groupby(['xbin', 'ybin', 'zbin', 'segclass']).agg({'energy': 'sum', 'nphotons': 'sum'}).reset_index()
        total_df = class_df.groupby(['xbin', 'ybin', 'zbin']).agg({'energy': 'sum', 'nphotons': 'sum'}).reset_index()

        max_ener_idx = class_df.groupby(['xbin', 'ybin', 'zbin'])['energy'].idxmax()
        class_winner_df = class_df.loc[max_ener_idx, ['xbin', 'ybin', 'zbin', 'segclass']]
        out = pd.merge(total_df, class_winner_df, on=['xbin', 'ybin', 'zbin'])

        xbin, ybin, zbin, ebin, phbin, segbin = out.xbin.values, out.ybin.values, out.zbin.values, out.energy.values, out.nphotons.values, out.segclass.values

        return xbin, ybin, zbin, ebin, phbin, segbin
    
    return create_voxels

# Probably these 2 functions won't go here, or there's another function from IC that can do this easily, but I ignore it
def diff_df_creator():
    def create_diff_df(evt, x, y, z, energy, nphotons, binclass, segclass):
        return pd.DataFrame({'event'    : evt, 
                             'xbin'     : x, 
                             'ybin'     : y, 
                             'zbin'     : z, 
                             'ebin'     : energy, 
                             'nphbin'   : nphotons, 
                             'binclass' : binclass,
                             'segclass' : segclass})
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

    assign_binclass = fl.map(binclass_creator(label_params['sig_creator']), 
                             args = 'creator_proc', 
                             out  = 'binclass')

    creates_hits_part_df = fl.map(hits_particle_df_creator(), 
                                  args = ('hit_part_id_a','hit_id_a', 'energy_a', 'particle_id', 'particle_name', 'creator_proc'), 
                                  out  = 'hits_part_df')
    
    assign_segclass = fl.map(segclass_creator(**label_params), 
                             args = ('hits_part_df', 'binclass'), 
                             out  = 'segclass_a')

    simulate_electrons = fl.map(ielectron_simulator_diffsim(**physics_params_),
                                args = ('x_a', 'y_a', 'z_a', 'time_a', 'energy_a', 'segclass_a'),
                                out  = ('x_ph', 'y_ph', 'z_ph', 'energy_ph', 'times_ph', 'nphotons', 'segclass_ph'))
    
    filter_events_out_fiducial = fl.map(event_fiducial_selector(**{key: voxel_params[key] for key in ['xlim', 'ylim', 'zlim']}), 
                                        args = ('x_ph', 'y_ph', 'z_ph'), 
                                        out = 'passed_fiducial')
    
    fiducial_events = fl.count_filter(bool, args='passed_fiducial')

    voxelize_events = fl.map(voxel_creator(**voxel_params), 
                             args = ('x_ph', 'y_ph', 'z_ph', 'energy_ph', 'nphotons', 'segclass_ph'), 
                             out = ('x_bin', 'y_bin', 'z_bin', 'e_bin', 'nph_bin', 'seg_bin'))
    
    creates_diff_df = fl.map(diff_df_creator(), 
                             args = ('event_number', 'x_bin', 'y_bin', 'z_bin', 'e_bin', 'nph_bin', 'binclass', 'seg_bin'), 
                             out = ('vox_diff_df'))

    count_photons = fl.map(lambda x: np.sum(x) > 0,
                           args= 'nphotons',
                           out = 'enough_photons')
    
    dark_events   = fl.count_filter(bool, args='enough_photons')

    event_count_in = fl.spy_count()
    evtnum_collect = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        write_nohits_filter   = fl.sink(event_filter_writer(h5out, "active_hits"), args=("event_number", "passed_active")  )
        write_dark_evt_filter = fl.sink(event_filter_writer(h5out, "dark_events"), args=("event_number", "enough_photons") )
        write_fiducial_filter = fl.sink(event_filter_writer(h5out,  "fid_events"), args=("event_number", "passed_fiducial"))
        write_diff            = fl.sink(diff_writer        (h5out = h5out       ), args=("vox_diff_df")                    )

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
                                        , assign_binclass
                                        , creates_hits_part_df
                                        , assign_segclass
                                        , simulate_electrons
                                        , filter_events_out_fiducial
                                        , fl.branch(write_fiducial_filter)
                                        , fiducial_events.filter
                                        , voxelize_events
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

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
from .components import MC_hits_from_files
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

@check_annotations
def filter_hits_after_max_time(max_time : float):
    """
    Function that filters and warns about delayed hits
    (hits at times larger that max_time configuration parameter)
    """
    def select_hits(x, y, z, energy, time, label, event_number):
        sel = ((time - min(time)) < max_time)
        if sel.all(): return x, y, z, energy, time, label
        else:
            warnings.warn(f"Delayed hits at event {event_number}")
            return x[sel], y[sel], z[sel], energy[sel], time[sel], label[sel]
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
    def select_hits(x, y, z, energy, time, label, name, name_id):

        if label.dtype == np.int32:
            active = name_id[name == "ACTIVE"][0]
            buff   = name_id[name == "BUFFER"][0]
        else:
            active = 'ACTIVE'
            buff   = 'BUFFER'

        sel = (label == active)
        if not active_only:
            sel =  sel | (label == buff)

        return x[sel], y[sel], z[sel], energy[sel], time[sel], label[sel]
    return select_hits


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
    def simulate_ielectrons(x, y, z, time, energy):
        nelectrons = generate_ionization_electrons(energy, wi, fano_factor)
        nelectrons = drift_electrons(z, nelectrons, lifetime, drift_velocity)
        dx, dy, dz = diffuse_electrons(x, y, z, nelectrons, transverse_diffusion, longitudinal_diffusion)
        nelec_ener = distribute_hits_energy_among_electrons(nelectrons, energy)
        dtimes = dz/drift_velocity + np.repeat(time, nelectrons)
        nphotons = np.random.normal(el_gain, np.sqrt(el_gain * conde_policarpo_factor), size=nelectrons.sum())
        nphotons = np.round(nphotons).astype(np.int32)
        return dx, dy, dz, nelec_ener, dtimes, nphotons
    return simulate_ielectrons

# Probably these 2 functions won't go here, or there's another function from IC that can do this easily, but I ignore it
def diff_df_creator():
    def create_diff_df(evt, x, y, z, energy, time, nphotons):
        return pd.DataFrame({'event'    : evt      , 
                             'dx'       : x        , 
                             'dy'       : y        , 
                             'dz'       : z        , 
                             'energy'   : energy   , 
                             'time'     : time     , 
                             'nphotons' : nphotons})
    return create_diff_df

def diff_writer(h5out):
    """
    For a given open table returns a writer for diffusion electrons dataframe
    """
    def write_diff(df):
        return df_writer(h5out              = h5out                      ,
                         df                 = df                         ,
                         group_name         = 'Diffsim'                   ,
                         table_name         = 'Diff_electrons'           ,
                         descriptive_string = 'Diffused electrons'       ,
                         columns_to_index   = ['event']                  )
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
            , rate           : float
            ):

    buffer_params_  = buffer_params .copy()
    physics_params_ = physics_params.copy()

    buffer_params_["max_time"] = check_max_time(buffer_params_["max_time"], buffer_params_["length"])

    filter_delayed_hits = fl.map(filter_hits_after_max_time(buffer_params_["max_time"]),
                                 args = ('x', 'y', 'z', 'energy', 'time', 'label', 'event_number'),
                                 out  = ('x', 'y', 'z', 'energy', 'time', 'label'))

    select_s1_candidate_hits = fl.map(hits_selector(False),
                                item = ('x', 'y', 'z', 'energy', 'time', 'label', 'name', 'name_id'))

    select_active_hits = fl.map(hits_selector(True),
                                args = ('x', 'y', 'z', 'energy', 'time', 'label', 'name', 'name_id'),
                                out = ('x_a', 'y_a', 'z_a', 'energy_a', 'time_a', 'labels_a'))

    filter_events_no_active_hits = fl.map(lambda x:np.any(x),
                                          args= 'energy_a',
                                          out = 'passed_active')
    events_passed_active_hits = fl.count_filter(bool, args='passed_active')

    simulate_electrons = fl.map(ielectron_simulator_diffsim(**physics_params_),
                                args = ('x_a', 'y_a', 'z_a', 'time_a', 'energy_a'),
                                out  = ('x_ph', 'y_ph', 'z_ph', 'energy_ph', 'times_ph', 'nphotons'))
    
    creates_diff_df = fl.map(diff_df_creator(), 
                             args = ('event_number', 'x_ph', 'y_ph', 'z_ph', 'energy_ph', 'times_ph', 'nphotons'), 
                             out = ('diff_df'))

    count_photons = fl.map(lambda x: np.sum(x) > 0,
                           args= 'nphotons',
                           out = 'enough_photons')
    
    dark_events   = fl.count_filter(bool, args='enough_photons')

    event_count_in = fl.spy_count()
    evtnum_collect = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        write_nohits_filter   = fl.sink(event_filter_writer(h5out, "active_hits"), args=("event_number", "passed_active") )
        write_dark_evt_filter = fl.sink(event_filter_writer(h5out, "dark_events"), args=("event_number", "enough_photons"))
        write_diff            = fl.sink(diff_writer        (h5out = h5out       ), args=("diff_df")                       )

        result = fl.push(source= MC_hits_from_files(files_in, rate),
                         pipe  = fl.pipe( fl.slice(*event_range, close_all=True)
                                        , event_count_in.spy
                                        , print_every(print_mod)
                                        , filter_delayed_hits
                                        , select_s1_candidate_hits
                                        , select_active_hits
                                        , filter_events_no_active_hits
                                        , fl.branch(write_nohits_filter)
                                        , events_passed_active_hits.filter
                                        , simulate_electrons
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

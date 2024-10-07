"""
-----------------------------------------------------------------------
                              Sensim
-----------------------------------------------------------------------
From Sensor Simulation. This city reads energy deposits (hits) and simulates
the SiPMs signal, using the PSF. 
This city outputs:
    - sns_df : SiPMs position, z-slice and signal 
    - MC  info
    - Run info
    - Filters
"""

import os
import warnings
import numpy  as np
import tables as tb

from . components import city
from . components import print_every
from . components import copy_mc_info
from . components import collect
from .components import MC_hits_and_part_from_files
from . components import check_max_time

from .. core.configure import check_annotations
from .. core.configure import EventRangeType
from .. core.configure import OneOrManyFiles

from .. core     import tbl_functions as tbl
from .. database import load_db       as db
from .. dataflow import dataflow      as fl

from .. io.event_filter_io  import event_filter_writer
from .. io.dst_io           import df_writer

from .. detsim.simulate_electrons import generate_ionization_electrons
from .. detsim.simulate_electrons import drift_electrons
from .. detsim.simulate_electrons import diffuse_electrons
from .. detsim.light_tables_c     import LT_SiPM
from .. detsim.s2_waveforms_c     import create_wfs_label

from .diffsim import binclass_creator
from .diffsim import hits_particle_df_creator
from .diffsim import segclass_creator
from .diffsim import ielectron_simulator_diffsim

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
def ielectron_simulator(*, wi: float, fano_factor: float, lifetime: float,
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
        dtimes = dz/drift_velocity + np.repeat(time, nelectrons)
        nphotons = np.random.normal(el_gain, np.sqrt(el_gain * conde_policarpo_factor), size=nelectrons.sum())
        nphotons = np.round(nphotons).astype(np.int32)
        return dx, dy, dz, dtimes, nphotons
    return simulate_ielectrons


def buffer_times_and_length_getter(pmt_width, sipm_width, el_gap, el_dv, max_length):
    """
    Auxiliar function that computes the signal absolute starting-time and an estimated buffer_length
    """
    max_sensor_bin = max(pmt_width, sipm_width)
    def get_buffer_times_and_length(time, times_ph):
        start_time = np.floor(min(time) / max_sensor_bin) * max_sensor_bin
        el_traverse_time = el_gap / el_dv
        end_time   = np.ceil((max(times_ph) + el_traverse_time)/max_sensor_bin) * max_sensor_bin
        buffer_length = min(max_length, end_time-start_time)
        return start_time, buffer_length
    return get_buffer_times_and_length


def s2_waveform_creator_sensim(sns_bin_width, LT, el_drift_velocity):
    """
    Same function as create_wfs in module detsim.s2_waveforms_c with poissonization.
    See description of the refered function for more details.
    """
    def create_s2_waveform(xs, ys, ts, ss, phs, tmin, buffer_length):
        waveforms = create_wfs_label(xs, ys, ts, ss, phs, LT, el_drift_velocity, sns_bin_width, buffer_length, tmin)
        return np.random.poisson(waveforms)
    return create_s2_waveform

def bin_edges_getter_sensim(sipm_width):
    """
    Auxiliar function that returns the waveform bin edges
    """
    def get_bin_edges(sipm_wfs):
        sipm_bins = np.arange(0, sipm_wfs.shape[2]) * sipm_width
        return sipm_bins
    return get_bin_edges

def sns_signal_getter(datasipm, dv, qcut):
    '''
    Using the SiPM WF bins, this function creates a dataframe with the pes 
    for each SiPM in each time bin (Z position)
    Also uses qcut to put a threshold on the pes of each SiPM
    '''
    def get_sns_signal(sipm_bin_wfs_seg, sipm_bins, event_energy, nevent, binclass):
        # collapse all the 3 histograms into one 
        sipm_bin_wfs = sipm_bin_wfs_seg.sum(axis = 0)
        # extract the sipms that had signal in a certain time bin
        values_index = sipm_bin_wfs.nonzero()

        # get the positions of those sipms
        xy_positions = datasipm.loc[values_index[0]][['X', 'Y']].rename(columns = {'X':'x_sipm', 'Y':'y_sipm'})
        # get the time bin of those and transform into Z position
        tbin = sipm_bins[values_index[1]]
        zbin = tbin * dv

        # get the number of photons
        pes = sipm_bin_wfs[values_index]

        # get the segmentation class from 3 histograms, choosing the label that created more gammas in a sensor and bin
        seg_bin_wfs = sipm_bin_wfs_seg.argmax(axis = 0)
        # choose only where there is any signal
        seg = seg_bin_wfs[values_index]

        # Create the df
        sipm_df = xy_positions.copy()
        sipm_df['z_slice']  = zbin
        sipm_df['pes']      = pes
        sipm_df['segclass'] = seg + 1 # return to original labels, because here we used position 
        sipm_df['event']    = nevent
        sipm_df['binclass'] = binclass

        # We put a threshold on pes for the SiPMs
        sipm_df = sipm_df[sipm_df['pes'] > qcut]

        # Finally we distribute the total energy of the event
        sipm_df['energy']   = (sipm_df['pes'] / sipm_df['pes'].sum()) * event_energy.sum()

        return sipm_df[['event', 'x_sipm', 'y_sipm', 'z_slice', 'energy', 'pes', 'binclass', 'segclass']]
    return get_sns_signal

def sns_writer(h5out):
    """
    For a given open table returns a writer for diffusion electrons dataframe
    """
    def write_sns(df):
        return df_writer(h5out              = h5out                ,
                         df                 = df                   ,
                         group_name         = 'Sensim'             ,
                         table_name         = 'sns_df'             ,
                         descriptive_string = 'Photons/SiPM/t_bin' ,
                         columns_to_index   = ['event']            )
    return write_sns

@city
def sensim( *
          , files_in       : OneOrManyFiles
          , file_out       : str
          , event_range    : EventRangeType
          , print_mod      : int
          , compression    : str
          , detector_db    : str
          , run_number     : int
          , sipm_psf       : str
          , buffer_params  : dict
          , physics_params : dict
          , label_params   : dict
          , qcut           : int
          , rate           : float
          ):

    buffer_params_  = buffer_params .copy()
    physics_params_ = physics_params.copy()

    buffer_params_["max_time"] = check_max_time(buffer_params_["max_time"], buffer_params_["length"])

    ws    = physics_params_.pop("ws")
    el_dv = physics_params_.pop("el_drift_velocity")

    datasipm = db.DataSiPM(detector_db, run_number)
    lt_sipm  = LT_SiPM(fname=os.path.expandvars(sipm_psf), sipm_database=datasipm)
    el_gap   = lt_sipm.el_gap_width

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
                             args = ('particle_id', 'particle_name', 'creator_proc', 'hit_part_id'), 
                             out  = 'binclass')
    
    creates_hits_part_df = fl.map(hits_particle_df_creator(), 
                                  args = ('hit_part_id_a','hit_id_a', 'energy_a', 'particle_id', 'particle_name', 'creator_proc'), 
                                  out  = 'hits_part_df')
    
    assign_segclass = fl.map(segclass_creator(**label_params), 
                             args = ('hits_part_df', 'binclass'), 
                             out  = ('segclass_a', 'ext_a'))

    simulate_electrons = fl.map(ielectron_simulator_diffsim(**physics_params_),
                                args = ('x_a', 'y_a', 'z_a', 'time_a', 'energy_a', 'segclass_a'),
                                out  = ('x_ph', 'y_ph', 'z_ph', 'energy_ph', 'times_ph', 'nphotons', 'segclass_ph'))

    count_photons = fl.map(lambda x: np.sum(x) > 0,
                           args= 'nphotons',
                           out = 'enough_photons')
    dark_events   = fl.count_filter(bool, args='enough_photons')

    get_buffer_info = buffer_times_and_length_getter(buffer_params_["pmt_width"],
                                                     buffer_params_["sipm_width"],
                                                     el_gap, el_dv,
                                                     buffer_params_["max_time"])
    get_buffer_times_and_length = fl.map(get_buffer_info,
                                         args = ('time', 'times_ph'),
                                         out = ('tmin', 'buffer_length'))

    create_sipm_waveforms = fl.map(s2_waveform_creator_sensim(buffer_params_["sipm_width"], lt_sipm, el_dv),
                                   args = ('x_ph', 'y_ph', 'times_ph', 'segclass_ph', 'nphotons', 'tmin', 'buffer_length'),
                                   out = 'sipm_bin_wfs')

    get_bin_edges  = fl.map(bin_edges_getter_sensim(buffer_params_["sipm_width"]),
                            args = ('sipm_bin_wfs'),
                            out = ('sipm_bins'))
    get_sns_signal = fl.map(sns_signal_getter(datasipm, physics_params_["drift_velocity"], qcut), 
                            args = ('sipm_bin_wfs', 'sipm_bins', 'energy_ph', 'event_number', 'binclass'),
                            out = ('sipm_df'))

    event_count_in = fl.spy_count()
    evtnum_collect = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        write_nohits_filter   = fl.sink(event_filter_writer(h5out, "active_hits"), args=("event_number", "passed_active") )
        write_dark_evt_filter = fl.sink(event_filter_writer(h5out, "dark_events"), args=("event_number", "enough_photons"))
        write_sns             = fl.sink(sns_writer         (h5out = h5out       ), args=("sipm_df")                       )


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
                                        , count_photons
                                        , fl.branch(write_dark_evt_filter)
                                        , dark_events.filter
                                        , get_buffer_times_and_length
                                        , create_sipm_waveforms
                                        , get_bin_edges
                                        , get_sns_signal
                                        , fl.branch(write_sns)
                                        , "event_number"
                                        , evtnum_collect.sink),
                         result = dict(events_in     = event_count_in.future,
                                       evtnum_list   = evtnum_collect.future,
                                       dark_events   = dark_events   .future))

        copy_mc_info(files_in, h5out, result.evtnum_list,
                     detector_db, run_number)

        return result

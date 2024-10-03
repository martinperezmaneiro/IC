import numpy  as np
import pandas as pd

def select_main_track(part, binclass, sig_creator):
    '''
    Uses particle information to select the particles that comform the main track,
    to separate them from the rest of the hits.

    If the event is signal (binclass=1), it selects the particles based in their 
    creator process, as signal is different in double beta / double escape events.
    If the event is background (binclass=0), it selects the electron with most energy.
    '''
    if binclass:
        main_track = part[(part.particle_name.isin(['e+', 'e-'])) & (part.creator_proc == sig_creator)]
    else:
        main_track_candidates = part[(part.particle_name == 'e-') & (part.creator_proc.isin(['compt', 'phot', 'none']))]
        main_idx = main_track_candidates.track_ener.idxmax()
        main_track = main_track_candidates.loc[[main_idx]]
    return main_track

def label_track_and_other(part, main_track, segclass_dct):
    '''
    Assign to the hits the label based on the information from the main track 
    (function above).
    '''
    label_part = part.merge(main_track.assign(segclass = segclass_dct['track']), how = 'outer') 
    label_part.segclass = label_part.segclass.fillna(segclass_dct['other']).astype(int)
    return  label_part

def label_blob_hits(label_hits, delta_ener_loss, segclass_dct):
    '''
    Label the end hits of the main track as blob, using a certain energy % loss of the total track 
    as threshold to define it.
    '''
    label_hits = label_hits.sort_values(['particle_id', 'hit_id'], ascending=[True, False])
    label_hits = label_hits.assign(cumenergy = label_hits.groupby('particle_id').energy.cumsum())
    label_hits = label_hits.assign(lost_ener = (label_hits.cumenergy / label_hits.track_ener).fillna(0))
    label_hits.loc[(label_hits.segclass == segclass_dct['track']) & (label_hits.lost_ener < delta_ener_loss), 'segclass'] = segclass_dct['blob']
    return label_hits

def small_blob_fix(label_hits, main_track, segclass_dct):
    '''
    For some signal events, one of the main track particles have so little energy that no hit 
    is labelled as blob, so the event ends with just 1 blob and has no representation.
    In this function, this is checked, and if it happens, it forces all its hits to be a blob.
    '''
    missing_blob_mask = main_track[['particle_id']].merge(label_hits[label_hits.segclass == segclass_dct['blob']][['particle_id']].drop_duplicates(), how='left', indicator=True)._merge == 'left_only'
    blobless_tracks  = main_track[missing_blob_mask.values]

    missing_hits_mask = label_hits[['particle_id']].merge(blobless_tracks, how='left', indicator=True)._merge == 'both'
    label_hits.loc[(label_hits.segclass == segclass_dct['track']) & missing_hits_mask.values, 'segclass'] = segclass_dct['blob']
    return label_hits

def get_extremes_label(hits_part, main_track, binclass):
    main_track_hits = hits_part.merge(main_track, how = 'inner')
    #signal
    if binclass:
        extreme_hits = main_track_hits.groupby('particle_id').apply(lambda x: x.loc[x['hit_id'].idxmax()]).reset_index(drop = True).sort_values(['track_ener'], ascending= False)
        extreme_hits['ext'] = [1, 2]
    #background
    else:
        start_hit = main_track_hits.groupby(['particle_id']).apply(lambda x: x.loc[x['hit_id'].idxmin()]).reset_index(drop=True)
        end_hit   = main_track_hits.groupby(['particle_id']).apply(lambda x: x.loc[x['hit_id'].idxmax()]).reset_index(drop=True)
        start_hit['ext'] = 2
        end_hit['ext']   = 1
        extreme_hits = pd.concat([start_hit, end_hit])

    return hits_part.merge(extreme_hits[['particle_id', 'hit_id', 'ext']], how = 'outer').fillna(0).ext.values.astype(int)
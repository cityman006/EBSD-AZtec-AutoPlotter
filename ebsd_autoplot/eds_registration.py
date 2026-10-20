"""Apply an explicit, source-bound EDS landmark project to an existing cache."""
from __future__ import annotations

from .registration import validate_landmark_config, resample_map_blocked
from .storage import check_cancelled


def load_registered_eds(project, cache_path, *, count_rate=False, block_rows=32):
    """Read native intensities and register all channels without reanalysis.

    Timing arrays stay on their native grid. Linear interpolation is explicit,
    unknown/outside support is masked, and residual QC is not external truth.
    Caller owns the returned EDSResult and must close its scratch buffers.
    """
    import numpy as np
    from .eds import load_native_h5oina_eds
    frozen = validate_landmark_config(project, cache_path=cache_path)
    check_cancelled()
    print('EDS: reading native source for landmark registration', flush=True)
    result = load_native_h5oina_eds(frozen['source_path'], frozen['scan_path'],
                                  count_rate=count_rate, block_rows=block_rows)
    try:
        if not result.available or not result.maps:
            raise ValueError('Landmark source has no native EDS window maps')
        shape = next(iter(result.maps.values())).shape
        frozen = validate_landmark_config(frozen, cache_path=cache_path, source_shape=shape)
        target = tuple(frozen['target_shape_hw'])
        print('EDS: resampling explicit landmark registration', flush=True)
        for name in list(result.maps):
            check_cancelled()
            print('EDS: registering ' + name, flush=True)
            values = result.allocate(target, np.float64)
            valid = result.allocate(target, bool)
            result.maps[name], result.masks[name] = resample_map_blocked(
                result.maps[name], target, frozen['matrix'], result.masks[name],
                block_rows=block_rows, out=values, valid_out=valid)
        validate_landmark_config(frozen, cache_path=cache_path, source_shape=shape)
        result.metadata['registration'] = {
            'method': 'explicit paired-landmark affine', 'resampled': True,
            'source_to_ebsd_pixels': frozen['matrix'], 'landmark_project': frozen,
            'aligned_to_ebsd': True, 'independently_validated': False,
            'interpolation': 'linear on native EDS intensities/rates; no extrapolation',
            'outside': 'NaN plus false validity mask',
            'timing': 'retained on original native EDS grid; not resampled'}
        return result
    except BaseException:
        result.close()
        raise

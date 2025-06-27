import numpy as np
import matplotlib.pyplot as plt
import os, yaml
import pipeline.reduction as red
from pathlib import Path
from scipy.interpolate import interp1d
from astropy.io import fits

def plot_scaled_model(wave_mod, mod_spec, mod_spec_scaled, path_fig):
    plt.figure(figsize = (10, 4), dpi = 200)
    plt.subplot(211)
    plt.plot(wave_mod, mod_spec, label = 'Original model', linewidth = 0.5)
    plt.legend()
    plt.subplot(212)
    plt.plot(wave_mod, 1 - mod_spec_scaled, label = 'Scaled model')
    plt.legend()

    if path_fig != '': plt.savefig(path_fig + '/scaled_model.pdf', bbox_inches = 'tight')

def main(config_dict, p, obs, visit_name, wave_mod, mod_spec, scratch_dir = None, path_fig = '', debug = False):

    ''' Takes as input a model and a raw data file to inject the model into at a given RV.
    Assumed a list of t.fits files for the night have already been made (so, that split nights has been run
    
    wave_mod and mod_spec: outputs of petitradtrans model -- mod_spec should be in units of transit depth (R_pl/R_star)**2
    '''

    # set save directory
    if scratch_dir == None:
        scratch_dir = Path(os.environ["SCRATCH"])
        scratch_dir /= Path(f'{config_dict["instrument"]}/Reductions/injected_fits/{"".join(p.name.split())}')
        scratch_dir.mkdir(parents=True, exist_ok=True)
    
    # getting the exposures in and out of transit
    obs.calc_sequence()
    in_transit = obs.iIn
    out_transit = obs.iOut

    # getting the lightcurve
    exp_times = obs.t
    lightcurve = obs.alpha

    # create the window function to scale the model 
    Wc = lightcurve / np.max(lightcurve)

    # Getting velocities for correction after
    obs.norv_sequence(RV=obs.planet.RV_sys.value[0])  # offset the RVs so that they are 0 at mid-transit
    
    # Calculate the velocity shift correction
    vshift = -obs.berv0 + obs.RV_sys + obs.vrp + obs.mid_vrp + config_dict['RV_inj']

    # Scale the inputted model
    mod_spec_scaled = (mod_spec - (p.R_pl / p.R_star)**2) * config_dict['scaling_factor']

    # save a plot of the scaled model
    plot_scaled_model(wave_mod, mod_spec, mod_spec_scaled, path_fig)

    # get list of all exposures
    with open(str(config_dict['obs_dir']) + '/' + f'list_tcorr_{visit_name}') as f:
        exp_list = f.readlines()

    # Initialize lists to store data
    wavelengths = []
    counts = []

    for i, exp in enumerate(exp_list):
        
        # Shift the model wavelength into the observer rest frame for this exposure
        wave_mod_shifted = wave_mod * (1 + vshift[i] / 299792.458)

        # setup a funciton to interpolate over the model
        interp_wavelength = interp1d(wave_mod_shifted, mod_spec_scaled, kind='cubic')
        
        if debug:
            # plot each exposure
            plt.figure(figsize = (10, 4), dpi = 200)
        
        # load the exposure
        with fits.open(str(config_dict['obs_dir']) + '/' + exp.strip(), memmap=False) as hdul:

            count = hdul[1].data
            wv = hdul[2].data / 1000
            
            new_count = count.copy()  # create a copy to modify

            # Iterate over the orders
            for w in range(len(wv)):

                if debug:
                    # plot original 
                    if w == 0: plt.plot(wv[w], count[w], label='Original')
                    else: plt.plot(wv[w], count[w])

                # interpolate the model to the wavelength, and scale by the lightcurve
                mod_interp = 1 - interp_wavelength(wv[w]) * Wc[i]

                # multiply the count by the model in that range
                new_count[w] = count[w] * mod_interp

                if debug:
                    # plot new
                    if w == 0: plt.plot(wv[w], hdul[1].data[w], label='Injected', color = 'blue', zorder = 0)
                    else: plt.plot(wv[w], hdul[1].data[w], color = 'blue', zorder = 0)

            # Assign the modified data back to the HDU
            hdul[1].data = new_count
            
            # save to new fits file with same name, in new folder
            hdul.writeto(str(scratch_dir / exp.strip()), overwrite=True)

            # Append wavelengths and counts for this exposure to the lists
            wavelengths.append(wv)
            counts.append(hdul[1].data)

            if debug:
                plt.title(f'Exposure {exp.strip()}')
                plt.legend()
                plt.savefig(str(scratch_dir / 'wavelength_time_plot.pdf'))
                # plt.show()

    # copy rest of files into scratch so we can use it as a new obs_dir
    # get list of all exposures
    with open(str(config_dict['obs_dir']) + '/' + f'list_e2ds_{visit_name}') as f:
        e2ds_list = f.readlines()

    for i, e2ds in enumerate(e2ds_list):
        os.system(f'cp {config_dict["obs_dir"]}/{e2ds.strip()} {scratch_dir}')

    # copy the e2ds and tcorr lists into the scratch
    os.system(f'cp {config_dict["obs_dir"]}/list_tcorr_{visit_name} {scratch_dir}')
    os.system(f'cp {config_dict["obs_dir"]}/list_e2ds_{visit_name} {scratch_dir}')

    
    

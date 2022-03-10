import pandas as pd
import numpy as np
import datetime as dt
from PyAstronomy import pyasl
from astropy.time import Time
import astropy.coordinates as coord
import astropy.units as u
import wget
import os.path
from tqdm import tqdm

from astral import LocationInfo
from astral.sun import dawn, dusk


def ComputeSNR(m, t, airm, method='Crires', mband='K'):
    '''
    Computes SNR of stellar spectrum from magnitude, exposure time (in seconds) and airmass
    
    Parameters
    ----------
    m : float
        Magnitude in band specified with the parameter "mband"
    t : float
        Exposure time [s]
    airm : float
        Airmass
    method : string, optional, {"Crires", "Carmenes"}
        Specifiy the instrument for/ method with which to compute the SNR
    mband : string, optional, {"K, J"}
        Specifiy in which band the magnitude "m" is measured
        
    Returns
    -------
    SNR : float
        Resulting SNR (only a rough estimate)
    '''
    ###First compute SNR assuming airmass = 1
    extcof = 0.05 #extinction coefficient, see Lombardi et al., 2018
    #This is from old Carmenes documentation, factor 1.1774 so that it agrees better with Ansgars result
    if method == 'Carmenes':
        if mband == 'J':
            SNR_noairmass = 1.1774*100/np.sqrt(40*10**((m-4.2)/2.5))*np.sqrt(t)
        else:
            print('Use Jmag for calculation of Carmenes SNR.')
            SNR_noairmass = np.nan
    elif method == 'Crires_old':
        if mband == 'K':
            SNR_noairmass = 449.4241*np.sqrt(10**(-m/2.5))*np.sqrt(t)- 6.3144
        else:
            print('Use Kmag for calculation of old Crires SNR.')
            SNR_noairmass = np.nan
    elif method == 'Crires':
        if mband == 'K':
            snr_airm1_2 = 247.31342303*np.sqrt(10**(-m/2.5))*np.sqrt(t) - 3.20150241
            SNR_noairmass = snr_airm1_2 * 10**(extcof/5*(1.2 - 1))
        elif mband == 'J':
            snr_airm1_2 = 479.05726751*np.sqrt(10**(-m/2.5))*np.sqrt(t) - 2.90701871
            SNR_noairmass = snr_airm1_2 * 10**(extcof/5*(1.2 - 1))
        else:
            print('Use Kmag or Jmag for calculation of Crires SNR.')
            SNR_noairmass = np.nan
        
    else:
        print('Method not recognized. Use Crires or Carmenes.')
        SNR_noairmass = np.nan
              
    #Scale to airmass = airm
    SNR = SNR_noairmass * 10**(extcof/5*(1 - airm))
    
    return SNR
    
def GetExpTimeForSNR(desiredSNR, m, airm, method='Crires', mband='K', verbose = True):
    '''
    Computes required exposure time to achieve a specified SNR value

    Parameters
    ----------
    desiredSNR : float
        The SNR that is desired
    m : float
        Magnitude in band specified with the parameter "mband"
    airm : float
        Airmass
    method : string, optional, {"Crires", "Carmenes"}
        Specifiy the instrument for/ method with which to compute the SNR
    mband : string, optional, {"K, J"}
        Specifiy in which band the magnitude "m" is measured
    verbose : bool
        Set to true if a message with the computed exp time and resulting SNR should be printed        
        
    Returns
    -------
    exptime : float
        Required exposure time in seconds
    '''
    #Compute rough estimate with exp time scaling (is always to large)
    SNR_10s = ComputeSNR(m, 10, airm, method, mband)
    estimated_required_expt = (desiredSNR / SNR_10s)**2 * 10
    required_expt = estimated_required_expt
    achieved_SNR = ComputeSNR(m, required_expt, airm, method, mband)
    
    #reduce exp time until SNR is below desired value
    while achieved_SNR > desiredSNR:
        required_expt -= 1
        achieved_SNR = ComputeSNR(m, required_expt, airm, method, mband)
    if verbose:
        print('SNR = {} with exposure time of '.format(round(achieved_SNR, 2)) + str(round(required_expt, 2)))
        
    return(required_expt)


def GetPlanetDataNexa(planet, catalogpath = '../NexaComposite.csv'):
    '''
    Loads data for given planet from nexa file and returns dictionary with all information.
    Downloads nexa catalog if file at filepath does not exist.
    
    Parameters
    ----------
    planet : string
        Name of the planet as it appears in the Nexa catalog
    
    Returns
    -------
    planetData : dictionary
        Dictionary containing the planetary parameters:
        
        ============    ====================================================
        Key             Description
        ------------    ----------------------------------------------------
        plName          Name of the planet
        ra              Right Ascension [decimal degrees]
        dec             Declination [decimal degrees]
        T0              T0 [BJD]
        orbPer          Orbital period [days]
        orbPerErr       Error of orbital period [days]
        orbInc          Orbital inclination [deg]
        orbEcc          Orbital eccentricity 
        SMA             Semi-major axis [au]
        RpJ             Planetary radius [R_Jup]
        RsSun           Stellar radius [R_Sun]
        MpJ             Planetary mass [M_Jup]
        MsSun           Stellar mass [M_Sun]
        Tdur            Transit duration [days]
        TransitDepth    Transit depth [%]
        Teq             Equilibirum temperature of the planet [K]
        Teff            Effective temperature of the host star [K]
        Vmag            Magnitude in the V band
        Hmag            Magnitude in the H band
        Jmag            Magnitude in the J band
        Kmag            Magnitude in the K band
    '''
    filepath = catalogpath
    
    if os.path.exists(filepath) == False:
        #download data if no file exists
        DownloadDataNexa(filepath)
    nexaData = pd.read_csv(filepath)
    
    pl_index = np.where(nexaData['pl_name'] == planet)[0]
    
    if pl_index.size == 0:
            #print(planet + ' not in the Nexa database')
        return None
    else:
        pl_index = pl_index[0]
        
    #Set error to 0 if missing
    pl_tranmiderr1 = nexaData['pl_tranmiderr1'][pl_index]
    pl_tranmiderr2 = nexaData['pl_tranmiderr2'][pl_index]
    pl_orbpererr1 = nexaData['pl_orbpererr1'][pl_index]
    pl_orbpererr2 = nexaData['pl_orbpererr2'][pl_index]
    
    if pd.isnull(pl_tranmiderr1):
        pl_tranmiderr1 = 0
    if pd.isnull(pl_tranmiderr2):
        pl_tranmiderr2 = 0
    if pd.isnull(pl_orbpererr1):
        pl_orbpererr1 = 0
    if pd.isnull(pl_orbpererr2):
        pl_orbpererr2 = 0    
        
    T0Err = np.max([pl_tranmiderr1, -pl_tranmiderr2])
    orbPerErr = np.max([pl_orbpererr1, -pl_orbpererr2])
    
    planetData = {'plName': nexaData['pl_name'][pl_index],
             'ra': nexaData['ra'][pl_index],
             'dec': nexaData['dec'][pl_index],
             'T0': nexaData['pl_tranmid'][pl_index],
             'T0Err': T0Err,
             'orbPer': nexaData['pl_orbper'][pl_index],
             'orbPerErr': orbPerErr,
             'orbInc': nexaData['pl_orbincl'][pl_index],
             'orbEcc': nexaData['pl_orbeccen'][pl_index],
             'Periastron':nexaData['pl_orblper'][pl_index],
             'ProjObliquity':nexaData['pl_projobliq'][pl_index],
             'SMA': nexaData['pl_orbsmax'][pl_index],
             'RpJ': nexaData['pl_radj'][pl_index],
             'RsSun': nexaData['st_rad'][pl_index],
             'MpJ': nexaData['pl_bmassj'][pl_index],
             'MsSun': nexaData['st_mass'][pl_index],
             'Tdur': nexaData['pl_trandur'][pl_index] / 24, #convert from hours to days
             'TransitDepth': nexaData['pl_trandep'][pl_index],
             'ImpactPar':nexaData['pl_imppar'][pl_index],
             'Teq': nexaData['pl_eqt'][pl_index],
             'Teff': nexaData['st_teff'][pl_index],
             'Vsini':nexaData['st_vsin'][pl_index],
             'RadVelStar':nexaData['st_radv'][pl_index],
             'Vmag': nexaData['sy_vmag'][pl_index],
             'Hmag': nexaData['sy_hmag'][pl_index],
             'Jmag': nexaData['sy_jmag'][pl_index],
             'Kmag': nexaData['sy_kmag'][pl_index],
             'Logg': nexaData['st_logg'][pl_index],
             'Metal': nexaData['st_mass'][pl_index]}
    
    return planetData
    
def DownloadDataNexa(filepath):
    '''
    Downloads all relevant columns for all planets from Nexa
    
    Parameters
    ----------
    filepath : string
        Location where the downloaded catalog is to be stored.
    '''
    properties = ['pl_name', 'pl_bmassj', 'pl_eqt', 'pl_imppar', 'pl_orbeccen', 'pl_orbincl', 'pl_orblper',
                  'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2', 'pl_orbsmax', 'pl_projobliq', 'pl_radj',
                  'pl_trandep', 'pl_trandur', 'pl_tranmid', 'pl_tranmiderr1', 'pl_tranmiderr2',
                  'ra', 'dec', 'sy_hmag', 'sy_jmag', 'sy_kmag', 'sy_vmag', 
                  'st_logg', 'st_mass', 'st_met', 'st_rad', 'st_radv', 'st_teff', 'st_vsin']                   
                   
    urlRoot = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    select = "select+"
    for p in properties:
        select = ''.join([select, ',', p])
    select = select[:7] + select[8:]
    table = "+from+pscomppars"
    outformat = "&format=csv"

    url = ''.join((urlRoot, select, table, outformat))

    wget.download(url, out=filepath)
    
def ListAllPlanetsNexa(maxKmag = 12):
    '''
    Creates a list of all planet names of Nexa with Kmag < maxKmag
    
    Parameters
    ----------
    maxKmag : string
        Maximum K magnitude a planets host star can have to be included in the list
        
    Returns
    -------
    planetlist : list
        A list of all planets that have Kmag < maxKmag
    '''
    filepath = '../NexaCatalog.csv'
    if os.path.exists(filepath) == False:
        #download data if no file exists
        DownloadDataNexa(filepath)
    nexaData = pd.read_csv(filepath)

    planetlist = []
    for planetnumber in range(len(nexaData['pl_name'])):
        if nexaData['sy_kmag'][planetnumber] < maxKmag:
            planetlist.append(nexaData['pl_name'][planetnumber])
            
    return planetlist
    
def GetPlanetDataCustom(planet):
    '''
    Loads parameters of given planet from a file with a custom table. The file has to contain the same column names as defined in Nexa.
    
    Parameters
    ----------
    planet : string
        Name of the planet as it appears in the Nexa catalog
    
    Returns
    -------
    planetData : dictionary
        Dictionary containing the planetary parameters:
        
        ============    ====================================================
        Key             Description
        ------------    ----------------------------------------------------
        plName          Name of the planet
        ra              Right Ascension [decimal degrees]
        dec             Declination [decimal degrees]
        T0              T0 [BJD]
        orbPer          Orbital period [days]
        orbPerErr       Error of orbital period [days]
        orbInc          Orbital inclination [deg]
        orbEcc          Orbital eccentricity 
        SMA             Semi-major axis [au]
        RpJ             Planetary radius [R_Jup]
        RsSun           Stellar radius [R_Sun]
        MpJ             Planetary mass [M_Jup]
        MsSun           Stellar mass [M_Sun]
        Tdur            Transit duration [days]
        TransitDepth    Transit depth [%]
        Teq             Equilibirum temperature of the planet [K]
        Teff            Effective temperature of the host star [K]
        Vmag            Magnitude in the V band
        Hmag            Magnitude in the H band
        Jmag            Magnitude in the J band
        Kmag            Magnitude in the K band
    '''
    filepath = '../CustomCatalog.csv'
    
    if os.path.exists(filepath) == False:
        print('Custom Catalog not found!')
    customData = pd.read_csv(filepath)
    
    pl_index = np.where(customData['pl_name'] == planet)[0]
    
    if pl_index.size == 0:
        print(planet + ' not in the custom database')
        return None
    else:
        pl_index = pl_index[0]

    #Set error to 0 if missing
    pl_tranmiderr1 = customData['pl_tranmiderr1'][pl_index]
    pl_tranmiderr2 = customData['pl_tranmiderr2'][pl_index]
    pl_orbpererr1 = customData['pl_orbpererr1'][pl_index]
    pl_orbpererr2 = customData['pl_orbpererr2'][pl_index]
    
    if pd.isnull(pl_tranmiderr1):
        pl_tranmiderr1 = 0
    if pd.isnull(pl_tranmiderr2):
        pl_tranmiderr2 = 0
    if pd.isnull(pl_orbpererr1):
        pl_orbpererr1 = 0
    if pd.isnull(pl_orbpererr2):
        pl_orbpererr2 = 0    
        
    T0Err = np.max([pl_tranmiderr1, -pl_tranmiderr2])
    orbPerErr = np.max([pl_orbpererr1, -pl_orbpererr2])
    
    planetData = {'ra':customData['ra'][pl_index],
             'dec':customData['dec'][pl_index],
             'T0':customData['pl_tranmid'][pl_index],
             'T0Err': T0Err,
             'orbPer': customData['pl_orbper'][pl_index],
             'orbPerErr': orbPerErr,
             'orbInc':customData['pl_orbincl'][pl_index],
             'orbEcc':customData['pl_orbeccen'][pl_index],
             'SMA':customData['pl_orbsmax'][pl_index],
             'RpJ':customData['pl_radj'][pl_index],
             'RsSun':customData['st_rad'][pl_index],
             'MpJ':customData['pl_bmassj'][pl_index],
             'MsSun':customData['st_mass'][pl_index],
             'Tdur':customData['pl_trandur'][pl_index] / 24, #convert from hours to days
             'plName':customData['pl_name'][pl_index],
             'orbPerErr':customData['pl_orbpererr1'][pl_index],
             'TransitDepth':customData['pl_trandep'][pl_index],
             'Teq':customData['pl_eqt'][pl_index],
             'Teff':customData['st_teff'][pl_index],
             'Vsini':customData['st_vsin'][pl_index],
             'RadVelStar':customData['st_radv'][pl_index],
             'Vmag': customData['sy_vmag'][pl_index],
             'Hmag': customData['sy_hmag'][pl_index],
             'Jmag': customData['sy_jmag'][pl_index],
             'Kmag': customData['sy_kmag'][pl_index],
             'Logg': customData['st_logg'][pl_index],
             'Metal': customData['st_mass'][pl_index]}
    
    return planetData
    
def CheckRequiredParameters(planetdata, mband='K'):
    '''
    Checks if all of the required parameters (ra, dec, T0, orbPer, SMA, RpJ, RsSun, Tdur, magnitude) are included 
    in the planet dictionary
    
    Parameters
    ----------
    planetData : dictionary
        Dictionary containing the planetary parameters
    
    Returns
    -------
    all_parameters_given : bool
        True if all required parameters are included in planetData, False otherwise
    '''
    if mband == 'J':
        band_parameter = 'Jmag'
    elif mband == 'K':
        band_parameter = 'Kmag'
    else:
        print('CheckRequiredParameters: Given magnitude band not recognized!')
        return False
        
    required_parameters = ['ra', 'dec', 'T0', 'orbPer', 'SMA', 'RpJ', 'RsSun', 'Tdur', band_parameter]

    all_parameters_given = True
    for parameter in required_parameters:
        if pd.isna(planetdata[parameter]):
            all_parameters_given = False

    return all_parameters_given

    
def TransitInformation(planet, catalog, observatory,d_start, d_end,observation_puffer=1/24,
                         min_required_altitude=0, max_airm_good=2, max_sunalt_good=-20, SNR_method='Crires', mband='K',
                         drop_bad_transit = True, verbose=True):
    '''
    Finds and anaylizes every tranist of the planet in a specified time frame, and computes many usefull parameters helpfull in determining the observability of transits.

    Parameters
    ----------
    planet : string
        Name of the system (as given in the respective Catalog)
    catalog : string, {"Nexa", "TEP", "Custom"}
        Name of the catalog to be used
    observatory : string
        Name of the observatory site as recognized by pyasl.observatory
    d_start : datetime object
        start date of the observation window
    d_end : datetime object
        end date of the observation window 
    observation_puffer : float, optional
        Time before and after transit that is observed (in days), default = 1/24
    min_required_altitude : float, optional
        minimal altitude of the planet during a transit such that the transit is considered as an option (degrees), default = 0
    max_airm_good : float, optional
        maximal airmass considered as 'good' observation conditions, default = 2
    max_sunalt_good : float
        maximal solar altitude considered as 'good' observation conditions, default = -20
    SNR_method : string, optional, {"Crires", "Carmenes"}
        Specifiy the instrument for/ method with which to compute the SNR values
    mband : string, optional, {"K, J"}
        Specifiy which magnitude band to use in the SNR calculation
    drop_bad_transit : bool, optional
        Set to True to only include transits with good obs. conditions in all_tr
    verbose : bool, optional
        Set to True to print status messages
        
    Returns
    -------
    all_tr : dictionary
        Dictionary containing information about each individual transit of the planet in the specified time frame:
        
        ============    ====================================================
        Key             Description
        ------------    ----------------------------------------------------
        System          Name of the planet
        Night           Date of the beginning of the night where the transit occurs (shows previous day if transit occurs after midnight)
        Tmid            Mid-tranist time [JD]
        Tmid_err        Estimated error of Tmid [JD]
        Obs_start       Start time of observation (includes observation buffer) [JD]
        Obs_end         End time of observation (includes observation buffer) [JD]
        Trans_start     Start time of transit event [JD]
        Trans_end       End time of transit event [JD]
        Sunalt_start    Solar altitude at the start of observation [deg]
        Sunalt_mid      Solar altitude at the transit midpoint [deg]
        Sunalt_end      Solar altitude at the end of observation [deg]
        Airm_start      Airmass at the start of observation
        Airm_mid        Airmass at the transit midpoint
        Airm_end        Airmass at the end of observation
        Moon_dist       Angular distance between target and moon during transit midpoint [deg]
        V_bary          Barycentric velocity in direction of star (includes Earth's rotation) at transit midpoint [km/s]
        V_total         Total velocity difference: V_system - V_bary
        GoodCond        Indicates whether good observation conditions are given (as defined by max_airm_good, max_sunalt_good) during the entire observation
        GoodCond_before Time before start of transit that is observable in good conditions [hours]
        GoodCond_after  Time after start of transit that is observable in good conditions [hours]

    system_tr : dictionary
        Dictionary containing general planetary parameters and overall information about the number of transits:
        
        ============    ====================================================
        Key             Description
        ------------    ----------------------------------------------------
        System          Name of the planet
        Mstar           Stellar mass [M_Sun]
        Rstar           Stellar radius [R_Sun]
        Mpl             Planetary mass [M_Jup]
        Rpl             Planetary radius [R_Jup]
        Period          Orbital period [days]
        Teff            Effective temperature of the host star [K]
        Kmag            Magnitude in the K band    
        transit_depth   Transit depth [%]
        transit_dur     Transit duration [days]
        SNR_nosmear     SNR with exposure time of t_nosmear
        t_nosmear       Maximal exposure time so that the planetary signal is not smeared across multiple pixel [s]
        Acc             Planetary acceleration at time of transit [km/s]
        K_p             Planetary radial velocity semi-amplitude [km/s]
        SNR_5min        SNR for a fixed exposure time of 5 minutes
        SNR_transdur    SNR when exposing for a full transit duration
        Teq             Equilibrium temperature of the planet [K]
        H               Scale height of planetary atmosphere [m]
        Delta_d         Change in transit depth due to planetary atmosphere [%]
        N_GoodTransit   Number of good transits (as defined by max_airm_good, max_sunalt_good)
        N_Trans_SNR1000 Number of transits required to achieve a SNR of 1000
        N_trans_req     Rough estimate of the number of transits required to make a detection
        N_GoodTransit_Semester   Number of good transits in the first half of 2021
        Systemscore     Ranking based on Delta_d, N_GoodTransit and N_Trans_SNR1000
                                          
    '''
        
    
    if catalog == 'TEP':
        physprop, hompar, hommes, obsplan = LoadTEP()
        planetData = GetPlanetDataTEP(planet)
    elif catalog == 'Nexa':
        planetData = GetPlanetDataNexa(planet)
        if planetData == None:
            if verbose:
                print('Planet not in Nexa, trying to search in custom catalog.')
            planetData = GetPlanetDataCustom(planet)
    elif catalog == 'Custom':
        planetData = GetPlanetDataCustom(planet)
    else:
        print('Requested catalog not recognized. Choices are TEP and Nexa')
        return None, None
    
    if planetData == None:
        if verbose:
            print(planet + ' not contained in catalog!')        
        return None, None
    
    if not CheckRequiredParameters(planetData, mband):
        if verbose:
            print('At least one of the required parameters is missing!')
        return None, None
    
    jd_start = pyasl.jdcnv(d_start)
    jd_end = pyasl.jdcnv(d_end)

    ###
    ###First create dataframe with all transits
    ###

    t_min = jd_start
    t_max = jd_end

    T0 = planetData['T0']
    T0_err = planetData['T0Err']
    period = planetData['orbPer']
    period_err =  planetData['orbPerErr']
    Tdur = planetData['Tdur']
    ra = planetData['ra']
    dec = planetData['dec']
    sys_vel = planetData['RadVelStar']

    observatory_data = pyasl.observatory(observatory)
    lon = observatory_data["longitude"]
    lat = observatory_data["latitude"]
    alt = observatory_data["altitude"]

    trnum_start = np.floor((t_min - T0) / period)
    trnum_end = np.ceil((t_max - T0) / period)
    # Relevant transit epochs
    tr = np.arange(trnum_start, trnum_end, 1)

    #Get list of all relevant times (start, mid and end of observation)
    t_list = []
    for epoch in tr:
        Tmid = T0 + float(epoch)*period
        T_before = Tmid - Tdur/2 - observation_puffer
        T_after = Tmid + Tdur/2 + observation_puffer

        if (Tmid < t_min) or (Tmid > t_max):
            # This may happen because the transit may occur in the first
            # relevant epoch but still before tmin. Likewise for tmax.
            continue

        t_list.extend([T_before, Tmid, T_after])

    t_list = np.array(t_list)  
    
    if len(t_list) == 0:
        #no transit has occured since T0
        if verbose:
            print('No transit will occure in the given time frame')
        return None, None
    #correct for light travel time to barycentre
    coos = coord.SkyCoord(ra, dec, unit=u.deg)
    loc = coord.EarthLocation(lon=lon, lat=lat, height=alt)

    t_list_timeobject = Time(t_list, format='jd')
    ltt_bary = t_list_timeobject.light_travel_time(coos, location = loc)
    t_list_corrected = (t_list_timeobject - ltt_bary)
    #subtract ltt_bary to go from barycentric to Earth frame
    t_list = t_list_corrected.value

    altaz = pyasl.eq2hor(t_list, np.ones(t_list.shape) * ra,np.ones(t_list.shape) * dec, lon=lon, lat=lat, alt=alt)
    altitude = altaz[0]

    #remove when planet not visible from observation site
    alt_filtered = []
    t_filtered = []

    nan_list = [np.nan, np.nan, np.nan]
    for i in range(int(altitude.size / 3)):
        minalt = np.where(altitude[i*3:i*3+3] >= min_required_altitude)[0]

        if (len(minalt)==3):
            #target visible
            alt_filtered.extend(altitude[i*3:i*3+3])
            t_filtered.extend(t_list[i*3:i*3+3])

    alt_filtered = np.array(alt_filtered)
    t_filtered = np.array(t_filtered)

    total_nights = int(t_filtered.size / 3)

    if total_nights == 0:
        if verbose:
            print('No transit found for ' + planet)
        return None, None

    #calculate sun altitude and airmass at each time
    notnan = pd.notnull(t_filtered)
    sunpos_radec = pyasl.sunpos(t_filtered[notnan])
    sunpos_altaz = pyasl.eq2hor(t_filtered[notnan], sunpos_radec[1][0],
                                 sunpos_radec[2][0],
                                 lon=lon, lat=lat, alt=alt)
    sunalt = np.ones(t_filtered.shape)*np.nan
    sunalt[notnan] = sunpos_altaz[0]

    airm = pyasl.airmassPP(90-alt_filtered)

    #calculate distance to moon (only for midpoints before and after)
    midpoints = np.array([t_filtered[i*3+1] for i in range(int(t_filtered.size / 3))])

    mpos = pyasl.moonpos(midpoints)
    moondist = pyasl.getAngDist(mpos[0], mpos[1], ra, dec)

    ##Get the date of each night
    t_jd_mid = np.array([t_filtered[i*3+1] for i in range(total_nights)])

    #rough determination of timezone
    timezone = ((int(lon/15)+12) % 24) - 12

    t_jd_mid += timezone/24

    #Subtract 1 day if night starts at day before (if time is before 8am)
    t_jd_mid[t_jd_mid % 1 > 0.5] -= 1
    dates_complete = Time(t_jd_mid, format='jd').iso
    night = [date[:10] for date in dates_complete]

    #Calculate error of transit midpoint time
    n_transits_since_T0 = np.round([(t_filtered[i*3+1] - T0)/period for i in range(total_nights)])

    Tmid_err = np.sqrt(T0_err**2 + (n_transits_since_T0 * period_err)**2)
    #Calculate barycentric velocity
    baryc_vel = []
    for tmid in [t_filtered[i*3+1] for i in range(total_nights)]:
        baryc_vel.append(pyasl.helcorr(lon, lat, alt, ra, dec, tmid)[0])
        
    total_vel = sys_vel - np.array(baryc_vel)
        
    data_dict = {'System': [planet for i in range(total_nights)],
                'Night': night,
                'Tmid': [t_filtered[i*3+1] for i in range(total_nights)],
                'Tmid_err': Tmid_err,
                'Obs_start': [t_filtered[i*3] for i in range(total_nights)],
                'Obs_end': [t_filtered[i*3+2] for i in range(total_nights)],
                'Trans_start': [t_filtered[i*3] + observation_puffer for i in range(total_nights)],
                'Trans_end': [t_filtered[i*3+2] - observation_puffer for i in range(total_nights)],
                'Sunalt_start': [sunalt[i*3] for i in range(total_nights)],
                'Sunalt_mid': [sunalt[i*3+1] for i in range(total_nights)],
                'Sunalt_end': [sunalt[i*3+2] for i in range(total_nights)],
                'Airm_start': np.array([airm[i*3] for i in range(total_nights)]),
                'Airm_mid': np.array([airm[i*3+1] for i in range(total_nights)]),
                'Airm_end': np.array([airm[i*3+2] for i in range(total_nights)]),
                'Moon_dist': moondist,
                'V_bary': baryc_vel,
                'V_total': total_vel}
    
    data_dict['GoodCond'] = ((np.array(data_dict['Airm_start']) < max_airm_good) & 
                               (np.array(data_dict['Sunalt_start']) < max_sunalt_good) &
                               (np.array(data_dict['Airm_end']) < max_airm_good) &
                               (np.array(data_dict['Sunalt_end']) < max_sunalt_good))

    
    #Calculate observable time before and after transit (only for good transits)
    
    altitudeFromAirmass = 90 - np.arccos(1/max_airm_good)*180/np.pi
    
    times = np.array(data_dict['Tmid'])[data_dict['GoodCond']]
    
    goodCond_before = np.empty(len(data_dict['Tmid']))*np.nan
    goodCond_after = np.empty(len(data_dict['Tmid']))*np.nan
    
    if times.size > 0:
        rising, setting = TimeAtAltitude(times, altitudeFromAirmass, planetData, observatory_data)
        sunrise, sunset = SunsetTime(times, max_sunalt_good, observatory_data)

        goodCond_start = np.copy(rising)
        goodCond_end = np.copy(setting)

        goodCond_start[(rising < sunset)] = sunset[(rising < sunset)]
        goodCond_end[(setting > sunrise)] = sunrise[(setting > sunrise)]

        goodCond_before[data_dict['GoodCond']] = np.array(data_dict['Trans_start'])[data_dict['GoodCond']] - goodCond_start
        goodCond_after[data_dict['GoodCond']] = goodCond_end - np.array(data_dict['Trans_end'])[data_dict['GoodCond']]
        
    data_dict['GoodCond_before'] = goodCond_before * 24
    data_dict['GoodCond_after'] = goodCond_after * 24
    
    #Alternative good condition: good if out of transit time is at least 70% of transit duration
    data_dict['GoodCond_2'] = (np.array(data_dict['GoodCond_before']) + np.array(data_dict['GoodCond_after']) > 0.7 * (Tdur * 24))
    
    ###Add system data that is the same for all transits of a planet
    system_dict = {}

    mag = mband + 'mag'
    #system name
    system_dict['System'] = planet
    system_dict['Mstar'] = planetData['MsSun']
    system_dict['Rstar'] = planetData['RsSun']
    system_dict['Mpl'] = planetData['MpJ']
    system_dict['Rpl'] = planetData['RpJ']
    system_dict['Period'] = planetData['orbPer']
    system_dict['Teff'] = planetData['Teff']
    system_dict[mag] = planetData[mag]
    
    #transit depth
    transit_depth = planetData['TransitDepth']
    if np.isnan(transit_depth):
        #calculate depth if not in catalog
        rjup2rsun = 0.1005
        transit_depth = (planetData['RpJ']*rjup2rsun / planetData['RsSun'])**2

    system_dict['transit_depth'] = transit_depth
    
    system_dict['transit_dur'] = data_dict['Trans_end'][0] - data_dict['Trans_start'][0]
    
    system_dict['Tmid_err'] = np.mean(Tmid_err)
    
    #stellar SNR
    #here units are km and s everywhere
    #first calculate maximum exposure so that lines do not shift between pixels
    pixel_size = 1   #pixel size in km/s
    K_p = 2*np.pi*(planetData['SMA'] * 1.496e+8)/(planetData['orbPer']*24*60*60)
    #Acceleration at time of transit is almost constant, the maximum of the acceleration sine curve
    acceleration = K_p*2*np.pi/(planetData['orbPer']*24*60*60)
    exposure_time = pixel_size / acceleration
    
    SNR = ComputeSNR(planetData[mag], exposure_time, data_dict['Airm_mid'].min(), SNR_method, mband)
    system_dict['SNR_nosmear'] = SNR
    system_dict['t_nosmear'] = exposure_time
    system_dict['Acc'] = acceleration
    system_dict['K_p'] = K_p
    
    
    #SNR and planet vel shift for fixed exposure time of 5 minutes
    SNR_5min = ComputeSNR(planetData[mag], 5*60, data_dict['Airm_mid'].min(), SNR_method, mband)
    
    system_dict['SNR_5min'] = SNR_5min
    #deltav_5min = acceleration * 5 * 60
    #data_dict['deltav_5min'] = ones_array * deltav_5min
    
    #SNR with exposure time of transit duration
    transit_dur = (data_dict['Trans_end'][0] - data_dict['Trans_start'][0]) * 24 * 60 * 60
    SNR_transdur = ComputeSNR(planetData[mag], transit_dur, data_dict['Airm_mid'].min(), SNR_method, mband)
    system_dict['SNR_transdur'] = SNR_transdur
    
    #scale height
    grav_constant = 6.67430*10**-11
    k_b = 1.380649*10**-23
    m_H = 1.6735575*10**-27 #mass of hydrogen atom [kg]
    surface_gravity = grav_constant * planetData['MpJ'] * 1.898*10**27 / (planetData['RpJ'] * 69911000)**2
    
    if planetData['RpJ']*11.2095 < 1.5:
        #water atmosphere for small planets with R < 1.5 R_earth
        mean_mol_weight = 18
    else:
        #for hot jupiter atmosphere
        mean_mol_weight = 2.3 
    
    Teq = planetData['Teq']
    if np.isnan(planetData['Teq']):
        #if no value for Teq, calculate it using formula from Kempton 2018
        Teq = planetData['Teff'] * np.sqrt(planetData['RsSun']*696340 / (planetData['SMA'] * 1.496e+8)) * 0.25**0.25
    H = k_b * Teq / (mean_mol_weight * m_H * surface_gravity)
    system_dict['Teq'] = Teq
    system_dict['H'] = H
    
    #Delta_d = change in transit depth bc of atmosphere
    Delta_d = 2 * planetData['RpJ'] * 69911000 * H / (planetData['RsSun'] * 696340000)**2
    system_dict['Delta_d'] = Delta_d
    
    #number of transits with good conditions
    moondist_okay_mask = np.array(data_dict['Moon_dist']) > 30
    goodtransit_mask = data_dict['GoodCond'] & moondist_okay_mask
    
    system_dict['N_GoodTransit'] = np.where(goodtransit_mask)[0].size
    
    system_dict['N_GoodTransit_2'] = np.where(goodtransit_mask & data_dict['GoodCond_2'])[0].size
    
    #number of transits required so that a SNR of SNR_desired is achieved
    SNR_desired = 1000
    t_required = (SNR_desired / system_dict['SNR_5min'])**2 * 5 * 60
    N_Trans_SNR1000 = t_required / (system_dict['transit_dur'] * 24 * 3600)
    system_dict['N_Trans_SNR1000'] = N_Trans_SNR1000
    
    #number of transits required for detection (scaled by value of HD 18 bc 1 transit was enough for that target)
    N_trans_req = (1/(SNR_transdur*Delta_d))**2 / 208.264983
    system_dict['N_trans_req'] = N_trans_req
    
    #number of good transits in a certain time interval
    timeinterval_start = d_start
    timeinterval_end = timeinterval_start + dt.timedelta(182)
    
    jd_timeinterval_start = pyasl.jdcnv(timeinterval_start)
    jd_timeinterval_end = pyasl.jdcnv(timeinterval_end)
    
    inInterval = (np.array(data_dict['Tmid']) > jd_timeinterval_start) & (np.array(data_dict['Tmid']) < jd_timeinterval_end)
    system_dict['N_GoodTransit_Semester'] = np.where(goodtransit_mask & inInterval)[0].size
    

    #calculate score to rank the systems 
    system_dict['Systemscore'] = Delta_d *system_dict['N_GoodTransit'] / N_Trans_SNR1000 * 1000
    
    planet_df = pd.DataFrame(data_dict)
    system_df = pd.DataFrame.from_records([system_dict])

    if drop_bad_transit:
        #drop transits with bad conditions
        planet_df = planet_df[planet_df.GoodCond].reset_index().drop('index', axis=1)
    return planet_df, system_df
    
def RequestFromList_Transit(planet_list, catalog, observatory,d_start, d_end,observation_puffer=1/24,
                         min_required_altitude=0, max_airm_good=2, max_sunalt_good=-20,
                            SNR_method='Crires', mband='K', drop_bad_transit=True, verbose=True):
    '''
    Parameters
    ----------
    planet_list : list
        List of all the planet names to be analyzed (as given in the respective Catalog)
    catalog : string, {"Nexa", "TEP", "Custom"}
        Name of the catalog to be used
    observatory : string
        Name of the observatory site as recognized by pyasl.observatory
    d_start : datetime object
        start date of the observation window
    d_end : datetime object
        end date of the observation window 
    observation_puffer : float, optional
        Time before and after transit that is observed (in days), default = 1/24
    min_required_altitude : float, optional
        minimal altitude of the planet during a transit such that the transit is considered as an option (degrees), default = 0
    max_airm_good : float, optional
        maximal airmass considered as 'good' observation conditions, default = 2
    max_sunalt_good : float
        maximal solar altitude considered as 'good' observation conditions, default = -20
    SNR_method : string, optional, {"Crires", "Carmenes"}
        Specifiy the instrument for/ method with which to compute the SNR values
    mband : string, optional, {"K, J"}
        Specifiy which magnitude band to use in the SNR calculation
    drop_bad_transit : bool, optional
        Set to True to only include transits with good obs. conditions in all_tr
    verbose : bool, optional
        Set to True to print status messages
    '''
    
    planet_transit_list = []
    system_list = []
    
    for planet in tqdm(planet_list):
        if verbose:
            print('Finding transits for ' + planet)
        planet_df, system_df = TransitInformation(planet, catalog, observatory, d_start, d_end,observation_puffer,
                             min_required_altitude, max_airm_good, max_sunalt_good, SNR_method, mband, drop_bad_transit, verbose)
        planet_transit_list.append(planet_df)
        system_list.append(system_df)
        
    alltrans = pd.concat(planet_transit_list, ignore_index=True)
    systemdat = pd.concat(system_list, ignore_index=True)
        
    systemdat_ranked = systemdat.sort_values('Systemscore', ascending=False, ignore_index=True)
    
    rounding_dict = {'SNR_nosmear': 0,
                 'Mpl': 3,
                 'Rpl': 3,
                 'Period': 2,
                 't_nosmear': 1,
                 'transit_depth': 3,
                 'transit_dur': 3,
                 'acc': 6,
                 'K_p': 1,
                 'SNR_5min': 0,
                 'SNR_transdur': 0,
                 'H': 0,
                 'Delta_d': 6,
                 'N_Trans_SNR1000': 3,
                 'N_trans_req': 2,
                 'Systemscore': 2,
                 'Sunalt_start': 2,
                 'Sunalt_mid': 2,
                 'Sunalt_end': 2,
                 'Airm_start': 2,
                 'Airm_mid': 2,
                 'Airm_end': 2,
                 'Moon_dist': 2,
                 'V_bary': 2,
                 'Sunalt_start_b': 2,
                 'Sunalt_mid_b': 2,
                 'Sunalt_end_b': 2,
                 'Sunalt_start_a': 2,
                 'Sunalt_mid_a': 2,
                 'Sunalt_end_a': 2,
                 'Airm_start_b': 2,
                 'Airm_mid_b': 2,
                 'Airm_end_b': 2,
                 'Airm_start_a': 2,
                 'Airm_mid_a': 2,
                 'Moon_dist_b': 2,
                 'Moon_dist_a': 2,
                }
        
    return alltrans.round(rounding_dict), systemdat_ranked.round(rounding_dict)
    
    
def TimeAtAltitude(times, altitude, objectData, observatoryData):
    """
    Returns (approximate) time at which an object is located at a certain altitude. 
    For current dates, the accuracy is on the order of 1 minute.
    """
    day2sidereal = 24/24.06570982441908

    h = altitude / 360 * 2 * np.pi
    dec = objectData['dec'] / 360 * 2 * np.pi
    lat = observatoryData['latitude'] / 360 * 2 * np.pi

    cosLHA = (np.sin(h) - np.sin(lat)*np.sin(dec))/(np.cos(lat)*np.cos(dec))
    
    with np.errstate(invalid='raise'):
        try:
            LHA = np.arccos(cosLHA) / 2 / np.pi * 360

        except:
            print('Warning: The object does not reach an altitude of {}'.format(altitude))
            return None, None
        
    loc_sidereal_time_rise = -LHA + objectData['ra']
    loc_sidereal_time_set = LHA + objectData['ra']

    long_hours = observatoryData['longitude'] / 15

    GMST_rise = loc_sidereal_time_rise / 15 - long_hours
    GMST_set = loc_sidereal_time_set / 15 - long_hours

    JD_rise_ref = (GMST_rise - 18.697374558)/24.06570982441908 + 2451545
    JD_set_ref = (GMST_set - 18.697374558)/24.06570982441908 + 2451545

    ind_rise_start = np.floor((np.min(times) - JD_rise_ref)/day2sidereal)
    ind_rise_end = np.ceil((np.max(times) - JD_rise_ref)/day2sidereal)
    ind_set_start = np.floor((np.min(times) - JD_set_ref)/day2sidereal)
    ind_set_end = np.ceil((np.max(times) - JD_set_ref)/day2sidereal)

    all_rise_times = JD_rise_ref + np.arange(ind_rise_start, ind_rise_end+1, 1)*day2sidereal
    all_set_times = JD_set_ref + np.arange(ind_set_start, ind_set_end+1, 1)*day2sidereal

    time_rising = []
    time_setting = []
    for t in times:
        closest_rising = all_rise_times[np.argmin(abs(t - all_rise_times))]
        closest_setting = all_set_times[np.argmin(abs(t - all_set_times))]

        time_rising.extend([closest_rising])
        time_setting.extend([closest_setting])
        
    return np.array(time_rising), np.array(time_setting)
    

def SunsetTime(times, altitude, observatoryData):
    """
    Returns (approximate) time at which the Sun is located at a certain altitude. 
    For current dates, the accuracy is on the order of 1 minute.
    """
    
    location = LocationInfo(latitude=observatoryData['latitude'], longitude=-(360-observatoryData['longitude']))
    
    #The methods dawn and dusk calculate times on the same date as the input time,
    #so the input time has to be adjusted to get start and end of the same night
    
    sunrise_date = []
    for t in times:
        if (t%1) < 0.5:
            t += 1
        s = dawn(location.observer, date=Time(t, format='jd').datetime, depression=-altitude)
        sunrise_date.extend([s])
    sunrise_jd = Time(sunrise_date, format = 'datetime').jd

    sunset_date = []
    for t in times:
        if (t%1) >= 0.5:
            t -= 1
        s = dusk(location.observer, date=Time(t, format='jd').datetime, depression=-altitude)
        sunset_date.extend([s])
    sunset_jd = Time(sunset_date, format = 'datetime').jd
    
    return sunrise_jd, sunset_jd
        
    
    
def EmissionInformation(planet, catalog, obs_time, time_in_eclipse, min_required_altitude,
                       d_start, d_end, max_airm_good, max_sunalt_good, observatory, SNR_method='Crires', mband='K', 
                       drop_bad_eclipse=True, verbose=True):
                             
    '''
    Everything with _b is before eclipse, with _a is after eclipse
    '''
    if catalog == 'TEP':
        physprop, hompar, hommes, obsplan = LoadTEP()
        planetData = GetPlanetDataTEP(planet)
    elif catalog == 'Nexa':
        planetData = GetPlanetDataNexa(planet)
        if planetData == None:
            if verbose:
                print('Try to search in custom catalog.')    
            planetData = GetPlanetDataCustom(planet)
    elif catalog == 'Custom':
        planetData = GetPlanetDataCustom(planet)
    else:
        print('Requested catalog not recognized. Choices are TEP, Nexa, and Custom')
        return None, None
    
    if planetData == None:
        if verbose:
            print(planet + ' not contained in catalog!')        
        return None, None
    
    if not CheckRequiredParameters(planetData, mband):
        if verbose:
            print('At least one of the required parameters is missing!')
        return None, None
    
    jd_start = pyasl.jdcnv(d_start)
    jd_end = pyasl.jdcnv(d_end)
    
    ###
    ###First create dataframe with all observation opportunities
    ###
    t_min = jd_start
    t_max = jd_end

    T0 = planetData['T0']
    T0_err = planetData['T0Err']
    period = planetData['orbPer']
    period_err =  planetData['orbPerErr']
    Tdur = planetData['Tdur']
    ra = planetData['ra']
    dec = planetData['dec']

    observatory_data = pyasl.observatory(observatory)
    lon = observatory_data["longitude"]
    lat = observatory_data["latitude"]
    alt = observatory_data["altitude"]

    trnum_start = np.floor((t_min - T0) / period)
    trnum_end = np.ceil((t_max - T0) / period)
    # Relevant transit epochs
    tr = np.arange(trnum_start, trnum_end, 1)

    #Get list of all relevant times (start, mid and end for both before and after)
    t_list = []
    t_sececl = []  #list of time of sec. eclipse (half a period after transit)
    for epoch in tr:
        Tmid = T0 + float(epoch)*period + period/2
        T_before = Tmid - Tdur/2 + time_in_eclipse - obs_time/2
        T_after = Tmid + Tdur/2 - time_in_eclipse + obs_time/2

        if (Tmid < t_min) or (Tmid > t_max):
            # This may happen because the transit may occur in the first
            # relevant epoch but still before tmin. Likewise for tmax.
            continue

        t_list.extend([T_before - obs_time/2, T_before, T_before + obs_time/2])
        t_list.extend([T_after - obs_time/2, T_after, T_after + obs_time/2])
        t_sececl.extend([Tmid, Tmid])

    t_list = np.array(t_list)  
    t_sececl = np.array(t_sececl)
    
    if len(t_list) == 0:
        #no sec. eclipse has occured since T0
        if verbose:
            print('No secondary eclipse will occure in the time frame')
        return None, None
    #correct for light travel time to barycentre
    coos = coord.SkyCoord(ra, dec, unit=u.deg)
    loc = coord.EarthLocation(lon=lon, lat=lat, height=alt)

    t_list_timeobject = Time(np.append(t_list, t_sececl), format='jd')
    ltt_bary = t_list_timeobject.light_travel_time(coos, location = loc)
    #subtract ltt_bary to go from barycentric to Earth frame
    t_list_corrected = (t_list_timeobject - ltt_bary)

    t_list = t_list_corrected.value[:len(t_list)]
    t_sececl = t_list_corrected.value[len(t_list):]
    
    altaz = pyasl.eq2hor(t_list, np.ones(t_list.shape) * ra,np.ones(t_list.shape) * dec, lon=lon, lat=lat, alt=alt)
    altitude = altaz[0]

    #remove when planet not visible from observation site
    alt_filtered = []
    t_filtered = []
    t_sececl_filtered = []

    for i in range(int(altitude.size / 3)):
        minalt = np.where(altitude[i*3:i*3+3] >= min_required_altitude)[0]

        if (len(minalt)<3):
            #target not visible at all -> skip
            continue
            
        alt_filtered.extend(altitude[i*3:i*3+3])
        t_filtered.extend(t_list[i*3:i*3+3])
        t_sececl_filtered.extend([t_sececl[i]])

    alt_filtered = np.array(alt_filtered)
    t_filtered = np.array(t_filtered)
    t_sececl_filtered = np.array(t_sececl_filtered)

    total_nights = int(t_sececl_filtered.size)

    if total_nights == 0:
        if verbose:
            print('No eclipse opportunity found for ' + planet)
        
        return None, None
    
    #calculate sun altitude and airmass at each time
    sunpos_radec = pyasl.sunpos(t_filtered)
    sunpos_altaz = pyasl.eq2hor(t_filtered, sunpos_radec[1][0],
                                 sunpos_radec[2][0],
                                 lon=lon, lat=lat, alt=alt)
    sunalt = np.ones(t_filtered.shape)*np.nan
    sunalt = sunpos_altaz[0]

    airm = pyasl.airmassPP(90-alt_filtered)

    #calculate distance to moon (only for midpoints before and after)
    moondist = np.ones(t_sececl_filtered.size)*np.nan
    midpoints = np.array([t_filtered[i*3+1] for i in range(int(t_filtered.size / 3))])

    mpos = pyasl.moonpos(midpoints[pd.notnull(midpoints)])
    moondist[pd.notnull(midpoints)] = pyasl.getAngDist(mpos[0], mpos[1], ra, dec)

    ##Get the date of each night
    t_jd = np.array([t_filtered[i*3+1] for i in range(total_nights)])

    #rough determination of timezone
    timezone = ((int(lon/15)+12) % 24) - 12

    t_jd += timezone/24

    #Subtract 1 day if night starts at day before (if time is before 8am)
    t_jd[t_jd % 1 > 0.5] -= 1
    dates_complete = Time(t_jd, format='jd').iso
    night = [date[:10] for date in dates_complete]

    #Calculate error of transit midpoint time
    n_transits_since_T0 = np.round([(t_filtered[i*3+1] - T0)/period for i in range(total_nights)])

    Tmid_err = np.sqrt(T0_err**2 + (n_transits_since_T0 * period_err)**2)
    
    data_dict = {'System': [planet for i in range(total_nights)],
                   'Night': night,
                   'T_sececl': t_sececl_filtered,
                   'Tmid': [t_filtered[i*3+1] for i in range(total_nights)],
                   'Tmid_err': Tmid_err,
                   'Obs_start': [t_filtered[i*3] for i in range(total_nights)],
                   'Obs_end': [t_filtered[i*3+2] for i in range(total_nights)],
                   'Sunalt_start': [sunalt[i*3] for i in range(total_nights)],
                   'Sunalt_mid': [sunalt[i*3+1] for i in range(total_nights)],
                   'Sunalt_end': [sunalt[i*3+2] for i in range(total_nights)],
                   'Airm_start': [airm[i*3] for i in range(total_nights)],
                   'Airm_mid': [airm[i*3+1] for i in range(total_nights)],
                   'Airm_end': [airm[i*3+2] for i in range(total_nights)],
                   'Moon_dist': moondist,
                  }        

    #Indicate wheter the opportunity is before or after sec. eclipse
    timing = ['Before']*t_sececl_filtered.size
    for i in range(len(timing)):
        if t_sececl_filtered[i] < data_dict['Tmid'][i]:
            timing[i] = 'After'
    data_dict['Timing'] = timing
            
    data_dict['GoodCond'] = ((np.array(data_dict['Airm_start']) < max_airm_good) & 
                               (np.array(data_dict['Sunalt_start']) < max_sunalt_good) &
                               (np.array(data_dict['Airm_end']) < max_airm_good) &
                               (np.array(data_dict['Sunalt_end']) < max_sunalt_good))
                               
    #Calculate observable time before and after opportunity (only for good opps)
    altitudeFromAirmass = 90 - np.arccos(1/max_airm_good)*180/np.pi
    
    times = np.array(data_dict['Tmid'])[data_dict['GoodCond']]
    
    goodCond_before = np.empty(len(data_dict['Tmid']))*np.nan
    goodCond_after = np.empty(len(data_dict['Tmid']))*np.nan
    
    if times.size > 0:
        rising, setting = TimeAtAltitude(times, altitudeFromAirmass, planetData, observatory_data)
        sunrise, sunset = SunsetTime(times, max_sunalt_good, observatory_data)

        goodCond_start = np.copy(rising)
        goodCond_end = np.copy(setting)

        goodCond_start[(rising < sunset)] = sunset[(rising < sunset)]
        goodCond_end[(setting > sunrise)] = sunrise[(setting > sunrise)]

        goodCond_before[data_dict['GoodCond']] = np.array(data_dict['Tmid'])[data_dict['GoodCond']] - goodCond_start
        goodCond_after[data_dict['GoodCond']] = goodCond_end - np.array(data_dict['Tmid'])[data_dict['GoodCond']]
        
    data_dict['GoodCond_before'] = goodCond_before * 24
    data_dict['GoodCond_after'] = goodCond_after * 24                           
    
    #Total time in good condition outside of the eclipse
    data_dict['GoodCond_outOfEclipse'] = data_dict['GoodCond_before'] + obs_time / 2 - time_in_eclipse
    
    obs_df = pd.DataFrame(data_dict)
    
    ###
    ###Create df with system data that is the same for all transits of a planet
    ###
    system_dict = {}

    mag = mband + 'mag'
    #system name
    system_dict['System'] = planet
    system_dict['Mstar'] = planetData['MsSun']
    system_dict['Rstar'] = planetData['RsSun']
    system_dict['Mpl'] = planetData['MpJ']
    system_dict['Rpl'] = planetData['RpJ']
    system_dict['Period'] = planetData['orbPer']
    system_dict['Ecc'] = planetData['orbEcc']
    system_dict['Teff'] = planetData['Teff']
    system_dict[mag] = planetData[mag]
    
    system_dict['transit_dur'] = planetData['Tdur']
    system_dict['Tmid_err'] = np.mean(Tmid_err)

    #stellar SNR
    #here units are km and s everywhere
    #first calculate maximum exposure so that lines do not shift between pixels
    pixel_size = 1   #pixel size in km/s
    K_p = 2*np.pi*(planetData['SMA'] * 1.496e+8)/(planetData['orbPer']*24*60*60)
    #Acceleration at time of transit is almost constant, the maximum of the acceleration sine curve
    acceleration = K_p*2*np.pi/(planetData['orbPer']*24*60*60)
    exposure_time = pixel_size / acceleration
    
    SNR = ComputeSNR(planetData[mag], exposure_time, np.nanmin(data_dict['Airm_mid']), SNR_method, mband)
    system_dict['SNR_nosmear'] = SNR
    system_dict['t_nosmear'] = exposure_time
    system_dict['acc'] = acceleration
    system_dict['K_p'] = K_p
    
    
    #SNR for fixed exposure time of 5 minutes
    SNR_5min = ComputeSNR(planetData[mag], 5*60, np.nanmin(data_dict['Airm_mid']), SNR_method, mband)
    system_dict['SNR_5min'] = SNR_5min
    
    #scale height
    grav_constant = 6.67430*10**-11
    k_b = 1.380649*10**-23
    m_H = 1.6735575*10**-27 #mass of hydrogen atom [kg]
    surface_gravity = grav_constant * planetData['MpJ'] * 1.898*10**27 / (planetData['RpJ'] * 69911000)**2
    mean_mol_weight = 2.3 #for hot jupiter atmosphere
    
    Teq = planetData['Teq']
    if np.isnan(planetData['Teq']):
        #if no value for Teq, calculate it using formula from Kempton 2018
        Teq = planetData['Teff'] * np.sqrt(planetData['RsSun']*696340 / (planetData['SMA'] * 1.496e+8)) * 0.25**0.25
    H = k_b * Teq / (mean_mol_weight * m_H * surface_gravity)
    system_dict['Teq'] = Teq
    system_dict['Teq_Calc'] = not np.isnan(planetData['Teq']) #True if Teq is calculated here
    system_dict['H'] = H
    
    #number of transits with good conditions
    moondist_okay_mask = np.array(data_dict['Moon_dist']) > 30
    #Removed moon dist chekc for now (01.02.2021)
    goodtransit_mask = (data_dict['GoodCond']==1)
    
    system_dict['N_Good'] = np.where(goodtransit_mask)[0].size
    
    #calculate score to rank the systems 
    #reference wavelength to calculate ratio of planck spectra
    if mband == 'K':
        lambda_ref = 2.2 * 10**-6
    elif mband == 'J':
        lambda_ref = 1.25 * 10**-6
    else:
        lambda_ref = 2.2 * 10**-6
    planet_signal = pyasl.planck(Teq, lam=lambda_ref) * (system_dict['Rpl'] * 69911000)**2
    star_signal = pyasl.planck(system_dict['Teff'], lam=lambda_ref) * (system_dict['Rstar'] * 696340000)**2
    signal_strength = planet_signal / star_signal
    
    system_dict['Signal_strength'] = signal_strength
    
    scoreNumberofNights = 1
    if (system_dict['N_Good'] == 0):
        scoreNumberofNights = 0
    elif (system_dict['N_Good'] > 3):
        scoreNumberofNights = 1.5
        
    score = SNR_5min * scoreNumberofNights * signal_strength*100
    system_dict['Systemscore'] = score
    
    system_df = pd.DataFrame.from_records([system_dict])
    
    if drop_bad_eclipse:
        #drop transits with bad conditions
        obs_df = obs_df[goodtransit_mask].reset_index().drop('index', axis=1)
    return obs_df, system_df
    
    
def RequestFromList_Emission(planet_list, catalog, observatory, d_start, d_end, obs_time, time_in_eclipse,
                         min_required_altitude=0, max_airm_good=2, max_sunalt_good=-20,
                            SNR_method='Crires', mband='K', drop_bad_eclipse=True, verbose=True):
    
    
    planet_event_list = []
    system_list = []

    for planet in tqdm(planet_list):
        if verbose:
            print('Finding emission observation opportunities for ' + planet)
        planet_df, system_df = EmissionInformation(planet, catalog, obs_time, time_in_eclipse, min_required_altitude,
                                                  d_start, d_end, max_airm_good, max_sunalt_good, observatory, 
                                               SNR_method, mband, drop_bad_eclipse, verbose)
        planet_event_list.append(planet_df)
        system_list.append(system_df)
        
    allevents = pd.concat(planet_event_list, ignore_index=True)
    systemdat = pd.concat(system_list, ignore_index=True)
        
    systemdat_ranked = systemdat.sort_values('Systemscore', ascending=False, ignore_index=True)
    
    rounding_dict = {'SNR_nosmear': 0,
                 'Mpl': 3,
                 'Rpl': 3,
                 'Period': 2,
                 't_nosmear': 1,
                 'transit_depth': 3,
                 'transit_dur': 3,
                 'acc': 6,
                 'K_p': 1,
                 'SNR_5min': 0,
                 'SNR_transdur': 0,
                 'H': 0,
                 'Delta_d': 6,
                 'N_Trans_SNR1000': 3,
                 'N_trans_req': 2,
                 'systemscore': 2,
                 'Sunalt_start': 2,
                 'Sunalt_mid': 2,
                 'Sunalt_end': 2,
                 'Airm_start': 2,
                 'Airm_mid': 2,
                 'Airm_end': 2,
                 'Moon_dist': 2,
                 'V_bary': 2,
                 'GoodCond_before': 2,
                 'GoodCond_after': 2,
                }
        
    allevents = allevents.round(rounding_dict)
    
    return allevents, systemdat_ranked.round(rounding_dict)
    
def RequestFromList_EmissionOld(planet_list, catalog, observatory, d_start, d_end, obs_time, time_in_eclipse,
                         min_required_altitude=0, max_airm_good=2, max_sunalt_good=-20,
                            SNR_method='Crires', mband='K', drop_bad_eclipse=True, verbose=True):
    
    
    planet_event_list = []
    system_list = []

    for planet in tqdm(planet_list):
        if verbose:
            print('Finding emission observation opportunities for ' + planet)
        planet_df, system_df = EmissionInformationOld(planet, catalog, obs_time, time_in_eclipse, min_required_altitude,
                                                  d_start, d_end, max_airm_good, max_sunalt_good, observatory, 
                                               SNR_method, mband, drop_bad_eclipse, verbose)
        planet_event_list.append(planet_df)
        system_list.append(system_df)
        
    allevents = pd.concat(planet_event_list, ignore_index=True)
    systemdat = pd.concat(system_list, ignore_index=True)
        
    systemdat_ranked = systemdat.sort_values('Systemscore', ascending=False, ignore_index=True)
    
    rounding_dict = {'SNR_nosmear': 0,
                 'Mpl': 3,
                 'Rpl': 3,
                 'Period': 2,
                 't_nosmear': 1,
                 'transit_depth': 3,
                 'transit_dur': 3,
                 'acc': 6,
                 'K_p': 1,
                 'SNR_5min': 0,
                 'SNR_transdur': 0,
                 'H': 0,
                 'Delta_d': 6,
                 'N_Trans_SNR1000': 3,
                 'N_trans_req': 2,
                 'systemscore': 2,
                 'Sunalt_start': 2,
                 'Sunalt_mid': 2,
                 'Sunalt_end': 2,
                 'Airm_start': 2,
                 'Airm_mid': 2,
                 'Airm_end': 2,
                 'Moon_dist': 2,
                 'V_bary': 2,
                 'Sunalt_start_b': 2,
                 'Sunalt_mid_b': 2,
                 'Sunalt_end_b': 2,
                 'Sunalt_start_a': 2,
                 'Sunalt_mid_a': 2,
                 'Sunalt_end_a': 2,
                 'Airm_start_b': 2,
                 'Airm_mid_b': 2,
                 'Airm_end_b': 2,
                 'Airm_start_a': 2,
                 'Airm_mid_a': 2,
                 'Moon_dist_b': 2,
                 'Moon_dist_a': 2,
                }
        
    allevents = allevents.round(rounding_dict)
    allevents = ConvertEmList(allevents)
    
    return allevents, systemdat_ranked.round(rounding_dict)
    
def EmissionInformationOld(planet, catalog, obs_time, time_in_eclipse, min_required_altitude,
                       d_start, d_end, max_airm_good, max_sunalt_good, observatory, SNR_method='Crires', mband='K', 
                       drop_bad_eclipse=True, verbose=True):
                             
    '''
    Everything with _b is before eclipse, with _a is after eclipse
    '''
    if catalog == 'TEP':
        physprop, hompar, hommes, obsplan = LoadTEP()
        planetData = GetPlanetDataTEP(planet)
    elif catalog == 'Nexa':
        planetData = GetPlanetDataNexa(planet)
        if planetData == None:
            if verbose:
                print('Try to search in custom catalog.')    
            planetData = GetPlanetDataCustom(planet)
    elif catalog == 'Custom':
        planetData = GetPlanetDataCustom(planet)
    else:
        print('Requested catalog not recognized. Choices are TEP, Nexa, and Custom')
        return None, None
    
    if planetData == None:
        if verbose:
            print(planet + ' not contained in catalog!')        
        return None, None
    
    if not CheckRequiredParameters(planetData):
        if verbose:
            print('At least one of the required parameters is missing!')
        return None, None
    
    jd_start = pyasl.jdcnv(d_start)
    jd_end = pyasl.jdcnv(d_end)
    
    ###
    ###First create dataframe with all observation opportunities
    ###
    t_min = jd_start
    t_max = jd_end

    T0 = planetData['T0']
    period = planetData['orbPer']
    Tdur = planetData['Tdur']
    ra = planetData['ra']
    dec = planetData['dec']

    observatory_data = pyasl.observatory(observatory)
    lon = observatory_data["longitude"]
    lat = observatory_data["latitude"]
    alt = observatory_data["altitude"]

    trnum_start = np.floor((t_min - T0) / period)
    trnum_end = np.ceil((t_max - T0) / period)
    # Relevant transit epochs
    tr = np.arange(trnum_start, trnum_end, 1)

    #Get list of all relevant times (start, mid and end for both before and after)
    t_list = []
    for epoch in tr:
        Tmid = T0 + float(epoch)*period + period/2
        T_before = Tmid - Tdur/2 + time_in_eclipse - obs_time/2
        T_after = Tmid + Tdur/2 - time_in_eclipse + obs_time/2

        if (Tmid < t_min) or (Tmid > t_max):
            # This may happen because the transit may occur in the first
            # relevant epoch but still before tmin. Likewise for tmax.
            continue

        t_list.extend([T_before - obs_time/2, T_before, T_before + obs_time/2,
                       T_after - obs_time/2, T_after, T_after + obs_time/2])

    t_list = np.array(t_list)  

    if len(t_list) == 0:
        #no sec. eclipse has occured since T0
        if verbose:
            print('No secondary eclipse will occure in the time frame')
        return None, None
    #correct for light travel time to barycentre
    coos = coord.SkyCoord(ra, dec, unit=u.deg)
    loc = coord.EarthLocation(lon=lon, lat=lat, height=alt)

    t_list_timeobject = Time(t_list, format='jd')
    ltt_bary = t_list_timeobject.light_travel_time(coos, location = loc)
    t_list_corrected = (t_list_timeobject - ltt_bary)
    #subtract ltt_bary to go from barycentric to Earth frame
    t_list = t_list_corrected.value
    
    altaz = pyasl.eq2hor(t_list, np.ones(t_list.shape) * ra,np.ones(t_list.shape) * dec, lon=lon, lat=lat, alt=alt)
    altitude = altaz[0]

    #remove when planet not visible from observation site
    alt_filtered = []
    t_filtered = []

    nan_list = [np.nan, np.nan, np.nan]
    for i in range(int(altitude.size / 6)):
        minalt_before = np.where(altitude[i*6:i*6+3] >= min_required_altitude)[0]
        minalt_after = np.where(altitude[i*6+3:i*6+6] >= min_required_altitude)[0]

        if (len(minalt_before)<3) & (len(minalt_after)<3):
            #target not visible at all -> skip
            continue
        if len(minalt_before)<3:
            alt_filtered.extend(nan_list)
            t_filtered.extend(nan_list)
        else:
            alt_filtered.extend(altitude[i*6:i*6+3])
            t_filtered.extend(t_list[i*6:i*6+3])
        if len(minalt_after)<3:
            alt_filtered.extend(nan_list)
            t_filtered.extend(nan_list)
        else:
            alt_filtered.extend(altitude[i*6+3:i*6+6])
            t_filtered.extend(t_list[i*6+3:i*6+6])
    alt_filtered = np.array(alt_filtered)
    t_filtered = np.array(t_filtered)

    total_nights = int(t_filtered.size / 6)


    if total_nights == 0:
        if verbose:
            print('No eclipse opportunity found for ' + planet)
        
        return None, None
    
    #calculate sun altitude and airmass at each time
    notnan = pd.notnull(t_filtered)
    sunpos_radec = pyasl.sunpos(t_filtered[notnan])
    sunpos_altaz = pyasl.eq2hor(t_filtered[notnan], sunpos_radec[1][0],
                                 sunpos_radec[2][0],
                                 lon=lon, lat=lat, alt=alt)
    sunalt = np.ones(t_filtered.shape)*np.nan
    sunalt[notnan] = sunpos_altaz[0]

    airm = pyasl.airmassPP(90-alt_filtered)

    #calculate distance to moon (only for midpoints before and after)
    moondist = np.ones(int(t_filtered.size / 3))*np.nan
    midpoints = np.array([t_filtered[i*3+1] for i in range(int(t_filtered.size / 3))])

    mpos = pyasl.moonpos(midpoints[pd.notnull(midpoints)])
    moondist[pd.notnull(midpoints)] = pyasl.getAngDist(mpos[0], mpos[1], ra, dec)

    ##Get the date of each night
    t_jd_b = np.array([t_filtered[i*6+1] for i in range(total_nights)])
    t_jd_a = np.array([t_filtered[i*6+4] for i in range(total_nights)])

    #rough determination of timezone
    timezone = ((int(lon/15)+12) % 24) - 12

    t_jd_b += timezone/24
    t_jd_a += timezone/24

    t_notnan_b = t_jd_b[pd.notnull(t_jd_b)]
    t_notnan_a = t_jd_a[pd.notnull(t_jd_a)]

    #Subtract 1 day if night starts at day before (if time is before 8am)
    t_notnan_b[t_notnan_b % 1 > 0.5] -= 1
    dates_complete_b = Time(t_notnan_b, format='jd').iso
    dates_b = [date[:10] for date in dates_complete_b]

    t_notnan_a[t_notnan_a % 1 > 0.5] -= 1
    dates_complete_a = Time(t_notnan_a, format='jd').iso
    dates_a = [date[:10] for date in dates_complete_a]

    night = [np.nan for i in range(total_nights)]
    #use night of b if available, because that is earlier than a
    for i in range(len(dates_a)):
        night_ind = np.where(pd.notnull(t_jd_a))[0][i]
        night[night_ind] = dates_a[i]
    for i in range(len(dates_b)):
        night_ind = np.where(pd.notnull(t_jd_b))[0][i]
        night[night_ind] = dates_b[i]



    data_dict = {'System': [planet for i in range(total_nights)],
                   'Night': night,
                   'Tmid_b': [t_filtered[i*6+1] for i in range(total_nights)],
                   'Tmid_a': [t_filtered[i*6+4] for i in range(total_nights)],
                   'Obs_start_b': [t_filtered[i*6] for i in range(total_nights)],
                   'Obs_end_b': [t_filtered[i*6+2] for i in range(total_nights)],
                   'Obs_start_a': [t_filtered[i*6+3] for i in range(total_nights)],
                   'Obs_end_a': [t_filtered[i*6+5] for i in range(total_nights)],
                   'Sunalt_start_b': [sunalt[i*6] for i in range(total_nights)],
                   'Sunalt_mid_b': [sunalt[i*6+1] for i in range(total_nights)],
                   'Sunalt_end_b': [sunalt[i*6+2] for i in range(total_nights)],
                   'Sunalt_start_a': [sunalt[i*6+3] for i in range(total_nights)],
                   'Sunalt_mid_a': [sunalt[i*6+4] for i in range(total_nights)],
                   'Sunalt_end_a': [sunalt[i*6+5] for i in range(total_nights)],
                   'Airm_start_b': [airm[i*6] for i in range(total_nights)],
                   'Airm_mid_b': [airm[i*6+1] for i in range(total_nights)],
                   'Airm_end_b': [airm[i*6+2] for i in range(total_nights)],
                   'Airm_start_a': [airm[i*6+3] for i in range(total_nights)],
                   'Airm_mid_a': [airm[i*6+4] for i in range(total_nights)],
                   'Airm_end_a': [airm[i*6+5] for i in range(total_nights)],
                   'Moon_dist_b': [moondist[i*2] for i in range(int(moondist.size / 2))],
                   'Moon_dist_a': [moondist[i*2+1] for i in range(int(moondist.size / 2))],    
                  }        

    data_dict['GoodCond_b'] = ((np.array(data_dict['Airm_start_b']) < max_airm_good) & 
                               (np.array(data_dict['Sunalt_start_b']) < max_sunalt_good) &
                               (np.array(data_dict['Airm_end_b']) < max_airm_good) &
                               (np.array(data_dict['Sunalt_end_b']) < max_sunalt_good))

    data_dict['GoodCond_a'] = ((np.array(data_dict['Airm_start_a']) < max_airm_good) & 
                               (np.array(data_dict['Sunalt_start_a']) < max_sunalt_good) &
                               (np.array(data_dict['Airm_end_a']) < max_airm_good) &
                               (np.array(data_dict['Sunalt_end_a']) < max_sunalt_good))


    obs_df = pd.DataFrame(data_dict)
    
    ###
    ###Create df with system data that is the same for all transits of a planet
    ###
    system_dict = {}

    mag = mband + 'mag'
    #system name
    system_dict['System'] = planet
    system_dict['Mstar'] = planetData['MsSun']
    system_dict['Rstar'] = planetData['RsSun']
    system_dict['Mpl'] = planetData['MpJ']
    system_dict['Rpl'] = planetData['RpJ']
    system_dict['Period'] = planetData['orbPer']
    system_dict['Ecc'] = planetData['orbEcc']
    system_dict['Teff'] = planetData['Teff']
    system_dict[mag] = planetData[mag]
    
    system_dict['transit_dur'] = planetData['Tdur']

    
    #stellar SNR
    #here units are km and s everywhere
    #first calculate maximum exposure so that lines do not shift between pixels
    pixel_size = 1   #pixel size in km/s
    K_p = 2*np.pi*(planetData['SMA'] * 1.496e+8)/(planetData['orbPer']*24*60*60)
    #Acceleration at time of transit is almost constant, the maximum of the acceleration sine curve
    acceleration = K_p*2*np.pi/(planetData['orbPer']*24*60*60)
    exposure_time = pixel_size / acceleration
    
    SNR = ComputeSNR(planetData[mag], exposure_time, np.nanmin(data_dict['Airm_mid_b']), SNR_method, mband)
    system_dict['SNR_nosmear'] = SNR
    system_dict['t_nosmear'] = exposure_time
    system_dict['acc'] = acceleration
    system_dict['K_p'] = K_p
    
    
    #SNR for fixed exposure time of 5 minutes
    SNR_5min = ComputeSNR(planetData[mag], 5*60, np.nanmin(data_dict['Airm_mid_b']), SNR_method, mband)
    system_dict['SNR_5min'] = SNR_5min
    
    #scale height
    grav_constant = 6.67430*10**-11
    k_b = 1.380649*10**-23
    m_H = 1.6735575*10**-27 #mass of hydrogen atom [kg]
    surface_gravity = grav_constant * planetData['MpJ'] * 1.898*10**27 / (planetData['RpJ'] * 69911000)**2
    mean_mol_weight = 2.3 #for hot jupiter atmosphere
    
    Teq = planetData['Teq']
    if np.isnan(planetData['Teq']):
        #if no value for Teq, calculate it using formula from Kempton 2018
        Teq = planetData['Teff'] * np.sqrt(planetData['RsSun']*696340 / (planetData['SMA'] * 1.496e+8)) * 0.25**0.25
    H = k_b * Teq / (mean_mol_weight * m_H * surface_gravity)
    system_dict['Teq'] = Teq
    system_dict['H'] = H
    
    #number of transits with good conditions
    moondist_okay_mask_b = np.array(data_dict['Moon_dist_b']) > 30
    goodtransit_mask_b = (data_dict['GoodCond_b']==1) & moondist_okay_mask_b
    moondist_okay_mask_a = np.array(data_dict['Moon_dist_a']) > 30
    goodtransit_mask_a = (data_dict['GoodCond_a']==1) & moondist_okay_mask_a
    
    system_dict['N_GoodBefore'] = np.where(goodtransit_mask_b)[0].size
    system_dict['N_GoodAfter'] = np.where(goodtransit_mask_a)[0].size
    system_dict['N_GoodBoth'] = np.where(goodtransit_mask_b & goodtransit_mask_a)[0].size
    
    #calculate score to rank the systems 
    #reference wavelength to calculate ratio of planck spectra
    if mband == 'K':
        lambda_ref = 2.2 * 10**-6
    elif mband == 'J':
        lambda_ref = 1.25 * 10**-6
    else:
        lambda_ref = 2.2 * 10**-6
    planet_signal = pyasl.planck(Teq, lam=lambda_ref) * (system_dict['Rpl'] * 69911000)**2
    star_signal = pyasl.planck(system_dict['Teff'], lam=lambda_ref) * (system_dict['Rstar'] * 696340000)**2
    signal_strength = planet_signal / star_signal
    
    scoreNumberofNights = 1
    if ((system_dict['N_GoodBefore'] == 0) & (system_dict['N_GoodAfter'] == 0)):
        scoreNumberofNights = 0
    elif ((system_dict['N_GoodBefore'] > 3) & (system_dict['N_GoodAfter'] > 3)):
        scoreNumberofNights = 1.5
        
    score = SNR_5min * scoreNumberofNights * signal_strength*100
    system_dict['Systemscore'] = score
    
    system_df = pd.DataFrame.from_records([system_dict])
    
    if drop_bad_eclipse:
        #drop transits with bad conditions
        obs_df = obs_df[goodtransit_mask_b | goodtransit_mask_a].reset_index().drop('index', axis=1)
    return obs_df, system_df
    
def ConvertEmList(df):    
    '''
    Converts the all_em list containing single rows combining before and after eclipse into new dataframe
    where each event has ist own row
    '''

    System = []
    Night = []
    Timing = []
    Tmid = []
    Obs_start = []
    Obs_end = []
    Sunalt_start = []
    Sunalt_mid = []
    Sunalt_end = []
    Airm_start = []
    Airm_mid = []
    Airm_end = []
    Moon_dist = []
    GoodCond = []

    for i in range(len(df)):
        try:
            if not pd.isnull(df.Tmid_b[i]):
                #Add entry for opportunity before eclipse
                System.append(df.System[i])
                Night.append(df.Night[i])
                Timing.append('Before')
                Tmid.append(df.Tmid_b[i])
                Obs_start.append(df.Obs_start_b[i])
                Obs_end.append(df.Obs_end_b[i])
                Sunalt_start.append(df.Sunalt_start_b[i])
                Sunalt_mid.append(df.Sunalt_mid_b[i])
                Sunalt_end.append(df.Sunalt_end_b[i])
                Airm_start.append(df.Airm_start_b[i])
                Airm_mid.append(df.Airm_mid_b[i])
                Airm_end.append(df.Airm_end_b[i])
                Moon_dist.append(df.Moon_dist_b[i])
                GoodCond.append(df.GoodCond_b[i])

            if not pd.isnull(df.Tmid_a[i]):
                #Add entry for opportunity after eclipse
                System.append(df.System[i])
                Night.append(df.Night[i])
                Timing.append('After')
                Tmid.append(df.Tmid_a[i])
                Obs_start.append(df.Obs_start_a[i])
                Obs_end.append(df.Obs_end_a[i])
                Sunalt_start.append(df.Sunalt_start_a[i])
                Sunalt_mid.append(df.Sunalt_mid_a[i])
                Sunalt_end.append(df.Sunalt_end_a[i])
                Airm_start.append(df.Airm_start_a[i])
                Airm_mid.append(df.Airm_mid_a[i])
                Airm_end.append(df.Airm_end_a[i])
                Moon_dist.append(df.Moon_dist_a[i])
                GoodCond.append(df.GoodCond_a[i])
        except:
            print('Dataframe has not all required columns')

    new_dict = {'System': System,
                'Night': Night,
                'Timing': Timing,
                'Tmid': Tmid,
                'Obs_start': Obs_start,
                'Obs_end': Obs_end,
                'Sunalt_start': Sunalt_start,
                'Sunalt_mid': Sunalt_mid,     
                'Sunalt_end': Sunalt_end,
                'Airm_start': Airm_start,
                'Airm_mid': Airm_mid,     
                'Airm_end': Airm_end,
                'Moon_dist': Moon_dist,
                'GoodCond': GoodCond
               }
    new_df = pd.DataFrame(new_dict)

    return new_df
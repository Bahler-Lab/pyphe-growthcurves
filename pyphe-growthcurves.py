import argparse
import pandas as pd
import numpy as np
from scipy import stats


def analyse_growthcurve(gdata, fitrange, t0_fitrange, lag_method, lag_threshold, plots, plot_ylim):
    
    t = gdata.index.tolist()
    
    def find_lag(x):
        '''Find the lag phase for a single growthcurve.
        Required arguments:
        x (array-like) -- 1D array-like containing the population/colony sizes
        t (array-like) -- 1D array-like containing the timepoints, must have same dimensions as x
        t0 (float) -- Inoculation biomass
        method (str) -- method to use to determine lag phase. Currently supported: rel_threshold and abs_threshold. Tobi method waits to be implemented
        thresh (float) -- The threshold value to use. The lag phase will be determined as the time it takes for the biomass to exceed this value (for abs_threshold) or t0*threshold for rel_threshold.
        Returns:
        lag (float) -- lag time
        '''
        t0 = np.array(x)[0:t0_fitrange].mean()

        if lag_method=='rel':
            for i, val in enumerate(x):
                if val > lag_threshold*t0:
                    return pd.Series([t0, t[i]], index=['initial biomass', 'lag'])

        elif lag_method=='abs':
            for i, val in enumerate(x):
                if val > lag_threshold:
                    return pd.Series([t0, t[i]], index=['initial biomass', 'lag'])              

        else:
            raise Exception('Unknown lag method %s' %method)

    #Analyse lags
    lags = gdata.apply(find_lag)
    
    def find_max_slope(x):
        '''Find max_slope, t_max, intercept and r2 for a single growthcurve. The regression is aware of the timepoints so this will work with unevenly samples growthcurves.
        Required arguments:
        x (array-like) -- 1D array-like containing the population/colony sizes
        t (array-like) -- 1D array-like containing the timepoints, must have same dimensions as x
        reg_fitrange (int) -- The number of timepoints over which to fit the linear regression
        Returns:
        {
        max_slope -- The slope of the regression
        t_max -- The mid-point of the fitrange of the regression
        intercept -- The y-inyercept of the regression
        r2 -- The R^2 value of the regression
        }
        '''
        regression_results = []
        x = x.tolist()
        
        for i in range(len(x)-fitrange):
            slope, intercept, r_value, p_value, std_err = stats.linregress(t[i:i+fitrange], x[i:i+fitrange])
            regression_results.append({'t_max':np.mean(t[i:i+fitrange]), 'max_slope':slope, 'r2':r_value**2, 'y-intercept':intercept})
        slope_result = pd.Series(max(regression_results, key=lambda x: x['max_slope']))
        slope_result['x-intercept'] = -slope_result['y-intercept']/slope_result['max_slope']
        return slope_result
        
        
    slopes = gdata.apply(find_max_slope)
    
    
    if plots:
        from matplotlib import pyplot as plt
        plt.rcParams['svg.fonttype'] = 'none'
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages('.'.join(args.input.split('.')[:-1]) + '_curves.pdf') as pdf:
            layout=(8,4)
            raw_kwargs={'color':'C0', 'linewidth':1}
            smoothed_kwargs={'color':'r', 'linewidth':0.5}
            regr_kwargs={'color':'k', 'linewidth':0.5, 'linestyle':'--'}
            
            toPlot = list(gdata)
            while toPlot:
                fig, ax = plt.subplots(layout[0], layout[1], figsize=(8.27,11.69))
                for a in ax.flat:
                    
                    a.plot(t, gdata[toPlot[0]], **raw_kwargs)
                    #Get ylim
                    ylim = a.get_ylim()

                    tmax = slopes.loc['t_max', toPlot[0]]
                    maxslope = slopes.loc['max_slope', toPlot[0]] 
                    intercept = slopes.loc['y-intercept', toPlot[0]]
                    if not pd.isnull([tmax, maxslope, intercept]).any():
                        x = np.array(t)
                        y = x*maxslope + intercept
                        a.plot(x, y, **regr_kwargs)

                    t0 = lags.loc['initial biomass', toPlot[0]]
                    lag = lags.loc['lag', toPlot[0]]
                    if not pd.isnull([t0, lag]).any():
                        a.axhline(t0, color='k', xmin=0, xmax=lag, linewidth=0.75, alpha=0.6)
                        a.axvline(lag, color='k', linewidth=0.75, alpha=0.6)
                        if 'lag_method' == 'abs':
                            a.axhline(lag_threshold, color='k', xmin=0, xmax=lag, linewidth=0.75, alpha=0.6)
                        else:
                            a.axhline(lag_threshold * t0, color='k', xmin=0, xmax=lag, linewidth=0.75, alpha=0.6)
                        
                    a.set_title(str(toPlot[0]))
                    if plot_ylim:
                        a.set_ylim([0,plot_ylim])
                    else: 
                        a.set_ylim(ylim)
                    toPlot.pop(0)
                    if not toPlot:
                        break
                        
                plt.tight_layout()                    
                pdf.savefig()
                plt.close()
                plt.clf()

    return pd.concat([lags, slopes], axis=0)

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Welcome to pyphe-growthcurves, a light-weight script for non-parametric analysis of microbial growth curves. Written by stephan.kamrad@crick.ac.uk and maintained at https://github.com/Bahler-Lab/pyphe-growthcurves.')
    
    parser.add_argument('--input', type=str, required=True, help='Path to the growth curve file to analyse. This file contains one growth curve per column. The first column must be the timepoints and there must be a header row with unique identifiers for each curve.')
    
    parser.add_argument('--fitrange', type=int, default=4, help='Number of timepoint over which to fit linear regression. Defaults to 4. Please adjust this to the density of your timepoints and use higher values for more noisy data.')
    parser.add_argument('--lag-method', type=str, choices=['abs', 'rel'], default='rel', help='Method to use for determining lag. "abs" will measure time until the defined biomass threshold is crossed. "rel" will fist determine the inital biomass and measure the time until the biomass has passed this value times the threshold. Defaults to "rel".')
    parser.add_argument('--lag-threshold', type=float, default=2.0, help='Threshold to use for determining lag. With method "abs", this will measure time until the defined biomass threshold is crossed. With "rel" will fist determine the inital biomass and measure the time until the biomass has passed this value times the threshold. Defaults to 2.0, so with method "rel", this will measure the time taken for the first doubling.')
    parser.add_argument('--t0-fitrange', type=int, default=3, help='Specify the number of timepoint to use at the beginning of the growth curve to determine the initial biomass by averaging them. Defaults to 3.')
    parser.add_argument('--plots', default=False, action='store_true', help='Set this option (no argument required) to produce a plot of all growthcurves as pdf.')
    parser.add_argument('--plot-ylim', type=float, help='Specify the upper limit of the y-axis of growth curve plots. Useful if you want curves to be directly comparable. If not set, the axis of each curve is scaled to the data.')

    args = parser.parse_args()
    
    #Import the data and perform some basic checks
    gdata = pd.read_csv(args.input, index_col=0)
    try: 
        gdata.index = gdata.index.map(float)
    except Exception as eo:
        print('The first column must contain the timepoint and these must only have numeric values (no units or other string).')
    assert all(gdata.index[i] <= gdata.index[i+1] for i in range(len(gdata.index)-1)), 'Timepoints must be in ascending order.'
    
    result = analyse_growthcurve(gdata, args.fitrange, args.t0_fitrange, args.lag_method, args.lag_threshold, args.plots, args.plot_ylim)
    result.to_csv('.'.join(args.input.split('.')[:-1]) + '_results.csv')
    
    print('Analysis done: %s'%args.input)

    
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pylab import *
from math import erf
from scipy.stats import beta, norm, uniform
from scipy.special import betaln
from random import random, normalvariate
import numpy as np
from datetime import *
import os
import time

#################################
# Input data for simulations

#Percent improvements of B conv rate over A conv
# eleven 5% improvements are just as good as three 20% improvements

#WP.com Homepage Signup
chart_name = 'WP.com Homepage Signup Simulation  (12.5k users per hour)'
prior_conv_rate = 0.1
weekend_var = 0.02 # +/- 2% for weekends/weekdays
samples_per_hour = 12500
days = 7 #21
b_improvements = [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]

#Akismet Plugin Signup
#chart_name = 'Akismet Plugin Signup Simulation  (175 users per hour)'
#prior_conv_rate = 0.1456
#weekend_var = 0.12	 # +/- 2% for weekends/weekdays
#samples_per_hour = 175
#days = 7 #21
b_improvements = [-0.2, -0.15, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2]

#Akismet Developer Signup
#chart_name = 'Akismet Developer Signup Simulation (3 users per hour)'
#prior_conv_rate = 13/255
#weekend_var = 0.05 # +/- 2% for weekends/weekdays
#samples_per_hour = int( round( 255 / 4 / 24 ) )
#days = 21
b_improvements = [-0.2, -0.15, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2]

#################################
# Configuration

significant_cutoff = 1000
significant_cutoff_text = '1,000 samples'

hi_threshold = 0.99 #0.95
hi_threshold_suffix = ' sim at 99%'
low_threshold = 0.01 #0.05
low_threshold_suffix = ' sim at 1%'

# simulate all cases where the test reduces/improves the conversion rate from -5% to +5%
start_dt = datetime( 2015, 2, 25, 12) #Wed at noon GMT
hour_dt = timedelta( hours = 1 )
min_wait_dt = timedelta( days = 5 )

#################################
# Model Parameters

#Custom audience distribution - assume start on Wed cause I like to deploy on Wed
#hour_dist = [.01] * 24 #first day get 1% of audience per hour
#hour_dist += [.005] * 24 #Thurs
#hour_dist += [.005] * 24 #Fri
#hour_dist += [.01] * 24 #Sat - uptick in new users
#hour_dist += [.005] * 24 #Sun
#hour_dist += [0.01/3.0] * 24 #Mon
#hour_dist += [0.01/3.0] * 24 #Tues

#f(x) = log(a(x-b))
#log 0.0 to 0.8
#a = .026879498
#b = 37.203075742

#Build a logarithmic distribution of the audience
#f(x) = log(a(x-b))
#log 0.0 to 1.0
a = .010227857
b = 97.772192161
hour_dist = []
last = 0
step = 1 #does not work right yet
samples_per_step = step * samples_per_hour
for i in range(0,7*24,step):
    n = - abs( log(a*(i+b)) )
    hour_dist.append( last - n )
    last = n


#################################
# AB Testing functions


# Based on http://www.evanmiller.org/bayesian-ab-testing.html
def prob_b_beats_a(N_a, s_a, N_b, s_b, alpha, beta):
    a_a = alpha + s_a
    b_a = beta + N_a - s_a
    a_b = alpha + s_b
    b_b = beta + N_b - s_b
    #Using numpy here rather than a for loop speeds this up by 100x
    # For loop left in for clarity
    #total = 0.0
    #for i in range(0, a_b-1):
    #    total += exp( betaln(a_a+i, b_b+b_a) - log(b_b+i) - betaln(1+i, b_b) - betaln(a_a, b_a) )
    d = np.arange(0, a_b-1)
    total = np.sum( np.exp( betaln(a_a+d,b_b+b_a) - np.log(b_b+d) - betaln(1+d, b_b) - betaln(a_a, b_a) ) )
    return total

def sigma( mu, s, N ):
    return sqrt( ( ( pow( 1 - mu, 2 ) ) * s + pow( mu, 2 ) * ( N - s ) ) / N )

def isvalid_b_beats_a( N_a, s_a, N_b, s_b ):
    mu_a = s_a / N_a
    mu_b = s_b / N_b
    mu = mu_a - mu_b
    sigma_a = sigma( mu_a, s_a, N_a )
    sigma_b = sigma( mu_b, s_b, N_b )
    sigma_sq = pow( sigma_a, 2 ) / N_a + pow( sigma_a, 2 ) / N_b
    p = ( 1 + erf( -mu / sqrt( 2 * sigma_sq ) ) ) / 2
    return p


#################################
# Simulations

num_samples = days * 24 * samples_per_step
a_mean_weekday = prior_conv_rate * (1 + 2*weekend_var)
a_mean_weekend = prior_conv_rate * (1 - 5*weekend_var)

print a_mean_weekday
print a_mean_weekend
print '---------------------'

results_bayes_uniform = []
results_bayes_linear = []
results_bayes_audience = []
results_isvalid = []
sim_time = time.clock()
worst_model_time = 0
for i in b_improvements:
    samples_a = []
    samples_b = []
    b_mean_weekday = a_mean_weekday * (1.0 + i)
    b_mean_weekend = a_mean_weekend * (1.0 + i)
    print b_mean_weekday
    print b_mean_weekend
    tmp_a = np.random.random(num_samples)
    tmp_b = np.random.random(num_samples)
    curr_dt = start_dt
    for j in xrange( 0, num_samples):
	if ( curr_dt.weekday() in set([5,6]) ):
	    samples_a.append( int(tmp_a[j] < a_mean_weekend) )
	    samples_b.append( int(tmp_b[j] < b_mean_weekend) )
	else:
	    samples_a.append( int(tmp_a[j] < a_mean_weekday) )
	    samples_b.append( int(tmp_b[j] < b_mean_weekday) )
	curr_dt += hour_dt * step

    #look at the samples in chunks of one hour
    bayes_series_linear = []
    bayes_series_uniform = []
    bayes_series_audience = []
    isvalid_series = []
    curr_dt = start_dt
    audience_percent = 0.0
    aud_idx = 0
    for j in xrange( 0, num_samples, samples_per_step):
	sample_cnt = j + samples_per_step
	conv_a_cnt = np.sum( samples_a[0:sample_cnt] )
	conv_b_cnt = np.sum( samples_b[0:sample_cnt] )
	if ( aud_idx < len(hour_dist) - 1 ):
	    audience_percent += hour_dist[aud_idx]
	else:
	    audience_percent = 1.0
	aud_idx += 1

	#uniform
	alpha = 1
	beta = 1
	bayes_series_uniform.append( prob_b_beats_a( sample_cnt, conv_a_cnt, sample_cnt, conv_b_cnt, alpha, beta ) )

	#linear
	# alpha = number of successful conversions
	# beta = number of failed conversions
	#alpha = num_samples - j + 1
	#alpha = int( (num_samples - j) * prior_conv_rate ) + 1
	#beta = int( alpha * (1/prior_conv_rate - 1) ) + 1
	#bayes_series_linear.append( prob_b_beats_a( sample_cnt, conv_a_cnt, sample_cnt, conv_b_cnt, alpha, beta ) )

	#audience
	# balance with total weekly audience not yet seen
	#alpha = int( ( audience_percent * num_samples ) * prior_conv_rate ) + 1
	#beta = int( alpha * (1/prior_conv_rate - 1) ) + 1

	#Prior of half a week's average conversion rate
	alpha = int( num_samples/2 * prior_conv_rate );
	beta = int( alpha * (1/prior_conv_rate - 1) ) + 1

	model_time = time.clock()
	p = prob_b_beats_a( sample_cnt, conv_a_cnt, sample_cnt, conv_b_cnt, alpha, beta )
	model_time = time.clock() - model_time
	if ( model_time > worst_model_time ):
	    worst_model_time = model_time
	bayes_series_audience.append( p )
	#print curr_dt.strftime( "%a %H: " ) + "N: " + str(sample_cnt) + " s_a: " + str(conv_a_cnt) + " s_b: " + str(conv_b_cnt) + " alpha: " + str(alpha) + " beta: " + str(beta) + " aud: " + str(audience_percent) + " prob: " + str(p)

	isvalid_series.append( isvalid_b_beats_a( sample_cnt, conv_a_cnt, sample_cnt, conv_b_cnt ) )
	curr_dt += hour_dt * step

    results_bayes_uniform.append(bayes_series_uniform)
    #results_bayes_linear.append(bayes_series_linear)
    results_bayes_audience.append(bayes_series_audience)
    results_isvalid.append(isvalid_series)

sim_time = time.clock() - sim_time

print "Total Simulation Time: " + str(sim_time)
print "Worst Model Time: " + str(worst_model_time)

#################################
# Post Process - thresholds, etc

hourly_sample_cnt = []
series_time = []
curr_dt = start_dt
cutoff_date = 0
idx = 0
thresh_uniform = {}
thresh_linear = {}
thresh_audience = {}
for j in range( 0, num_samples, samples_per_step):
    hourly_sample_cnt.append( j + samples_per_step )
    if ( ( cutoff_date == 0 ) and ( j > significant_cutoff ) ):
	cutoff_date = curr_dt
    series_time.append( curr_dt )

    #Generate threshold crossings
    # Must have a minimum number of hours (6 right now) before we call something in case there are no conversions
    test_idx = 0
    if ( idx > 24 ):
	for i in b_improvements:
	    if ( i > 0 ):
		prefix = '+'
	    else:
		prefix = '-'
	    name =  prefix + str(abs(i * 100)) + '%'

	    if ( not thresh_uniform.has_key(i) and ( results_bayes_uniform[test_idx][idx] >= hi_threshold ) ):
		thresh_uniform[i] = (curr_dt,name + hi_threshold_suffix)
	    if ( not thresh_uniform.has_key(i) and ( results_bayes_uniform[test_idx][idx] <= low_threshold ) ):
		thresh_uniform[i] = (curr_dt,name + low_threshold_suffix)
#	    if ( not thresh_linear.has_key(i) and ( results_bayes_linear[test_idx][idx] >= hi_threshold ) ):
#		thresh_linear[i] = (curr_dt,name + hi_threshold_suffix)
#	    if ( not thresh_linear.has_key(i) and ( results_bayes_linear[test_idx][idx] <= low_threshold ) ):
#		thresh_linear[i] = (curr_dt,name + low_threshold_suffix)
	    if ( not thresh_audience.has_key(i) and ( results_bayes_audience[test_idx][idx] >= hi_threshold ) ):
		thresh_audience[i] = (curr_dt,name + hi_threshold_suffix)
	    if ( not thresh_audience.has_key(i) and ( results_bayes_audience[test_idx][idx] <= low_threshold ) ):
		thresh_audience[i] = (curr_dt,name + low_threshold_suffix)
	    test_idx += 1

    curr_dt += hour_dt * step
    idx += 1

#################################
# Plotting


# http://matplotlib.org/users/pyplot_tutorial.html

days_loc    = mdates.DayLocator()   # every day
hours_loc   = mdates.HourLocator()  # every hour
daysFmt = mdates.DateFormatter('%a') #day of week

# Colormaps: http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
colormap = plt.cm.winter
fig, ax = plt.subplots(3, 1)

#IsValid
ax[0].set_title('IsValid Probability B beats A', fontsize=12)
ax[0].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(b_improvements))])
lines = []
for i in range( 0, len(b_improvements) ):
    #print str( results_isvalid[i] )
    #print str( series_time )
    l, = ax[0].plot( series_time, results_isvalid[i] )
    lines.append( l )

if ( cutoff_date != 0 ):
    ax[0].axvline(cutoff_date, color='k')
    ax[0].annotate(significant_cutoff_text, xy=(cutoff_date, 0.5), xytext=(cutoff_date + 12*hour_dt, 0.5), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4) )

ax[0].axvline(start_dt + min_wait_dt, color='r')
ax[0].annotate("How long we're actually waiting?", xy=(start_dt + min_wait_dt, 0.5), xytext=(start_dt + min_wait_dt + 12*hour_dt, 0.5), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4) )

ax[0].axhline(0.95, color='k', linestyle='-.')
ax[0].axhline(0.99, color='k', linestyle='--')
ax[0].axhline(0.05, color='k', linestyle='-.')
ax[0].axhline(0.01, color='k', linestyle='--')
ax[0].xaxis.set_major_locator(days_loc)
ax[0].xaxis.set_major_formatter(daysFmt)
ax[0].xaxis.set_minor_locator(hours_loc)
ax[0].set_ylim([0,1.0])

line_names = []
for i in b_improvements:
    if ( i > 0 ):
	prefix = '+'
    else:
	prefix = '-'
    line_names.append( prefix + str(abs(i * 100)) + '% : ' + str(round(prior_conv_rate*(1+i)*100,1)) )
fig.legend( tuple(lines), tuple(line_names), loc='center right', title='% improvement of B over A' )
fig.suptitle( chart_name, fontsize=14 )

for sn in range(1,3):
    splt = ax[sn]
    if ( sn == 1 ):
	data = results_bayes_uniform
	thresh = thresh_uniform
	splt.set_title('Bayesian Probability B beats A (uniform prior)', fontsize=12)
#    elif ( sn == 2 ):
#	data = results_bayes_linear
#	thresh = thresh_linear
#	splt.set_title('Bayesian Probability B beats A (prior linearly decreases over one week)', fontsize=12)
    elif ( sn == 2 ):
	data = results_bayes_audience
	thresh = thresh_audience
	splt.set_title('Bayesian Probability B beats A (prior is ave for one half week)', fontsize=12)

    splt.set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(b_improvements))])
    for i in range( 0, len(b_improvements) ):
	splt.plot( series_time, data[i] )

    splt.axhline(0.95, color='k', linestyle='-.')
    splt.axhline(0.99, color='k', linestyle='--')
    splt.axhline(0.05, color='k', linestyle='-.')
    splt.axhline(0.01, color='k', linestyle='--')
    splt.xaxis.set_major_locator(days_loc)
    splt.xaxis.set_major_formatter(daysFmt)
    splt.xaxis.set_minor_locator(hours_loc)
    splt.set_ylim([0,1.0])

    hgt = 0.1
    for k in thresh.keys():
	splt.axvline(thresh[k][0],color='k')
	splt.annotate(thresh[k][1], xy=(thresh[k][0], hgt), xytext=(thresh[k][0] + 12*hour_dt, hgt), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4) )
	hgt += 0.1


plt.show()

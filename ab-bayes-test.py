from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pylab import *
from math import erf
from scipy.stats import beta, norm, uniform
from scipy.special import betaln
from random import random, normalvariate
from numpy import *
from datetime import *
import os

# Input data
alpha = 1
beta = 1
N_a = 200
N_b = 204
s_a = 16
s_b = 18

# Based on http://www.evanmiller.org/bayesian-ab-testing.html
def prob_b_beats_a(N_a, s_a, N_b, s_b, alpha, beta):
    a_a = alpha + s_a
    b_a = beta + N_a - s_a
    a_b = alpha + s_b
    b_b = beta + N_b - s_b
    total = 0.0
    for i in range(0, a_b-1):
	total += exp( betaln(a_a+i, b_b+b_a) - log(b_b+i) - betaln(1+i, b_b) - betaln(a_a, b_a) )
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


#prob_a_b = prob_b_beats_a(N_a, s_a, N_b, s_b, alpha, beta)
#print "Prob B beats A: " + str(prob_a_b)
#
#prob_a_b = isvalid_b_beats_a(N_a, s_a, N_b, s_b)
#print "isvalid Prob B beats A: " + str(prob_a_b)
#
#exit()

#test cases

significant_cutoff = 1000
significant_cutoff_text = '1,000 samples'
hi_threshold = 0.95
hi_threshold_suffix = ' reached 95%'
low_threshold = 0.05
low_threshold_suffix = ' reached 5%'

#WP.com Homepage Signup
#chart_name = 'WP.com Homepage Signup Simulation  (xxx users per hour)'
#prior_conv_rate = 0.35
#samples_per_hour = 100
#days = 7

#Akismet Plugin Signup : https://mc.a8c.com/tracks/akismet/acquisition/plugin-signup/
#chart_name = 'Akismet Plugin Signup Simulation  (175 users per hour)'
#prior_conv_rate = 0.1456
#samples_per_hour = 175
#days = 7

#Akismet Developer Signup
chart_name = 'Akismet Developer Signup Simulation (3 users per hour)'
prior_conv_rate = 13/255
samples_per_hour = int( round( 255 / 4 / 24 ) )
days = 21

#Build an audience distribution - assume start on Wed cause I like to deploy on Wed
#hour_dist = [.01] * 24 #first day get 1% of audience per hour
#hour_dist += [.005] * 24 #Thurs
#hour_dist += [.005] * 24 #Fri
#hour_dist += [.01] * 24 #Sat - uptick in new users
#hour_dist += [.005] * 24 #Sun
#hour_dist += [0.01/3.0] * 24 #Mon
#hour_dist += [0.01/3.0] * 24 #Tues

#f(x) = log(a(x-b))


num_samples = days * 24 * samples_per_hour
a_mean = prior_conv_rate

# simulate all cases where the test reduces/improves the conversion rate from -5% to +5%
start_dt = datetime( 2015, 2, 25, 12) #Wed at noon GMT
hour_dt = timedelta( hours = 1 )
#b_improvements = linspace( -0.1, 0.1, 21)
b_improvements = [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]
results_bayes_uniform = []
results_bayes_linear = []
results_bayes_audience = []
results_isvalid = []
for i in b_improvements:
    b_mean = a_mean * (1.0 + i)
    samples_a = []
    samples_b = []
    for j in range(0,num_samples):
	if ( random.random() < a_mean ):
	    samples_a.append( 1 )
	else:
	    samples_a.append( 0 )
	if ( random.random() < b_mean ):
	    samples_b.append( 1 )
	else:
	    samples_b.append( 0 )

    #look at the samples in chunks of one hour
    bayes_series_linear = []
    bayes_series_uniform = []
    bayes_series_audience = []
    isvalid_series = []
    curr_dt = start_dt
    audience_percent = 0.0
    aud_idx = 0
    for j in range( 0, num_samples, samples_per_hour):
	sample_cnt = j + samples_per_hour
	conv_a_cnt = sum( samples_a[0:sample_cnt] )
	conv_b_cnt = sum( samples_b[0:sample_cnt] )
	#TODO: is this the right formula for a beta dist with a mean?
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
	# TODO: correct alpha/beta
	# alpha = number of successful conversions
	# beta = number of failed conversions
	#alpha = num_samples - j + 1
	alpha = int( (num_samples - j) * prior_conv_rate ) + 1
	beta = int( alpha * (1/prior_conv_rate - 1) ) + 1
	bayes_series_linear.append( prob_b_beats_a( sample_cnt, conv_a_cnt, sample_cnt, conv_b_cnt, alpha, beta ) )

	#audience
	# TODO: correct alpha/beta
	# balance with total weekly audience not yet seen
	#alpha = int(audience_percent * num_samples) + 1
	alpha = int( ( audience_percent * num_samples ) * prior_conv_rate ) + 1
	beta = int( alpha * (1/prior_conv_rate - 1) ) + 1
	p = prob_b_beats_a( sample_cnt, conv_a_cnt, sample_cnt, conv_b_cnt, alpha, beta )
	bayes_series_audience.append( p )
	#print curr_dt.strftime( "%a %H: " ) + "N: " + str(sample_cnt) + " s_a: " + str(conv_a_cnt) + " s_b: " + str(conv_b_cnt) + " alpha: " + str(alpha) + " beta: " + str(beta) + " aud: " + str(audience_percent) + " prob: " + str(p)

	isvalid_series.append( isvalid_b_beats_a( sample_cnt, conv_a_cnt, sample_cnt, conv_b_cnt ) )
	curr_dt += hour_dt

    results_bayes_uniform.append(bayes_series_uniform)
    results_bayes_linear.append(bayes_series_linear)
    results_bayes_audience.append(bayes_series_audience)
    results_isvalid.append(isvalid_series)

hourly_sample_cnt = []
series_time = []
curr_dt = start_dt
cutoff_date = 0
idx = 0
thresh_uniform = {}
thresh_linear = {}
thresh_audience = {}
for j in range( 0, num_samples, samples_per_hour):
    hourly_sample_cnt.append( j + samples_per_hour )
    if ( ( cutoff_date == 0 ) and ( j > significant_cutoff ) ):
	cutoff_date = curr_dt
    series_time.append( curr_dt )

    #Generate threshold crossings
    # Must have a minimum number of hours before we call something in case there are no conversions
    if ( idx > 12 ):
	test_idx = 0
	for i in b_improvements:
	    if ( not thresh_uniform.has_key(i) and ( results_bayes_uniform[test_idx][idx] >= hi_threshold ) ):
		thresh_uniform[i] = (curr_dt,str(i) + hi_threshold_suffix)
	    if ( not thresh_uniform.has_key(i) and ( results_bayes_uniform[test_idx][idx] <= low_threshold ) ):
		thresh_uniform[i] = (curr_dt,str(i) + low_threshold_suffix)
	    if ( not thresh_linear.has_key(i) and ( results_bayes_linear[test_idx][idx] >= hi_threshold ) ):
		thresh_linear[i] = (curr_dt,str(i) + hi_threshold_suffix)
	    if ( not thresh_linear.has_key(i) and ( results_bayes_linear[test_idx][idx] <= low_threshold ) ):
		thresh_linear[i] = (curr_dt,str(i) + low_threshold_suffix)
	    if ( not thresh_audience.has_key(i) and ( results_bayes_audience[test_idx][idx] >= hi_threshold ) ):
		thresh_audience[i] = (curr_dt,str(i) + hi_threshold_suffix)
	    if ( not thresh_audience.has_key(i) and ( results_bayes_audience[test_idx][idx] <= low_threshold ) ):
		thresh_audience[i] = (curr_dt,str(i) + low_threshold_suffix)
	    test_idx += 1

    curr_dt += hour_dt
    idx += 1

# http://matplotlib.org/users/pyplot_tutorial.html

days_loc    = mdates.DayLocator()   # every day
hours_loc   = mdates.HourLocator()  # every hour
daysFmt = mdates.DateFormatter('%a') #day of week

# Colormaps: http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
colormap = plt.cm.winter
fig, ax = plt.subplots(4, 1)

#IsValid
ax[0].set_title('IsValid Probability B beats A')
ax[0].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(b_improvements))])
for i in range( 0, len(b_improvements) ):
    #print str( results_isvalid[i] )
    #print str( series_time )
    ax[0].plot( series_time, results_isvalid[i] )

if ( cutoff_date != 0 ):
    ax[0].axvline(cutoff_date, color='k')
    ax[0].annotate(significant_cutoff_text, xy=(cutoff_date, 0.5), xytext=(cutoff_date + 12*hour_dt, 0.5), arrowprops=dict(facecolor='black', shrink=0.05) )

ax[0].axhline(0.95, color='k', linestyle='-.')
ax[0].axhline(0.99, color='k', linestyle='--')
ax[0].axhline(0.05, color='k', linestyle='-.')
ax[0].axhline(0.01, color='k', linestyle='--')
ax[0].xaxis.set_major_locator(days_loc)
ax[0].xaxis.set_major_formatter(daysFmt)
ax[0].xaxis.set_minor_locator(hours_loc)
ax[0].set_ylim([0,1.0])
#ax[0].legend(b_improvements, loc='upper right', title='% improvement of B over A')
#TODO: use figlegend

for sn in range(1,4):
    splt = ax[sn]
    if ( sn == 1 ):
	data = results_bayes_uniform
	thresh = thresh_uniform
	splt.set_title('Bayesian Probability B beats A (uniform prior)')
    elif ( sn == 2 ):
	data = results_bayes_linear
	thresh = thresh_linear
	splt.set_title('Bayesian Probability B beats A (prior linearly decreases over one week)')
    elif ( sn == 3 ):
	data = results_bayes_audience
	thresh = thresh_audience
	splt.set_title('Bayesian Probability B beats A (prior based on percentage of weekly audience)')

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
	splt.annotate(thresh[k][1], xy=(thresh[k][0], hgt), xytext=(thresh[k][0] + 12*hour_dt, hgt), arrowprops=dict(facecolor='black', shrink=0.05) )
	hgt += 0.2


#Uniform
#ax[1].set_title('Bayesian Probability B beats A (uniform prior)')
#ax[1].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(b_improvements))])
#for i in range( 0, len(b_improvements) ):
#    ax[1].plot( series_time, results_bayes_uniform[i] )
#
#ax[1].axhline(0.95, color='k', linestyle='-.')
#ax[1].axhline(0.99, color='k', linestyle='--')
#ax[1].axhline(0.05, color='k', linestyle='-.')
#ax[1].axhline(0.01, color='k', linestyle='--')
#ax[1].xaxis.set_major_locator(days_loc)
#ax[1].xaxis.set_major_formatter(daysFmt)
#ax[1].xaxis.set_minor_locator(hours_loc)
#ax[1].set_ylim([0,1.0])
#
#hgt = 0.1
#for k in thresh_uniform.keys():
#    ax[1].axvline(thresh_uniform[k][0],color='k')
#    ax[1].annotate(thresh_uniform[k][1], xy=(thresh_uniform[k][0], hgt), xytext=(thresh_uniform[k][0] + 6*hour_dt, hgt), arrowprops=dict(facecolor='black', shrink=0.05) )
#    hgt += 0.2
#
##Linear
#ax[2].set_title('Bayesian Probability B beats A (prior linearly decreases over one week)')
#ax[2].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(b_improvements))])
#for i in range( 0, len(b_improvements) ):
#    ax[2].plot( series_time, results_bayes_linear[i] )
#
#ax[2].axhline(0.95, color='k', linestyle='-.')
#ax[2].axhline(0.99, color='k', linestyle='--')
#ax[2].axhline(0.05, color='k', linestyle='-.')
#ax[2].axhline(0.01, color='k', linestyle='--')
#ax[2].xaxis.set_major_locator(days_loc)
#ax[2].xaxis.set_major_formatter(daysFmt)
#ax[2].xaxis.set_minor_locator(hours_loc)
#ax[2].set_ylim([0,1.0])
#
##Audience
#ax[3].set_title('Bayesian Probability B beats A (prior based on percentage of weekly audience)')
#ax[3].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(b_improvements))])
#for i in range( 0, len(b_improvements) ):
#    ax[3].plot( series_time, results_bayes_linear[i] )
#
#ax[3].axhline(0.95, color='k', linestyle='-.')
#ax[3].axhline(0.99, color='k', linestyle='--')
#ax[3].axhline(0.05, color='k', linestyle='-.')
#ax[3].axhline(0.01, color='k', linestyle='--')
#ax[3].xaxis.set_major_locator(days_loc)
#ax[3].xaxis.set_major_formatter(daysFmt)
#ax[3].xaxis.set_minor_locator(hours_loc)
#ax[3].set_ylim([0,1.0])

plt.show()

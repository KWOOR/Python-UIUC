# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 03:32:23 2019

@author: kur7
"""


import fix_yahoo_finance as yf
yf.pdr_override() # <== that's 

l = ['AA',
'AAN',
'AB',
'ABC',
'ABM',
'ABT',
'ADC',
'ADM',
'AEE',
'AEO',
'AEP',
'AES',
'AFG',
'AFL',
'AGCO',
'AGN',
'AIG',
'AIR',
'AIT',
'AIV',
'AJG',
'AJRD',
'AKR',
'AKS',
'ALB',
'ALE',
'ALK',
'ALL',
'AME',
'AMG',
'AMT',
'AN',
'ANF',
'ANH',
'AOS',
'APA',
'APC',
'APD',
'APH',
'APU',
'ARE',
'ARNC',
'ARW',
'ASB',
'ASGN',
'ASH',
'ATO',
'ATR',
'AVA',
'AVB',
'AVP',
'AVX',
'AVY',
'AXL',
'AXP',
'AZO',
'BA',
'BAC',
'BAX',
'BBT',
'BBY',
'BC',
'BCO',
'BDC',
'BDN',
'BDX',
'BEN',
'BF-B',
'BGG',
'BHE',
'BID',
'BIG',
'BIO',
'BK',
'BKE',
'BKH',
'BKI',
'BKS',
'BLL',
'BMS',
'BMY',
'BOH',
'BPL',
'BPT',
'BRC',
'BRK-B',
'BRO',
'BSX',
'BWA',
'BXG',
'BXMT',
'BXP',
'BXS',
'BYD',
'BZH',
'C',
'CAG',
'CAH',
'CAL',
'CAT',
'CATO',
'CB',
'CBB',
'CBM',
'CBT',
'CBZ',
'CCI',
'CCK',
'CCL',
'CDE',
'CFR',
'CHD',
'CHH',
'CHK',
'CHS',
'CI',
'CIEN',
'CL',
'CLF',
'CLGX',
'CLH',
'CLI',
'CLX',
'CMA',
'CMC',
'CMD',
'CMI',
'CMO',
'CMS',
'CNA',
'CNP',
'COF',
'COG',
'COO',
'COP',
'CPB',
'CPE',
'CPT',
'CR',
'CRF',
'CRK',
'CRS',
'CRY',
'CSL',
'CTB',
'CTL',
'CUB',
'CUZ',
'CVA',
'CVS',
'CVX',
'CW',
'CWT',
'CXW',
'D',
'DAR',
'DBD',
'DCI',
'DDD',
'DDS',
'DE',
'DECK',
'DGX',
'DHI',
'DHR',
'DIN',
'DIS',
'DLX',
'DNR',
'DO',
'DOV',
'DRE',
'DRI',
'DRQ',
'DTE',
'DUK',
'DVA',
'DVN',
'DX',
'DY',
'EAT',
'ECL',
'ED',
'EE',
'EFX',
'EGHT',
'EIX',
'EL',
'ELS',
'ELY',
'EME',
'EMN',
'EMR',
'EOG',
'EPD',
'EPR',
'EQC',
'EQR',
'EQT',
'ES',
'ESS',
'ETH',
'ETM',
'ETR',
'EV',
'EXC',
'EXP',
'F',
'FBC',
'FBP',
'FCF',
'FCN',
'FCX',
'FDS',
'FDX',
'FE',
'FHN',
'FICO',
'FII',
'FIX',
'FL',
'FLO',
'FLS',
'FMC',
'FNB',
'FOE',
'FR',
'FRT',
'FSS',
'FUL',
'FUN',
'GATX',
'GBX',
'GCO',
'GD',
'GE',
'GEF',
'GEL',
'GEO',
'GES',
'GFF',
'GGG',
'GIS',
'GLT',
'GLW',
'GPC',
'GPK',
'GPS',
'GRA',
'GVA',
'GWR',
'GWW',
'HAE',
'HAL',
'HCP',
'HD',
'HE',
'HEI',
'HEI-A',
'HES',
'HFC',
'HIG',
'HIW',
'HL',
'HLX',
'HOG',
'HON',
'HP',
'HPQ',
'HR',
'HRB',
'HRC',
'HRL',
'HRS',
'HSC',
'HST',
'HSY',
'HT',
'HUM',
'HXL',
'HZO',
'IBM',
'IDA',
'IEX',
'IFF',
'IGT',
'INGR',
'INT',
'IP',
'IPG',
'IRM',
'IT',
'ITT',
'ITW',
'IVZ',
'JBL',
'JCI',
'JEC',
'JLL',
'JNJ',
'JPM',
'JW-A',
'JWN',
'K',
'KBH',
'KEM',
'KEX',
'KEY',
'KFY',
'KIM',
'KMB',
'KMPR',
'KMT',
'KMX',
'KNX',
'KO',
'KR',
'KRC',
'KSS',
'KSU',
'L',
'LAD',
'LB',
'LEG',
'LEN',
'LGF-A',
'LH',
'LLL',
'LLY',
'LM',
'LMT',
'LNC',
'LNG',
'LOW',
'LPT',
'LPX',
'LSI',
'LUV',
'LXP',
'LZB',
'M',
'MAA',
'MAC',
'MAN',
'MAS',
'MBI',
'MCD',
'MCK',
'MCO',
'MCS',
'MCY',
'MD',
'MDC',
'MDP',
'MDR',
'MDT',
'MDU',
'MED',
'MEI',
'MFA',
'MGM',
'MHK',
'MHO',
'MKC',
'MLI',
'MLM',
'MMC',
'MMM',
'MMS',
'MNR',
'MO',
'MOS',
'MOV',
'MRK',
'MRO',
'MS',
'MSI',
'MSM',
'MTB',
'MTG',
'MTH',
'MTN',
'MTW',
'MTZ',
'MUR',
'NAV',
'NBL',
'NCI',
'NCR',
'NCS',
'NEE',
'NEM',
'NFG',
'NHI',
'NI',
'NJR',
'NKE',
'NLY',
'NNN',
'NOC',
'NOV',
'NR',
'NSC',
'NSP',
'NUE',
'NUS',
'NX',
'NYCB',
'NYT',
'O',
'OFC',
'OFG',
'OGE',
'OHI',
'OI',
'OII',
'OKE',
'OLN',
'OMC',
'ORCL',
'ORI',
'OSK',
'OXY',
'PAA',
'PAG',
'PB',
'PBI',
'PCG',
'PEG',
'PEI',
'PFE',
'PG',
'PGR',
'PH',
'PHM',
'PII',
'PKI',
'PLD',
'PLT',
'PNC',
'PNM',
'PNW',
'PPG',
'PPL',
'PRA',
'PSA',
'PVH',
'PWR',
'PXD',
'R',
'RAD',
'RBC',
'RCL',
'RDN',
'RES',
'RF',
'RGS',
'RHI',
'RHP',
'RJF',
'RL',
'RMD',
'ROK',
'ROL',
'ROP',
'RPM',
'RPT',
'RRC',
'RRD',
'RS',
'RSG',
'RTN',
'RWT',
'RYN',
'S',
'SAH',
'SCCO',
'SCHW',
'SCI',
'SCS',
'SEE',
'SF',
'SHW',
'SJI',
'SJM',
'SKT',
'SKY',
'SLB',
'SLG',
'SM',
'SMG',
'SNA',
'SNV',
'SO',
'SON',
'SPG',
'SPGI',
'SPN',
'SPXC',
'SR',
'SRE',
'SRI',
'SSD',
'STAR',
'STE',
'STI',
'STL',
'STT',
'STZ',
'SUI',
'SWK',
'SWN',
'SWX',
'SXT',
'SYK',
'SYY',
'T',
'TAP',
'TCF',
'TDS',
'TEN',
'TEX',
'TFX',
'TGI',
'TGNA',
'TGT',
'THC',
'THG',
'THO',
'TIF',
'TISI',
'TJX',
'TKR',
'TLRD',
'TMK',
'TMO',
'TOL',
'TPC',
'TREX',
'TRN',
'TRV',
'TRXC',
'TSN',
'TSS',
'TTC',
'TTI',
'TUP',
'TWI',
'TXT',
'TYL',
'UDR',
'UGI',
'UHS',
'UIS',
'UNFI',
'UNH',
'UNM',
'UNP',
'UNT',
'URI',
'USB',
'USG',
'USM',
'USNA',
'UTX',
'VAR',
'VFC',
'VGR',
'VHI',
'VLO',
'VMC',
'VNO',
'VSH',
'VTR',
'VZ',
'WAB',
'WAT',
'WBS',
'WCN',
'WDR',
'WEC',
'WELL',
'WFC',
'WGO',
'WHR',
'WM',
'WMB',
'WMT',
'WNC',
'WOR',
'WP',
'WPC',
'WRB',
'WRE',
'WRI',
'WSM',
'WSO',
'WST',
'WTR',
'WWW',
'WY',
'X',
'XOM',
'XRX',
'YUM']

import pandas as pd

#%%



data= pd.DataFrame()
for i in range(len(l)):
    buff = yf.download(l[i], start = "2007-01-01", end = "2019-03-31")
    buff1 = pd.DataFrame([l[i]]*len(buff), columns=['Ticker'], index=buff.index)
    empty = pd.concat([buff1, buff], axis=1)
    data = pd.concat( [data, empty])

#%%
    
    
data.iloc[:int(len(data)/2),:].to_csv('SA1.csv')
data.iloc[int(len(data)/2):,:].to_csv('SA2.csv')





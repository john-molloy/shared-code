"""
US Bureau of Labor Statistics


API Version 2.0 Python Sample Code
Multiple Series and Multiple Years
Use this code to retrieve data for more than one timeseries and more than one year.

TODO
----
* Determine the best way of identifying data error returns.
* Add Python error-handling.

Modification History
--------------------
02/01/2021 JM: New.
07/01/2021 JM: Change the way it specifies log and download directories.
09/01/2021 JM: Change back to saving csv files in sub-directory of script dir.
22/04/2021 JM: 1) Use a params dictionary. 2) Check p.status_code.

@author: johnm
"""

import datetime
import json
import os
import requests


def download_directory():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        'downloads')

def log_file_directory():
    return os.path.dirname(os.path.abspath(__file__));

def log_time():
    now = datetime.datetime.now()
    return '{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}{5:02d}_{6:02d}'.format(
            now.year, now.month, now.day,
            now.hour, now.minute, now.second,
            int(round(now.microsecond / 10000, 0)))


if __name__ == "__main__":

    params = {}
    params['downloader_desc']      = 'BLS'
    params['download_file_prefix'] = 'bls'
    params['base_url']   = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'


    log_file = os.path.join(
        log_file_directory(),
        'download_log.t')

    now = datetime.datetime.now()
    output_file_timestamp = '{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}{5:02d}'.format(
        now.year, now.month, now.day,
        now.hour, now.minute, now.second)

    with open(log_file, 'at') as f:
        f.write("{0}: {1}: download start\n".format(log_time(), params['downloader_desc']))


    series = ['CUUR0000SA0', 'SUUR0000SA0']
    startyear = str(now.year - 1)
    endyear = str(now.year)

    headers = {'Content-type': 'application/json'}

    data = json.dumps({
        "seriesid": series,
        "startyear": startyear,
        "endyear": endyear})

    p = requests.post(
        params['base_url'],
        data=data,
        headers=headers)

    if p.status_code != 200:

        with open(log_file, 'at') as f:
            f.write("{0}: {1}: error: {2}\n".format(log_time(), params['downloader_desc'], p.status_code))

    else:

        json_data = json.loads(p.text)

        for series in json_data['Results']['series']:

            data_by_row = []
            data_by_row.append("SeriesId,Year,Period,Value,Footnotes")

            series_id = series['seriesID']

            for item in series['data']:
                year = item['year']
                period = item['period']
                value = item['value']
                footnotes = ""
                for footnote in item['footnotes']:
                    if footnote:
                        footnotes = footnotes + footnote['text'] + ','
                qq = ''
                for q in [series_id, year, period, value, footnotes[0:-1]]:
                    qq = qq + q + ','
                data_by_row.append(qq)

            output_file = os.path.join(
                download_directory(),
                "{0}_{1}_{2}.csv".format(
                    params['download_file_prefix'],
                    series_id,
                    output_file_timestamp))

            with open(output_file, 'w') as f:
                for row in data_by_row:
                    f.write(row + '\n')

    with open(log_file, 'at') as f:
        f.write("{0}: {1}: download end\n".format(log_time(), params['downloader_desc']))

#
# End Of File
#

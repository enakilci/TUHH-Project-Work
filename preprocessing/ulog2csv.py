#! /usr/bin/env python

from __future__ import print_function
import argparse
import os
from core import ULog


def convert_ulog2csv(ulog_file_name, messages=None, output=None, delimiter=",", disable_str_exceptions=False):
    """Converts ULog file to csv files
    """


    if output and not os.path.isdir(output):
        print('Creating output directory {:}'.format(output))
        os.mkdir(output)

    msg_filter = messages.split(',') if messages else None

    ulog = ULog(ulog_file_name, msg_filter, disable_str_exceptions)
    data = ulog.data_list

    output_file_prefix = ulog_file_name
    # strip '.ulg'
    if output_file_prefix.lower().endswith('.ulg'):
        output_file_prefix = output_file_prefix[:-4]
    
    # write to different output path?

    
    if output:
        base_name = os.path.basename(output_file_prefix)
        output_file_prefix = os.path.join(output, base_name)
    
    print("converting Ulog file...")
    for d in data:
        fmt = '{0}_{1}_{2}.csv'
        output_file_name = fmt.format(output_file_prefix, d.name, d.multi_id)
        fmt = 'Writing {0} ({1} data points)'
        with open(output_file_name, 'w') as csvfile:

            # use same field order as in the log, except for the timestamp
            data_keys = [f.field_name for f in d.field_data]
            data_keys.remove('timestamp')
            data_keys.insert(0, 'timestamp')  # timestamp at first position

            # write the header
            csvfile.write(delimiter.join(data_keys) + '\n')

            # write the data
            last_elem = len(data_keys)-1
            for i in range(len(d.data['timestamp'])):
                for k in range(len(data_keys)):
                    csvfile.write(str(d.data[data_keys[k]][i]))
                    if k != last_elem:
                        csvfile.write(delimiter)
                csvfile.write('\n')
    print("ULog file has been succesfully converted and saved to the directory {}".format(os.path.dirname(output_file_prefix)))


def main(argv):

    convert_ulog2csv(argv[1],output=argv[2])


if __name__ == "__main__":
    file_path = "/home/eminc/Desktop/projektarbeit/log_files/log_08_11/log_20_2019-8-11-18-19-14.ulg"
    output ="/home/eminc/Desktop/projektarbeit/log_files/log_08_11/csv_files"
    inp = ["",file_path,output]
    main(inp)

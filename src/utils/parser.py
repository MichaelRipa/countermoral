#! /usr/bin/python3

def parse_list(fpath):

    f = open(fpath, 'r')
    contents = f.read()
    f.close()

    # Discards empty string
    return contents.split('\n')[0:-1]




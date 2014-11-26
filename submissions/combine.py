#!/usr/bin/env python2.7

import numpy as np
import sys
import gzip

if len(sys.argv) != 3:
  print >>sys.err, 'Usage: ./combine.py submission0.csv.gz submission1.csv.gz | gzip >combined-0-1.csv.gz'

filenames = sys.argv[1:]
files = [gzip.open(filename, 'rb') for filename in filenames]

print [f.readline() for f in files][0],

def split(line):
  t, p = line.split(',')
  return t, float(p)

while True:
  lines = [f.readline() for f in files]
  if lines[0] == "":
    break;

  t, p = zip(*[split(line) for line in lines])

  for tt in t:
    assert(tt == t[0])

  p = np.mean(p)
  print '%s,%.10f' % (t[0], p)

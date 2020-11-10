# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import sys
import subprocess

# scratchdir is provided by user issuing the command
scratchdir = sys.argv[1]
hdfsdir = scratchdir.replace("/scratch", "/user")

(status1,output1) = subprocess.getstatusoutput('find %s -type f -exec ls -l {} +' % scratchdir)
if status1 != 0: 
    print("The find command failed")
    sys.exit(1)
(status2,output2) = subprocess.getstatusoutput('hdfs dfs -ls -R %s | grep -v "^d"' % hdfsdir)
if status2 != 0: 
    print("The HDFS command failed")
    sys.exit(1)

scratchfiles = {}
hdfsfiles = {}
for file in output1.splitlines():
    items = file.split()
    scratchfiles[items[8]] = items[4]
for file in output2.splitlines():
    items = file.split()
    hdfsfiles[items[7]] = items[4]

for sfile, ssize in scratchfiles.items():
    hfile = sfile.replace("/scratch", "/user")
    if not hfile in hdfsfiles:
        print("Found new file %s, copying to HDFS" % sfile)
        # os.system('hdfs dfs -put %s %s' %(sfile, hfile))
    elif ssize != hdfsfiles[hfile]:
        print("The file %s is updated, updating to HDFS" % sfile)
        # os.system('hdfs dfs -put -f %s %s' %(sfile, hfile))

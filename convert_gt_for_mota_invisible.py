import glob
import os

if __name__ == "__main__":
    files = sorted(glob.glob("/data/tkhurana/MOT17/train/*/gt/gt.txt"))
    for file in files:
        outfile = file.replace("gt.txt", "gt_motai.txt")
        outfile = open(outfile, "w")
        lines = open(file).readlines()
        for line in lines:
            fields = line.strip().split(',')
            if fields[7] == '1' and float(fields[8]) > 0.1:
                outfile.write("{},{},{},{},{},{},{},2,{}\n".format(
                    fields[0], fields[1], fields[2], fields[3],
                    fields[4], fields[5], 0, fields[8]))
            elif fields[7] == 1:
                outfile.write("{},{},{},{},{},{},{},1,{}\n".format(
                    fields[0], fields[1], fields[2], fields[3],
                    fields[4], fields[5], fields[6], fields[8]))
            else:
                outfile.write("{},{},{},{},{},{},{},{},{}\n".format(
                    fields[0], fields[1], fields[2], fields[3],
                    fields[4], fields[5], fields[6], fields[7], fields[8]))



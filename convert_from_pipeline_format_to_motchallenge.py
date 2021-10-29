import glob
import os

if __name__ == "__main__":
    files = sorted(glob.glob("/data/tkhurana/tk/deep_sort/results/deepsort+extrapolate+depth+tunedoccluded+tunedfreespace+tunedtemporal+ego_bugfix_covarianceoutput_ISE_MOT17R_MOT17train/*.txt"))
    for file in files:
        outfile = file.replace("deepsort+extrapolate+depth+tunedoccluded+tunedfreespace+tunedtemporal+ego_bugfix_covarianceoutput_ISE_MOT17R_MOT17train", "deepsort+extrapolate+depth+tunedoccluded+tunedfreespace+tunedtemporal+ego_bugfix_covarianceoutput_ISE_MOT17R_MOT17train_motformat")
        outfile = open(outfile, "w")
        lines = open(file).readlines()
        for line in lines:
            fields = line.strip().split(',')[:6]
            outfile.write("{},{},{},{},{},{},1,-1,-1,-1\n".format(
                fields[0], fields[1], fields[2], fields[3], fields[4], fields[5]))


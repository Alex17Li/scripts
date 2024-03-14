#!/bin/bash
#SBATCH --job-name=rm_data
#SBATCH --partition=cpu
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/mnt/sandbox1/%u/logs/%A_%x.txt
#SBATCH --ntasks=1
cd /data/jupiter/datasets/
TO_KILL=(
    "Jupiter_20230720_HHH3_1805_1835"
    "Jupiter_20230803_HHH2_2030_2100"
    "Jupiter_20230803_HHH3_2115_2145"
    "Jupiter_20230814_HHH1_1415_1445"
    "Jupiter_20230823_HHH3_1815_1845"
    "Jupiter_20230803_HHH2_1400_1430"
    "Jupiter_20230825_HHH1_1730_1800"
    "Jupiter_20230926_HHH1_1815_1845"
    "Jupiter_20230927_HHH1_0100_0130"
    "Jupiter_20231007_HHH1_2350_0020"
    "Jupiter_20231019_HHH6_1615_1700"
    "Jupiter_20231019_HHH6_1800_1830"
    "Jupiter_20231026_HHH8_1515_1545"
)
for DATASET in ${TO_KILL[@]}
do
    echo $DATASET
    rm -rf /data/jupiter/datasets/$DATASET
done


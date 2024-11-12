#!/bin/bash   

grid=NNPDF40_nnlo_pch_as_01180_NNev
cd /unix/theory2/lang/fit_standalone_theory2/outputs/evgrids/
cd $grid
cd nnfit
mkdir $grid
cp ${grid}.info $grid

nlines=$(find rep* -maxdepth 0 -type d | wc -l)
echo "${nlines}"

#nlines="$((${nlines}-2))"
# echo "${nlines}"


ntot=${nlines}
echo "$ntot"
sed -i "7s/.*/NumMembers: ${ntot}/" $grid/${grid}.info   

for ((n=1;n<=$ntot+1;n++))
do
gnum=$(($n - 1))
# gnum=$(($n - 2))
echo "$gnum"

if [ $gnum -lt 10 ]
then 
    cp replica_${n}/${grid}.dat ${grid}/${grid}_000${gnum}.dat
elif [ $gnum -lt 100 ]
then
    cp replica_${n}/${grid}.dat ${grid}/${grid}_00${gnum}.dat
else
    cp replica_${n}/${grid}.dat ${grid}/${grid}_0${gnum}.dat
fi

done

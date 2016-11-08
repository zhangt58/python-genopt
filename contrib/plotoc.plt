#!/usr/bin/gnuplot --persist


set terminal post eps color lw 2 solid enhanced "Helvetica" 20
#set output 'dakota_oc1.eps'
#set output 'dakota_oc2.eps'
#set output 'dakota_oc3.eps'
set output 'dakota_oc4.eps'
set size 1.0, 1.0

set style line 1 lt 1 lw 2
set style line 2 lt 2 lw 2
set style line 3 lt 3 lw 2

set xlabel "z [m]" font "Helvetica, 22"
set ylabel "Envelop [mm]" font "Helvetica, 22"

set key right top font "Helvetica, 22"

#plot 'zxy_dakota2.dat' u 1:2 w l ls 1 t '#S2000', \
#     'zxy_dakota3.dat' u 1:2 w l ls 2 t '#S500'
     #'zxy_scipy.dat' u 1:2 w l ls 2 t '#scipy'
     #'zxy_dakota2.dat' u 1:2 w l ls 2 t '#S1'
     # 'zxy_dakota1.dat' u 1:2 w l ls 1 t '#G1', \

plot './contrib/dakota1.dat' u 1:19 w l ls 1 t '#G1 (40s)', \
     './contrib/dakota2.dat' u 1:19 w l ls 2 t '#S500 (20s)'

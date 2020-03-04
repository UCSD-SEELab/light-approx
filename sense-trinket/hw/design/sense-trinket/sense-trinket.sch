EESchema Schematic File Version 4
LIBS:sense-trinket-cache
EELAYER 26 0
EELAYER END
$Descr USLetter 11000 8500
encoding utf-8
Sheet 1 1
Title "Sense Trinket"
Date "2020-01-24"
Rev "1.0"
Comp "UC San Diego"
Comment1 "SEE Lab"
Comment2 "Michael H. Ostertag"
Comment3 ""
Comment4 "A portable light sensing header for the Arduino Nano"
$EndDescr
$Comp
L Sensor_Optical:AS7262 U1
U 1 1 5E266FD3
P 3450 3750
F 0 "U1" H 3050 4400 50  0000 L CNN
F 1 "AS7262" H 3050 4300 50  0000 L CNN
F 2 "common:AMS_LGA-20_4.7x4.5mm_P0.65mm_handsolder" H 3450 3050 50  0001 C CNN
F 3 "http://ams.com/eng/content/download/976551/2309439/498718" H 4050 3950 50  0001 C CNN
	1    3450 3750
	1    0    0    -1  
$EndComp
$Comp
L Battery_Management:MCP73831-2-OT U2
U 1 1 5E267132
P 4050 1650
F 0 "U2" H 3650 1750 50  0000 R CNN
F 1 "MCP73831-2-OT" H 3650 1650 50  0000 R CNN
F 2 "common:SOT-23-5_HandSoldering" H 4100 1400 50  0001 L CIN
F 3 "http://ww1.microchip.com/downloads/en/DeviceDoc/20001984g.pdf" H 3900 1600 50  0001 C CNN
	1    4050 1650
	-1   0    0    -1  
$EndComp
$Comp
L Timer_RTC:PCF8563TS U3
U 1 1 5E26724C
P 7750 2450
F 0 "U3" H 7200 2950 50  0000 L CNN
F 1 "PCF85063AT" H 7200 2850 50  0000 L CNN
F 2 "common:SOIC-8_3.9x4.9mm_P1.27mm" H 7750 2450 50  0001 C CNN
F 3 "http://www.nxp.com/documents/data_sheet/PCF8563.pdf" H 7750 2450 50  0001 C CNN
	1    7750 2450
	1    0    0    -1  
$EndComp
$Comp
L MCU_Module:Arduino_Nano_v3.x J2
U 1 1 5E2673FE
P 3450 6350
F 0 "J2" H 2600 7500 50  0000 C CNN
F 1 "Arduino_Nano_ble_v3.x" H 2900 7400 50  0000 C CNN
F 2 "Module:Arduino_Nano" H 3600 5400 50  0001 L CNN
F 3 "http://www.mouser.com/pdfdocs/Gravitech_Arduino_Nano3_0.pdf" H 3450 5350 50  0001 C CNN
	1    3450 6350
	1    0    0    -1  
$EndComp
$Comp
L Connector_Generic:Conn_01x02 J1
U 1 1 5E26803B
P 2150 1550
F 0 "J1" H 2070 1767 50  0000 C CNN
F 1 "Conn_01x02" H 2070 1676 50  0000 C CNN
F 2 "sense-trinket:JST_socket_2pin_horizontal" H 2150 1550 50  0001 C CNN
F 3 "~" H 2150 1550 50  0001 C CNN
	1    2150 1550
	-1   0    0    -1  
$EndComp
$Comp
L sense-trinket-library:BattRetainer J3
U 1 1 5E268CC5
P 8600 1250
F 0 "J3" H 8300 1550 50  0000 C CNN
F 1 "BattRetainer" H 8500 1500 50  0000 C CNN
F 2 "sense-trinket:COIN_CELL_S8201R" H 8600 1250 50  0001 C CNN
F 3 "" H 8600 1250 50  0001 C CNN
	1    8600 1250
	1    0    0    -1  
$EndComp
$Comp
L Device:LED D2
U 1 1 5E26939F
P 3050 1350
F 0 "D2" H 3150 1150 50  0000 R CNN
F 1 "LED" H 3150 1250 50  0000 R CNN
F 2 "passives:LED_1206_3216Metric_Pad1.42x1.75mm_HandSolder" H 3050 1350 50  0001 C CNN
F 3 "~" H 3050 1350 50  0001 C CNN
	1    3050 1350
	-1   0    0    1   
$EndComp
$Comp
L Device:LED D3
U 1 1 5E26957D
P 3050 1750
F 0 "D3" H 3000 1650 50  0000 L CNN
F 1 "LED" H 3000 1550 50  0000 L CNN
F 2 "passives:LED_1206_3216Metric_Pad1.42x1.75mm_HandSolder" H 3050 1750 50  0001 C CNN
F 3 "~" H 3050 1750 50  0001 C CNN
	1    3050 1750
	1    0    0    -1  
$EndComp
$Comp
L Device:LED D1
U 1 1 5E2695A9
P 2250 4050
F 0 "D1" H 2300 4150 50  0000 R CNN
F 1 "LED" H 2300 4250 50  0000 R CNN
F 2 "passives:LED_1206_3216Metric_Pad1.42x1.75mm_HandSolder" H 2250 4050 50  0001 C CNN
F 3 "~" H 2250 4050 50  0001 C CNN
	1    2250 4050
	-1   0    0    1   
$EndComp
$Comp
L Device:LED D4
U 1 1 5E2697A2
P 6650 4600
F 0 "D4" V 6700 4800 50  0000 R CNN
F 1 "LED" V 6600 4800 50  0000 R CNN
F 2 "passives:LED_1206_3216Metric_Pad1.42x1.75mm_HandSolder" H 6650 4600 50  0001 C CNN
F 3 "~" H 6650 4600 50  0001 C CNN
	1    6650 4600
	0    -1   -1   0   
$EndComp
$Comp
L Device:D D6
U 1 1 5E26998C
P 8050 1500
F 0 "D6" V 8096 1421 50  0000 R CNN
F 1 "D" V 8005 1421 50  0000 R CNN
F 2 "passives:LED_1206_3216Metric_Pad1.42x1.75mm_HandSolder" H 8050 1500 50  0001 C CNN
F 3 "~" H 8050 1500 50  0001 C CNN
	1    8050 1500
	0    -1   -1   0   
$EndComp
$Comp
L Device:R_US R5
U 1 1 5E269DC9
P 4550 1950
F 0 "R5" H 4618 1996 50  0000 L CNN
F 1 "5.0k" H 4618 1905 50  0000 L CNN
F 2 "passives:R_0805_2012Metric_Pad1.15x1.40mm_HandSolder" V 4590 1940 50  0001 C CNN
F 3 "~" H 4550 1950 50  0001 C CNN
	1    4550 1950
	1    0    0    -1  
$EndComp
$Comp
L Device:R_US R3
U 1 1 5E26A316
P 3400 1350
F 0 "R3" V 3200 1300 50  0000 L CNN
F 1 "R_US" V 3300 1300 50  0000 L CNN
F 2 "passives:R_0805_2012Metric_Pad1.15x1.40mm_HandSolder" V 3440 1340 50  0001 C CNN
F 3 "~" H 3400 1350 50  0001 C CNN
	1    3400 1350
	0    1    1    0   
$EndComp
$Comp
L Device:R_US R4
U 1 1 5E26A52F
P 3400 1750
F 0 "R4" V 3500 1700 50  0000 C CNN
F 1 "R_US" V 3600 1750 50  0000 C CNN
F 2 "passives:R_0805_2012Metric_Pad1.15x1.40mm_HandSolder" V 3440 1740 50  0001 C CNN
F 3 "~" H 3400 1750 50  0001 C CNN
	1    3400 1750
	0    1    1    0   
$EndComp
Wire Wire Line
	3200 1350 3250 1350
Wire Wire Line
	3200 1750 3250 1750
Wire Wire Line
	2350 1650 2400 1650
Wire Wire Line
	2400 1650 2400 2050
Wire Wire Line
	4050 1950 4050 2200
Wire Wire Line
	4550 2200 4550 2100
Wire Wire Line
	4450 1750 4550 1750
Wire Wire Line
	4550 1750 4550 1800
$Comp
L power:GND #PWR02
U 1 1 5E26B14C
P 2400 2200
F 0 "#PWR02" H 2400 1950 50  0001 C CNN
F 1 "GND" H 2405 2027 50  0000 C CNN
F 2 "" H 2400 2200 50  0001 C CNN
F 3 "" H 2400 2200 50  0001 C CNN
	1    2400 2200
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR05
U 1 1 5E26B17E
P 2850 2200
F 0 "#PWR05" H 2850 1950 50  0001 C CNN
F 1 "GND" H 2855 2027 50  0000 C CNN
F 2 "" H 2850 2200 50  0001 C CNN
F 3 "" H 2850 2200 50  0001 C CNN
	1    2850 2200
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR013
U 1 1 5E26B1A9
P 4050 2200
F 0 "#PWR013" H 4050 1950 50  0001 C CNN
F 1 "GND" H 4055 2027 50  0000 C CNN
F 2 "" H 4050 2200 50  0001 C CNN
F 3 "" H 4050 2200 50  0001 C CNN
	1    4050 2200
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR017
U 1 1 5E26B1D4
P 4550 2200
F 0 "#PWR017" H 4550 1950 50  0001 C CNN
F 1 "GND" H 4555 2027 50  0000 C CNN
F 2 "" H 4550 2200 50  0001 C CNN
F 3 "" H 4550 2200 50  0001 C CNN
	1    4550 2200
	1    0    0    -1  
$EndComp
$Comp
L Device:D D5
U 1 1 5E26C4EE
P 7750 1500
F 0 "D5" V 7796 1421 50  0000 R CNN
F 1 "D" V 7705 1421 50  0000 R CNN
F 2 "passives:LED_1206_3216Metric_Pad1.42x1.75mm_HandSolder" H 7750 1500 50  0001 C CNN
F 3 "~" H 7750 1500 50  0001 C CNN
	1    7750 1500
	0    -1   -1   0   
$EndComp
Wire Wire Line
	7750 1350 7750 1300
$Comp
L power:+3.3V #PWR024
U 1 1 5E26D14C
P 7750 1300
F 0 "#PWR024" H 7750 1150 50  0001 C CNN
F 1 "+3.3V" H 7765 1473 50  0000 C CNN
F 2 "" H 7750 1300 50  0001 C CNN
F 3 "" H 7750 1300 50  0001 C CNN
	1    7750 1300
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR07
U 1 1 5E26E418
P 3450 4450
F 0 "#PWR07" H 3450 4200 50  0001 C CNN
F 1 "GND" H 3455 4277 50  0000 C CNN
F 2 "" H 3450 4450 50  0001 C CNN
F 3 "" H 3450 4450 50  0001 C CNN
	1    3450 4450
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR027
U 1 1 5E26E478
P 8600 1650
F 0 "#PWR027" H 8600 1400 50  0001 C CNN
F 1 "GND" H 8750 1550 50  0000 C CNN
F 2 "" H 8600 1650 50  0001 C CNN
F 3 "" H 8600 1650 50  0001 C CNN
	1    8600 1650
	1    0    0    -1  
$EndComp
Wire Wire Line
	8600 1650 8600 1550
Wire Wire Line
	8900 1300 8900 1600
Wire Wire Line
	8900 1600 8300 1600
$Comp
L power:GND #PWR025
U 1 1 5E26FDD6
P 7750 2900
F 0 "#PWR025" H 7750 2650 50  0001 C CNN
F 1 "GND" H 7755 2727 50  0000 C CNN
F 2 "" H 7750 2900 50  0001 C CNN
F 3 "" H 7750 2900 50  0001 C CNN
	1    7750 2900
	1    0    0    -1  
$EndComp
Wire Wire Line
	7750 2850 7750 2900
Wire Wire Line
	8150 2650 8650 2650
Wire Wire Line
	8150 2350 8550 2350
Wire Wire Line
	8150 2250 8550 2250
Wire Wire Line
	2350 1550 2600 1550
$Comp
L power:VBUS #PWR04
U 1 1 5E27347A
P 2850 1100
F 0 "#PWR04" H 2850 950 50  0001 C CNN
F 1 "VBUS" H 2865 1273 50  0000 C CNN
F 2 "" H 2850 1100 50  0001 C CNN
F 3 "" H 2850 1100 50  0001 C CNN
	1    2850 1100
	1    0    0    -1  
$EndComp
$Comp
L Device:C_Small C1
U 1 1 5E273E51
P 2600 1850
F 0 "C1" H 2692 1896 50  0000 L CNN
F 1 "10u" H 2692 1805 50  0000 L CNN
F 2 "passives:C_0805_2012Metric_Pad1.15x1.40mm_HandSolder" H 2600 1850 50  0001 C CNN
F 3 "~" H 2600 1850 50  0001 C CNN
	1    2600 1850
	1    0    0    -1  
$EndComp
Wire Wire Line
	2600 1750 2600 1550
Connection ~ 2600 1550
Wire Wire Line
	2600 1550 3650 1550
$Comp
L Device:C_Small C4
U 1 1 5E275AB7
P 4250 1200
F 0 "C4" V 4021 1200 50  0000 C CNN
F 1 "10u" V 4112 1200 50  0000 C CNN
F 2 "passives:C_0805_2012Metric_Pad1.15x1.40mm_HandSolder" H 4250 1200 50  0001 C CNN
F 3 "~" H 4250 1200 50  0001 C CNN
	1    4250 1200
	0    1    1    0   
$EndComp
$Comp
L power:GND #PWR016
U 1 1 5E275BFE
P 4450 1250
F 0 "#PWR016" H 4450 1000 50  0001 C CNN
F 1 "GND" H 4455 1077 50  0000 C CNN
F 2 "" H 4450 1250 50  0001 C CNN
F 3 "" H 4450 1250 50  0001 C CNN
	1    4450 1250
	1    0    0    -1  
$EndComp
Wire Wire Line
	2600 1950 2600 2050
Wire Wire Line
	2600 2050 2400 2050
Connection ~ 2400 2050
Wire Wire Line
	2400 2050 2400 2200
Wire Wire Line
	4050 1350 4050 1200
Wire Wire Line
	4050 1200 4150 1200
Wire Wire Line
	4350 1200 4450 1200
Wire Wire Line
	4450 1200 4450 1250
Wire Wire Line
	4050 1200 4050 1100
Connection ~ 4050 1200
$Comp
L power:VBUS #PWR012
U 1 1 5E27796B
P 4050 1100
F 0 "#PWR012" H 4050 950 50  0001 C CNN
F 1 "VBUS" H 4065 1273 50  0000 C CNN
F 2 "" H 4050 1100 50  0001 C CNN
F 3 "" H 4050 1100 50  0001 C CNN
	1    4050 1100
	1    0    0    -1  
$EndComp
Wire Wire Line
	3550 1750 3600 1750
Wire Wire Line
	2850 1750 2900 1750
Wire Wire Line
	2850 1750 2850 2200
Wire Wire Line
	3550 1350 3600 1350
Connection ~ 3600 1750
Wire Wire Line
	3600 1750 3650 1750
Wire Wire Line
	3600 1350 3600 1750
Wire Wire Line
	2900 1350 2850 1350
Wire Wire Line
	2850 1350 2850 1100
Text Notes 4750 2450 0    50   ~ 0
10.k = 100 mA\n5.0k = 200 mA\n2.0k = 500 mA\n1.0k = 1.0 A
Wire Wire Line
	7750 1650 7750 1700
Wire Wire Line
	8050 1350 8050 1300
Wire Wire Line
	8300 1300 8300 1600
Connection ~ 8300 1300
Wire Wire Line
	8050 1650 8050 1700
Wire Wire Line
	8050 1700 7750 1700
Connection ~ 7750 1700
Wire Wire Line
	7750 1700 7750 2050
$Comp
L Device:C_Small C7
U 1 1 5E280A82
P 8050 1850
F 0 "C7" H 7850 1900 50  0000 L CNN
F 1 "0.1u" H 7850 1800 50  0000 L CNN
F 2 "passives:C_0805_2012Metric_Pad1.15x1.40mm_HandSolder" H 8050 1850 50  0001 C CNN
F 3 "~" H 8050 1850 50  0001 C CNN
	1    8050 1850
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR026
U 1 1 5E280BE6
P 8050 2000
F 0 "#PWR026" H 8050 1750 50  0001 C CNN
F 1 "GND" H 8150 1900 50  0000 C CNN
F 2 "" H 8050 2000 50  0001 C CNN
F 3 "" H 8050 2000 50  0001 C CNN
	1    8050 2000
	1    0    0    -1  
$EndComp
Wire Wire Line
	8050 2000 8050 1950
Wire Wire Line
	8050 1750 8050 1700
Connection ~ 8050 1700
Wire Wire Line
	8050 1300 8300 1300
Wire Wire Line
	8150 2550 8650 2550
Wire Wire Line
	3950 3950 4500 3950
Wire Wire Line
	3950 3850 4500 3850
Wire Wire Line
	2950 3450 2350 3450
Wire Wire Line
	3450 3150 3450 3100
Wire Wire Line
	3450 3100 3550 3100
Wire Wire Line
	3550 3100 3550 3150
Wire Wire Line
	3550 3100 3550 2900
Connection ~ 3550 3100
Wire Wire Line
	3550 2900 3850 2900
Connection ~ 3550 2900
Wire Wire Line
	3550 2900 3550 2850
$Comp
L Device:C_Small C3
U 1 1 5E2BA589
P 4050 3050
F 0 "C3" H 4142 3096 50  0000 L CNN
F 1 "10u" H 4142 3005 50  0000 L CNN
F 2 "passives:C_0805_2012Metric_Pad1.15x1.40mm_HandSolder" H 4050 3050 50  0001 C CNN
F 3 "~" H 4050 3050 50  0001 C CNN
	1    4050 3050
	1    0    0    -1  
$EndComp
Wire Wire Line
	4050 2900 4050 2950
$Comp
L power:GND #PWR014
U 1 1 5E2BB51B
P 4050 3200
F 0 "#PWR014" H 4050 2950 50  0001 C CNN
F 1 "GND" H 4055 3027 50  0000 C CNN
F 2 "" H 4050 3200 50  0001 C CNN
F 3 "" H 4050 3200 50  0001 C CNN
	1    4050 3200
	1    0    0    -1  
$EndComp
Wire Wire Line
	4050 3200 4050 3150
Wire Wire Line
	3450 4450 3450 4350
$Comp
L power:+3.3V #PWR09
U 1 1 5E2BD5B7
P 3550 2850
F 0 "#PWR09" H 3550 2700 50  0001 C CNN
F 1 "+3.3V" H 3565 3023 50  0000 C CNN
F 2 "" H 3550 2850 50  0001 C CNN
F 3 "" H 3550 2850 50  0001 C CNN
	1    3550 2850
	1    0    0    -1  
$EndComp
$Comp
L Device:C_Small C2
U 1 1 5E2BE68E
P 3850 3050
F 0 "C2" H 3650 3100 50  0000 L CNN
F 1 "0.1u" H 3650 3000 50  0000 L CNN
F 2 "passives:C_0805_2012Metric_Pad1.15x1.40mm_HandSolder" H 3850 3050 50  0001 C CNN
F 3 "~" H 3850 3050 50  0001 C CNN
	1    3850 3050
	1    0    0    -1  
$EndComp
Wire Wire Line
	3850 2950 3850 2900
Connection ~ 3850 2900
Wire Wire Line
	3850 2900 4050 2900
Wire Wire Line
	3850 3150 4050 3150
Connection ~ 4050 3150
$Comp
L Device:R_US R2
U 1 1 5E2C0837
P 2850 3150
F 0 "R2" H 2650 3200 50  0000 L CNN
F 1 "10k" H 2650 3100 50  0000 L CNN
F 2 "passives:R_0805_2012Metric_Pad1.15x1.40mm_HandSolder" V 2890 3140 50  0001 C CNN
F 3 "~" H 2850 3150 50  0001 C CNN
	1    2850 3150
	1    0    0    -1  
$EndComp
Wire Wire Line
	2850 3300 2850 3350
Wire Wire Line
	2850 3350 2950 3350
Wire Wire Line
	2850 3000 2850 2900
Wire Wire Line
	2850 2900 3550 2900
Wire Wire Line
	2100 4050 2100 3800
$Comp
L power:+3.3V #PWR01
U 1 1 5E2C847D
P 2100 3800
F 0 "#PWR01" H 2100 3650 50  0001 C CNN
F 1 "+3.3V" H 2115 3973 50  0000 C CNN
F 2 "" H 2100 3800 50  0001 C CNN
F 3 "" H 2100 3800 50  0001 C CNN
	1    2100 3800
	1    0    0    -1  
$EndComp
$Comp
L power:+BATT #PWR03
U 1 1 5E2C89D0
P 2600 1450
F 0 "#PWR03" H 2600 1300 50  0001 C CNN
F 1 "+BATT" H 2615 1623 50  0000 C CNN
F 2 "" H 2600 1450 50  0001 C CNN
F 3 "" H 2600 1450 50  0001 C CNN
	1    2600 1450
	1    0    0    -1  
$EndComp
Wire Wire Line
	2600 1550 2600 1450
Wire Wire Line
	3950 4050 4050 4050
Wire Wire Line
	4050 4050 4050 3750
$Comp
L power:+3.3V #PWR015
U 1 1 5E2CC797
P 4050 3750
F 0 "#PWR015" H 4050 3600 50  0001 C CNN
F 1 "+3.3V" H 4065 3923 50  0000 C CNN
F 2 "" H 4050 3750 50  0001 C CNN
F 3 "" H 4050 3750 50  0001 C CNN
	1    4050 3750
	1    0    0    -1  
$EndComp
Text Label 4500 3850 2    50   ~ 0
I2C_SDA
Text Label 4500 3950 2    50   ~ 0
I2C_SCL
NoConn ~ 2950 3750
NoConn ~ 2950 3650
NoConn ~ 3950 3650
NoConn ~ 3950 3550
NoConn ~ 3950 3450
NoConn ~ 2950 3950
$Comp
L Device:R_US R1
U 1 1 5E2D5889
P 2650 4050
F 0 "R1" V 2750 4000 50  0000 C CNN
F 1 "R_US" V 2850 4050 50  0000 C CNN
F 2 "passives:R_0805_2012Metric_Pad1.15x1.40mm_HandSolder" V 2690 4040 50  0001 C CNN
F 3 "~" H 2650 4050 50  0001 C CNN
	1    2650 4050
	0    1    1    0   
$EndComp
Wire Wire Line
	2400 4050 2500 4050
Wire Wire Line
	2800 4050 2950 4050
Text Label 8250 2350 0    50   ~ 0
I2C_SDA
Text Label 8250 2250 0    50   ~ 0
I2C_SCL
$Comp
L Connector:Micro_SD_Card J4
U 1 1 5E2DF102
P 8700 4600
F 0 "J4" H 8000 5300 50  0000 L CNN
F 1 "Micro_SD_Card" H 8000 5200 50  0000 L CNN
F 2 "sense-trinket:sd_molex_1051620001" H 9850 4900 50  0001 C CNN
F 3 "http://katalog.we-online.de/em/datasheet/693072010801.pdf" H 8700 4600 50  0001 C CNN
	1    8700 4600
	1    0    0    -1  
$EndComp
Wire Wire Line
	7800 4600 7700 4600
Wire Wire Line
	7700 4600 7700 4050
$Comp
L Device:C_Small C6
U 1 1 5E2E0A84
P 7550 3850
F 0 "C6" H 7350 3900 50  0000 L CNN
F 1 "0.1u" H 7350 3800 50  0000 L CNN
F 2 "passives:C_0805_2012Metric_Pad1.15x1.40mm_HandSolder" H 7550 3850 50  0001 C CNN
F 3 "~" H 7550 3850 50  0001 C CNN
	1    7550 3850
	1    0    0    -1  
$EndComp
$Comp
L Device:C_Small C8
U 1 1 5E2E0E2A
P 8250 1850
F 0 "C8" H 8342 1896 50  0000 L CNN
F 1 "10u" H 8342 1805 50  0000 L CNN
F 2 "passives:C_0805_2012Metric_Pad1.15x1.40mm_HandSolder" H 8250 1850 50  0001 C CNN
F 3 "~" H 8250 1850 50  0001 C CNN
	1    8250 1850
	1    0    0    -1  
$EndComp
Wire Wire Line
	8050 1700 8250 1700
Wire Wire Line
	8250 1700 8250 1750
Wire Wire Line
	8250 1950 8050 1950
Connection ~ 8050 1950
$Comp
L power:GND #PWR020
U 1 1 5E2E788D
P 7200 4000
F 0 "#PWR020" H 7200 3750 50  0001 C CNN
F 1 "GND" H 7100 3850 50  0000 C CNN
F 2 "" H 7200 4000 50  0001 C CNN
F 3 "" H 7200 4000 50  0001 C CNN
	1    7200 4000
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR023
U 1 1 5E2E78CC
P 7700 5150
F 0 "#PWR023" H 7700 4900 50  0001 C CNN
F 1 "GND" H 7705 4977 50  0000 C CNN
F 2 "" H 7700 5150 50  0001 C CNN
F 3 "" H 7700 5150 50  0001 C CNN
	1    7700 5150
	1    0    0    -1  
$EndComp
Wire Wire Line
	7800 4800 7700 4800
Wire Wire Line
	7700 4800 7700 5150
Wire Wire Line
	7550 3750 7550 3700
Wire Wire Line
	7550 3700 7700 3700
Connection ~ 7700 3700
Wire Wire Line
	7700 3700 7700 3650
$Comp
L power:+3.3V #PWR022
U 1 1 5E2ECC0F
P 7700 3650
F 0 "#PWR022" H 7700 3500 50  0001 C CNN
F 1 "+3.3V" H 7715 3823 50  0000 C CNN
F 2 "" H 7700 3650 50  0001 C CNN
F 3 "" H 7700 3650 50  0001 C CNN
	1    7700 3650
	1    0    0    -1  
$EndComp
Wire Wire Line
	7250 4900 7800 4900
Wire Wire Line
	7800 4500 7250 4500
Wire Wire Line
	7800 4400 7550 4400
Wire Wire Line
	7800 4700 7250 4700
NoConn ~ 7800 5000
NoConn ~ 7800 4300
$Comp
L Device:C_Small C5
U 1 1 5E2FE732
P 7200 3850
F 0 "C5" H 7000 3900 50  0000 L CNN
F 1 "10u" H 7000 3800 50  0000 L CNN
F 2 "passives:C_0805_2012Metric_Pad1.15x1.40mm_HandSolder" H 7200 3850 50  0001 C CNN
F 3 "~" H 7200 3850 50  0001 C CNN
	1    7200 3850
	1    0    0    -1  
$EndComp
Wire Wire Line
	7200 3750 7200 3700
Wire Wire Line
	7200 3700 7550 3700
Connection ~ 7550 3700
Wire Wire Line
	7200 4000 7200 3950
Wire Wire Line
	7550 3950 7200 3950
Connection ~ 7200 3950
$Comp
L Device:R_US R9
U 1 1 5E30BA1C
P 7550 4200
F 0 "R9" H 7350 4250 50  0000 L CNN
F 1 "100k" H 7350 4150 50  0000 L CNN
F 2 "passives:R_0805_2012Metric_Pad1.15x1.40mm_HandSolder" V 7590 4190 50  0001 C CNN
F 3 "~" H 7550 4200 50  0001 C CNN
	1    7550 4200
	1    0    0    -1  
$EndComp
Wire Wire Line
	7550 4050 7700 4050
Connection ~ 7700 4050
Wire Wire Line
	7700 4050 7700 3700
Wire Wire Line
	7550 4350 7550 4400
Connection ~ 7550 4400
Wire Wire Line
	6350 4400 6650 4400
Wire Wire Line
	6650 4450 6650 4400
Connection ~ 6650 4400
Wire Wire Line
	6650 4400 7550 4400
$Comp
L Device:R_US R8
U 1 1 5E3156EB
P 6650 4950
F 0 "R8" H 6450 5000 50  0000 L CNN
F 1 "100k" H 6450 4900 50  0000 L CNN
F 2 "passives:R_0805_2012Metric_Pad1.15x1.40mm_HandSolder" V 6690 4940 50  0001 C CNN
F 3 "~" H 6650 4950 50  0001 C CNN
	1    6650 4950
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR019
U 1 1 5E31576D
P 6650 5150
F 0 "#PWR019" H 6650 4900 50  0001 C CNN
F 1 "GND" H 6655 4977 50  0000 C CNN
F 2 "" H 6650 5150 50  0001 C CNN
F 3 "" H 6650 5150 50  0001 C CNN
	1    6650 5150
	1    0    0    -1  
$EndComp
Wire Wire Line
	6650 5150 6650 5100
Wire Wire Line
	6650 4800 6650 4750
Text Label 7250 4900 0    50   ~ 0
SPI_MISO
Text Label 7250 4700 0    50   ~ 0
SPI_SCK
Text Label 7250 4500 0    50   ~ 0
SPI_MOSI
Text Label 6350 4400 0    50   ~ 0
SD_CS_N
Text Label 2350 3450 0    50   ~ 0
AS7262_INT
Text Label 8650 2650 2    50   ~ 0
RTC_INT_N
Text Label 8650 2550 2    50   ~ 0
RTC_CLK
$Comp
L power:GND #PWR028
U 1 1 5E31C12B
P 9500 5750
F 0 "#PWR028" H 9500 5500 50  0001 C CNN
F 1 "GND" H 9505 5577 50  0000 C CNN
F 2 "" H 9500 5750 50  0001 C CNN
F 3 "" H 9500 5750 50  0001 C CNN
	1    9500 5750
	1    0    0    -1  
$EndComp
$Comp
L Device:R_US R10
U 1 1 5E31EBDB
P 9500 5500
F 0 "R10" H 9300 5550 50  0000 L CNN
F 1 "0" H 9300 5450 50  0000 L CNN
F 2 "passives:R_0805_2012Metric_Pad1.15x1.40mm_HandSolder" V 9540 5490 50  0001 C CNN
F 3 "~" H 9500 5500 50  0001 C CNN
	1    9500 5500
	1    0    0    -1  
$EndComp
$Comp
L Device:C_Small C9
U 1 1 5E31EC93
P 9650 5500
F 0 "C9" H 9750 5550 50  0000 L CNN
F 1 "DNP" H 9750 5450 50  0000 L CNN
F 2 "passives:C_0805_2012Metric_Pad1.15x1.40mm_HandSolder" H 9650 5500 50  0001 C CNN
F 3 "~" H 9650 5500 50  0001 C CNN
	1    9650 5500
	1    0    0    -1  
$EndComp
Wire Wire Line
	9500 5200 9500 5350
Wire Wire Line
	9500 5350 9650 5350
Wire Wire Line
	9650 5350 9650 5400
Connection ~ 9500 5350
Wire Wire Line
	9650 5600 9650 5650
Wire Wire Line
	9650 5650 9500 5650
Wire Wire Line
	9500 5750 9500 5650
Connection ~ 9500 5650
Text Notes 8050 5950 0    50   ~ 0
NOTE:\nPassives for connecting shield\nare probably not needed but \nplaced in case of EMI. Short \none of these!
Text Notes 8800 4650 0    50   ~ 0
MicroSD\n
Wire Wire Line
	3550 5350 3550 5150
$Comp
L power:+3.3V #PWR010
U 1 1 5E32D810
P 3550 5150
F 0 "#PWR010" H 3550 5000 50  0001 C CNN
F 1 "+3.3V" H 3600 5300 50  0000 C CNN
F 2 "" H 3550 5150 50  0001 C CNN
F 3 "" H 3550 5150 50  0001 C CNN
	1    3550 5150
	1    0    0    -1  
$EndComp
Wire Wire Line
	3350 5350 3350 5150
Wire Wire Line
	2950 6950 2350 6950
Wire Wire Line
	2950 6850 2350 6850
Text Label 2350 6950 0    50   ~ 0
SPI_MISO
Text Label 2350 6850 0    50   ~ 0
SPI_MOSI
$Comp
L power:+BATT #PWR06
U 1 1 5E3371A0
P 3350 5150
F 0 "#PWR06" H 3350 5000 50  0001 C CNN
F 1 "+BATT" H 3300 5300 50  0000 C CNN
F 2 "" H 3350 5150 50  0001 C CNN
F 3 "" H 3350 5150 50  0001 C CNN
	1    3350 5150
	1    0    0    -1  
$EndComp
Wire Wire Line
	3650 5350 3650 5300
$Comp
L power:VBUS #PWR011
U 1 1 5E33A4AD
P 3650 5300
F 0 "#PWR011" H 3650 5150 50  0001 C CNN
F 1 "VBUS" H 3700 5450 50  0000 C CNN
F 2 "" H 3650 5300 50  0001 C CNN
F 3 "" H 3650 5300 50  0001 C CNN
	1    3650 5300
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR08
U 1 1 5E34A8ED
P 3500 7450
F 0 "#PWR08" H 3500 7200 50  0001 C CNN
F 1 "GND" H 3505 7277 50  0000 C CNN
F 2 "" H 3500 7450 50  0001 C CNN
F 3 "" H 3500 7450 50  0001 C CNN
	1    3500 7450
	1    0    0    -1  
$EndComp
Wire Wire Line
	3500 7450 3500 7400
Wire Wire Line
	3500 7400 3450 7400
Wire Wire Line
	3450 7400 3450 7350
Wire Wire Line
	3500 7400 3550 7400
Wire Wire Line
	3550 7400 3550 7350
Connection ~ 3500 7400
Wire Wire Line
	2950 7050 2350 7050
Text Label 2350 7050 0    50   ~ 0
SPI_SCK
Wire Wire Line
	2950 6750 2350 6750
Text Label 2350 6750 0    50   ~ 0
SD_CS_N
Text Label 4500 6750 2    50   ~ 0
I2C_SDA
Text Label 4500 6850 2    50   ~ 0
I2C_SCL
Wire Wire Line
	2950 5950 2350 5950
Wire Wire Line
	2950 6050 2350 6050
Text Label 2350 6050 0    50   ~ 0
AS7262_INT
Text Label 2350 5950 0    50   ~ 0
RTC_INT_N
Wire Wire Line
	2950 6650 2350 6650
Text Label 2350 6650 0    50   ~ 0
RTC_CLK
$Comp
L Device:R_US R6
U 1 1 5E36D0AE
P 4600 6500
F 0 "R6" H 4400 6550 50  0000 L CNN
F 1 "5.0k" H 4400 6450 50  0000 L CNN
F 2 "passives:R_0805_2012Metric_Pad1.15x1.40mm_HandSolder" V 4640 6490 50  0001 C CNN
F 3 "~" H 4600 6500 50  0001 C CNN
	1    4600 6500
	1    0    0    -1  
$EndComp
$Comp
L Device:R_US R7
U 1 1 5E36D1DE
P 4750 6600
F 0 "R7" H 4818 6646 50  0000 L CNN
F 1 "5.0k" H 4818 6555 50  0000 L CNN
F 2 "passives:R_0805_2012Metric_Pad1.15x1.40mm_HandSolder" V 4790 6590 50  0001 C CNN
F 3 "~" H 4750 6600 50  0001 C CNN
	1    4750 6600
	1    0    0    -1  
$EndComp
Wire Wire Line
	4600 6750 4600 6650
Wire Wire Line
	3950 6750 4600 6750
Wire Wire Line
	4750 6850 4750 6750
Wire Wire Line
	3950 6850 4750 6850
Wire Wire Line
	4750 6450 4750 6300
Wire Wire Line
	4750 6300 4650 6300
Wire Wire Line
	4600 6300 4600 6350
Wire Wire Line
	4650 6300 4650 6150
Connection ~ 4650 6300
Wire Wire Line
	4650 6300 4600 6300
$Comp
L power:+3.3V #PWR018
U 1 1 5E38B75A
P 4650 6150
F 0 "#PWR018" H 4650 6000 50  0001 C CNN
F 1 "+3.3V" H 4700 6300 50  0000 C CNN
F 2 "" H 4650 6150 50  0001 C CNN
F 3 "" H 4650 6150 50  0001 C CNN
	1    4650 6150
	1    0    0    -1  
$EndComp
NoConn ~ 2950 6550
NoConn ~ 2950 6450
NoConn ~ 2950 6350
NoConn ~ 2950 6250
NoConn ~ 2950 6150
NoConn ~ 2950 5850
NoConn ~ 2950 5750
NoConn ~ 3950 5750
NoConn ~ 3950 5850
NoConn ~ 3950 6150
NoConn ~ 3950 6350
NoConn ~ 3950 6450
NoConn ~ 3950 6550
NoConn ~ 3950 6650
NoConn ~ 3950 6950
NoConn ~ 3950 7050
$Comp
L Device:Crystal_GND23 Y1
U 1 1 5E4151DF
P 7100 2450
F 0 "Y1" H 7500 2350 50  0000 L CNN
F 1 "32.768k" H 7300 2450 50  0000 L CNN
F 2 "sense-trinket:XTL_4pin_MC-146" H 7100 2450 50  0001 C CNN
F 3 "~" H 7100 2450 50  0001 C CNN
	1    7100 2450
	-1   0    0    1   
$EndComp
Wire Wire Line
	7100 2650 7350 2650
Wire Wire Line
	7100 2250 7350 2250
Wire Wire Line
	6950 2450 6950 2700
Wire Wire Line
	6950 2700 7250 2700
Wire Wire Line
	7250 2700 7250 2450
Wire Wire Line
	7250 2700 7250 2900
Connection ~ 7250 2700
$Comp
L power:GND #PWR021
U 1 1 5E427B0C
P 7250 2900
F 0 "#PWR021" H 7250 2650 50  0001 C CNN
F 1 "GND" H 7255 2727 50  0000 C CNN
F 2 "" H 7250 2900 50  0001 C CNN
F 3 "" H 7250 2900 50  0001 C CNN
	1    7250 2900
	1    0    0    -1  
$EndComp
Text Label 8900 1600 0    50   ~ 0
VBATT_BACKUP
$EndSCHEMATC
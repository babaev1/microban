# motor velicity measurement
import math
takt = 72000000 # frequesncy of stm32 in motor
divider = 35000 
time_slot = divider/takt
measured_ticks = 1
measured_speed = measured_ticks / time_slot/ (2**21) * 2 * math.pi
print('measured_speed :', measured_speed)
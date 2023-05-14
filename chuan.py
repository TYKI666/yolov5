import serial

# ser为串口对象，后续调用均用点运算符
ser = serial.Serial('COM10', 3000000, 8, 'N', 1) # 'COM7', 3000000, bytesize=8, parity='N', stopbits=1
flag = ser.is_open

if flag:
	print('success\n')
	ser.close()
else:
	print('Open Error\n')

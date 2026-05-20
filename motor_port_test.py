import serial
import time

PORT = "/dev/ttyACM0"
BAUD = 1000000

print("START", flush=True)
print(f"Opening {PORT} ...", flush=True)

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("Serial open OK", flush=True)

    time.sleep(1)

    print("Port info:")
    print("name:", ser.name)
    print("baudrate:", ser.baudrate)
    print("is_open:", ser.is_open)

    ser.close()
    print("DONE", flush=True)

except Exception as e:
    print("FAILED:", repr(e), flush=True)

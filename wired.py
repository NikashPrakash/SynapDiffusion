import serial
import serial.tools.list_ports
import time

manual = True

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if 'Arduino' in p.description:
            return p.device
    return ports[0].device if ports else None

def send_frame_serial(ser, frame_number):
    ser.write(str(frame_number).encode())
    time.sleep(0.1)  # Short delay to ensure transmission
    if ser.in_waiting:
        response = ser.readline().decode().strip()
        print(f"Arduino response: {response}")

def main():
    print("Serial Arduino LED Matrix Control")
    port = find_arduino_port()
    if port is None:
        print("No Arduino found. Please check the connection.")
        return
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)  # Allow time for the serial connection to establish
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return

    print("Enter '0' or '1' to change the frame")
    print("Enter 'q' to quit")

    if manual:
        try:
            while True:
                user_input = input("Enter command (0, 1, or q): ").strip().lower()
                if user_input == 'q':
                    print("Quitting program...")
                    break
                elif user_input in ['0', '1']:
                    send_frame_serial(ser, user_input)
                else:
                    print("Invalid input. Please enter 0, 1, or q.")
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
        finally:
            if 'ser' in locals():
                ser.close()
            print("Goodbye!")
    else: # TODO call predict function of model
        pass

if __name__ == "__main__":
    main()
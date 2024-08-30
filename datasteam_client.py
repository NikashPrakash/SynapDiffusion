manual = True

def main():
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
                    send_frame_wifi(user_input)
                else:
                    print("Invalid input. Please enter 0, 1, or q.")
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
        finally:
            print("Goodbye!")
        pass
import threading
import time
def my_thread(n):
    time.sleep(0.8)
    print("Thread is running:",n)

# Main program continues to run
i = 0
while True:
    # Add your code here
    i += 1
    thread = threading.Thread(target=my_thread, args=(i,))
    # Start the thread
    thread.start()
    time.sleep(1)
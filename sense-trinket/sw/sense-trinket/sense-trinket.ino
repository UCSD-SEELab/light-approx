/*
 * sense-trinket
 *
 * Description
 *
 */

void clock_input_cb() {
  // When an interrupt comes in, check the current number of ticks

  // If number of ticks is divisible by sample rate, then capture data

  // Otherwise, do nothing

  // In either case, reset watchdog timer
}

void setup() {
  // Initialize RTC and get current time
  // When initializing RTC, set up interrupt input from clock

  // Initialize light sensor

  // Initialize SD card read/write

  // Initialize watchdog timer

}

void loop() {
  // If USB is connected, start a serial connection

  // Process serial connection, including taking in current time information, system name, etc

  // Once serial connection is closed or if it was never connected, then...
  
  // Wait for interrupt from RTC clock or watchdog timer

  // If watchdog timer triggers, reset light sensor/SD card/entire system, perhaps save a log entry
  
}

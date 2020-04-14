
/*
 * sense-trinket
 *
 * Description
 * RTC PCF85063A https://github.com/jvsalo/pcf85063a
 */
 
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <TimeLib.h>
#include "Adafruit_AS726x.h"
#include <RTClib.h>
#include <avr/wdt.h>

const int chipSelect = 4;
const unsigned long SAMPLING_RATE = 60; //1 minute
const unsigned long TIMEOUTPERIOD = 10000;

Adafruit_AS726x ams; // Initialize light sensor
uint16_t sensorValues[AS726x_NUM_CHANNELS]; //buffer to hold raw values
PCF85063A rtc; // real time clock

void clock_input_cb() {
    
    ams.startMeasurement(); //begin a measurement
    
    //wait till data is available
    bool rdy = false;
    while(!rdy){
      delay(5);
      rdy = ams.dataReady();
    }
    
    ams.readRawValues(sensorValues);
    
    tmElements_t timeStamp;
    rtc.time_get(&timeStamp);
    
    String dataString = String(timeStamp) + "," + String(sensorValues[AS726x_VIOLET]) + "," + String(sensorValues[AS726x_BLUE]) + 
                        "," + String(sensorValues[AS726x_GREEN]) + + "," + String(sensorValues[AS726x_YELLOW]) + + "," + 
                        String(sensorValues[AS726x_ORANGE]) + "," + String(sensorValues[AS726x_RED]);
    
    
    File dataFile = SD.open("data.txt", FILE_WRITE);
    // if the file is available, write to it:
    if (dataFile) {
      dataFile.println(dataString);
      dataFile.close();
    }
    // if the file isn't open, pop up an error:
    else {
      Serial.println("error opening datalog.txt");
    }

  // Reset watchdog timer
  wdt_reset();
}

void setup() {
  // Initialize RTC
  Serial.begin(57600);
  rtc.reset();  
  
  // Enabling repeating countdown timer with 60 seconds interval (interrupt)
  rtc.countdown_set(true, PCF85063A::CNTDOWN_CLOCK_1HZ, SAMPLING_RATE, false, false);
  
  if(!ams.begin()){
    Serial.println("could not connect to sensor! Please check your wiring.");
    while(1);
  }
  
  // Initialize SD card read/write
  while (!SD.begin(chipSelect)) {
    Serial.println("Card failed, or not present");
  }
  Serial.println("card initialized.");
  
  // Initialize watchdog timer
  wdt_enable(WDTO_8S);
  
  // Initialize USBDetect
  pinMode(13,OUTPUT);
  USBCON|=(1<<OTGPADE); //enables VBUS pad
}

void loop() {
  // If USB is connected, start a serial connection
  if(USBSTA&(1<<VBUS)){  //checks state of VBUS
      if(!Serial)
         Serial.begin(9600);
      digitalWrite(13,HIGH);
      Serial.println("Serial is connected");
      // Process serial connection, including taking in current time information, system name, etc
      
   }
  // Once serial connection is closed or if it was never connected, then...
  else {
    if(Serial)
       Serial.end();
    digitalWrite(13,LOW);
  }
   
   //delay(1000);
  
  // Wait for interrupt from RTC clock or watchdog timer
  while (1) {
    wdt_reset()
  }
  // If watchdog timer triggers, reset light sensor/SD card/entire system, perhaps save a log entry
  
  File errorFile = SD.open("error.txt", FILE_WRITE);
  // if the file is available, write to it:
  if (errorFile) {
    tmElements_t timeStamp;
    rtc.time_get(&timeStamp);
    errorString = timeStamp + " Watchdog timer triggered, system resetting";
    errorFile.println(errorString);
    errorFile.close();
    // print to the serial port too:
    Serial.println(errorString);
  }
  // if the file isn't open, pop up an error:
  else {
    Serial.println("error opening datalog.txt");
  }
  
}
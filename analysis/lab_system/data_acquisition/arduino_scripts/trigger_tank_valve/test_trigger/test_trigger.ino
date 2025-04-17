int pos = 0;          // Initial ball valve position.
int runLoop = 0;      // Run loop flag.
int triggerPin = A3;  // Pin to read trigger signal.
int valvePin = 9;     // Pin to control valve.
int val = 0;          // Value to store analog signal.
int trigger = 500;    // Trigger value above which Arduino will run the valve code.

void setup() {
  Serial.begin(9600);
  analogWrite(valvePin, 0);  // Start the valve at closed.
}

void loop() {
  // Read trigger from the DAQ. Analog voltage signal of 1 V.
  val = analogRead(triggerPin);  // read the input pin
  
  if (val > trigger & runLoop == 0) {
    runLoop = 1;
    Serial.println(analogRead(triggerPin));   // debug value
  }
  
  if (runLoop == 0) {
    Serial.println(analogRead(triggerPin));   // debug value
  }

}
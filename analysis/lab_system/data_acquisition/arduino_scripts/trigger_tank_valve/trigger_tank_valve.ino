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
  Serial.println(analogRead(triggerPin));   // debug value

  if (val > trigger) {
    delay(2000); // Wait a bit for data collection to start.
    runLoop = 1;
  }

  if (runLoop == 1) {
    analogWrite(valvePin, 255);
    delay(8000); // How long to keep valve open.
    analogWrite(valvePin, 0);
  }
  runLoop = 0;
}
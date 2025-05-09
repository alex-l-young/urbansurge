int trialLength = 8000; // How long to keep valve1 open (ms). 
int pos = 0;          // Initial ball valve position.
int runLoop = 0;      // Run loop flag.
int triggerPin = A3;  // Pin to read trigger signal.
int valve1Pin = 9;     // Pin to control valve1 (tank to system).
int valve2Pin = 10;    // Pin to control valve2 (tank filling).
int val = 0;          // Value to store analog signal for valve1.
int val2 = 0;         // Value to store analog signal for valve2.
int trigger = 500;    // Trigger value above which Arduino will run the valve code.

void setup() {
  Serial.begin(9600);
  analogWrite(valve1Pin, 0);  // Start valve1 at closed.
  analogWrite(valve2Pin, 0);  // Start valve2 at closed.
}

void loop() {
  // Read trigger from the DAQ. Analog voltage signal of 1 V.
  val = analogRead(triggerPin);  // read the input pin

  if (val > trigger) {
    delay(2000); // Wait a bit for data collection to start.
    analogWrite(valve2Pin, 255);
    Serial.println("VALVE2 OPENED");   // debug value

    val2 = analogRead(triggerPin);
    while (val2 < trigger) {   // wait until signal for closing valve2 is received.
      val2 = analogRead(triggerPin);
    }
    analogWrite(valve2Pin, 0);  // close valve2.
    Serial.println("VALVE2 CLOSED");
    val2 = 0;

    analogWrite(valve1Pin, 255);  // open valve1.
    Serial.println("VALVE1 OPENED");
    delay(trialLength); // How long to keep valve open.
    analogWrite(valve1Pin, 0);    // close valve1. 
    Serial.println("VALVE1 CLOSED");
  }

  val = 0; // make sure the valve closes next run.
}
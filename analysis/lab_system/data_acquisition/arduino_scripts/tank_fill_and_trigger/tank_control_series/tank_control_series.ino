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
    analogWrite(valvePin, 160);
    delay(3137); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 240);
    delay(4706); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 219);
    delay(4294); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 215);
    delay(4216); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 88);
    delay(1725); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 161);
    delay(3157); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 107);
    delay(2098); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 134);
    delay(2627); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 218);
    delay(4275); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 226);
    delay(4431); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 247);
    delay(4843); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 203);
    delay(3980); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 159);
    delay(3118); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 213);
    delay(4176); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 202);
    delay(3961); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 189);
    delay(3706); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 98);
    delay(1922); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 167);
    delay(3275); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
    analogWrite(valvePin, 174);
    delay(3412); // Time at value.
    analogWrite(valvePin, 0);
    delay(30000); // Time at value.
  }
  runLoop = 0;
}
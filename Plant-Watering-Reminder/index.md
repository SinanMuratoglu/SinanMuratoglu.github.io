---
title: Plant Watering Reminder
---

# Plant Watering Reminder

A small device that keeps an eye on a plant's soil and, when it's dry, gets the attention of whoever walks by. Built with a Raspberry Pi Pico 2 running CircuitPython for Smart Electronics in Design.

## How it works

The system has two operating modes.

**Soil probing mode.** Once a day, the Pico reads a capacitive soil moisture sensor. If the soil is still moist, it goes back to waiting until the next day's check.

**Alert mode.** If the soil is dry, the system switches to alert mode and waits for the ultrasonic sensor to detect someone nearby. When someone walks by, a servo rocks the watering can back and forth as a reminder to water the plant. A cooldown period follows, giving time for the plant to be watered before the system returns to soil probing mode.

## Hardware

- Raspberry Pi Pico 2 running CircuitPython
- Capacitive soil moisture sensor
- HC-SR04 ultrasonic distance sensor for motion detection
- 9 g micro servo driving a rocking platform for the watering can

## Sensing the soil

I chose a capacitive moisture sensor over a resistive one because its electrodes are sealed under the circuit board's coating, so no current passes through the soil and the probe doesn't corrode. The sensor's output drops as the soil gets wetter: water raises the capacitance around the probe, which lowers the board's output voltage. I calibrated it by recording readings in dry and wet soil and set the "dry" cutoff between them.

<!-- Video goes here -->

<!-- Photos go here -->

[← Back to projects](../)

---
title: Soil Sentinel ™
---

# The Soil Sentinel ™

Keeps an eye on a plant's soil and, when it's dry, gets the attention of whoever walks by. Built with a Raspberry Pi Pico 2 running CircuitPython.

## Operation

**Soil probing mode.** Measures soil moisture once a day.

**Alert mode.** If soil is dry, system switches to alert mode and waits for the ultrasonic sensor to detect someone walking by; a servo then rocks the watering can back and forth as a reminder to water the plant.

## Hardware

- Raspberry Pi Pico 2 running CircuitPython
- Capacitive soil moisture sensor (selected over resistive sensor to mitigate corrosion)
- HC-SR04 ultrasonic distance sensor for motion detection
- 9 g micro servo driving a rocking platform for the watering can

<!-- Video goes here -->

<!-- Photos go here -->

[← Back to projects](../)
